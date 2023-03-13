from typing import Iterator, List, Dict, Optional
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn import ModuleList

import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField, MetadataField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
# from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.attention import Attention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    get_device_of, masked_softmax, weighted_sum, \
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal

from allennlp.training.metrics import CategoricalAccuracy, MeanAbsoluteError
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataloader import DataLoader
from allennlp.training.trainer import Trainer
# from allennlp.predictors import Predictor, Seq2SeqPredictor, SimpleSeq2SeqPredictor, SentenceTaggerPredictor
from allennlp.predictors import Predictor, SentenceTaggerPredictor
from allennlp.nn.activations import Activation
from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.common.util import JsonDict, sanitize

from allennlp_models.generation.predictors import Seq2SeqPredictor
from allennlp_models.generation.models.simple_seq2seq import SimpleSeq2Seq



from tqdm import tqdm

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

import os
import itertools
import json
from collections import defaultdict
from inspect import signature

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from utils.spider import process_sql


class SpeakQLEncoder(torch.nn.Module, Registrable):
    def __init__(self,
                 ## START Deprecated - attention_type specific args
                 audio_attention_layer: MatrixAttention = None,
                 audio_attention_residual: str = '+',
                 num_audio_attention_layers: int = 1,
                 full_attention_layer: MatrixAttention = None,
                 full_attention_residual: str = '+',
                 num_full_attention_layers: int = 1,
                 ## END Deprecated
                 attention_type: str = None,    # ['audio', 'full', 'phoneme']  TODO: audio-phoneme
                 attention_layer: MatrixAttention = None,
                 attention_residual: str = '+',
                 num_attention_layers: int = 1,
                 seq2seq_encoders: List[Seq2SeqEncoder] = [],
                 seq2vec_encoder: Seq2VecEncoder = None) -> None:
        
        super().__init__()
        self.audio_attention_layer = audio_attention_layer
        self.audio_attention_residual = audio_attention_residual
        self.num_audio_attention_layers = num_audio_attention_layers
        self.full_attention_layer = full_attention_layer
        self.full_attention_residual = full_attention_residual
        self.num_full_attention_layers = num_full_attention_layers
        if self.audio_attention_layer is not None or self.full_attention_layer is not None:
            print('WARNING: SpeakQLEncoder: audio_attention_layer and full_attention_layer are deprecated')

        self.attention_type = attention_type
        self.attention_layer = attention_layer
        self.attention_residual = attention_residual
        self.num_attention_layers = num_attention_layers

        self.seq2seq_encoders = ModuleList([] if seq2seq_encoders is None else seq2seq_encoders)
        self.seq2vec_encoder = seq2vec_encoder

        assert (len(seq2seq_encoders) > 0) or (seq2vec_encoder is not None)

        self.seq2seq_is_bidirectional = False if len(seq2seq_encoders) == 0 else seq2seq_encoders[-1].is_bidirectional()

        self.save_intermediate = False
        self.intermediates = dict()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.save_intermediate = save_intermediate

    def get_intermediates(self):
        return dict(self.intermediates)

    def _do_attentions(self,
                    encoder_in,
                    attention_feats_dict,   # Dict[str, Tensor]
                    mask):

        att_map_logits = None
        att_map = None

        ## Find (and infer) the correct setting of all params, back-compatible
        attention_type = self.attention_type        
        attention_layer = self.attention_layer
        attention_residual = self.attention_residual
        num_attention_layers = self.num_attention_layers
        if attention_layer is None:
            if self.audio_attention_layer is not None:
                assert attention_type in [None, 'audio'], attention_type
                attention_type = 'audio'
                attention_layer = self.audio_attention_layer
                attention_residual = self.audio_attention_residual
                num_attention_layers = self.num_audio_attention_layers
            elif self.full_attention_layer is not None:
                assert attention_type in [None, 'full'], attention_type
                attention_type = 'full'
                attention_layer = self.full_attention_layer
                attention_residual = self.full_attention_residual
                num_attention_layers = self.num_full_attention_layers

        if attention_type in ['audio', 'full', 'phoneme']:
            attention_feats = attention_feats_dict[attention_type]
        elif attention_type is None:
            pass
        else:
            raise NotImplementedError(attention_type)

        assert (attention_type is None) == (attention_layer is None), (attention_type, attention_layer)
        
        # Perform attention if specified 
        if attention_layer is not None:
            for i in range(num_attention_layers):
                att_map_logits = attention_layer(attention_feats, attention_feats)
                # att_map: (batch, seq_len, seq_len)
                att_map = masked_softmax(att_map_logits, mask)
                att_out = weighted_sum(encoder_in, att_map)

                self._maybe_save(att_map.detach().cpu().numpy(), f'{attention_type}_attention_map_{i}')
                self._maybe_save(att_out.detach().cpu().numpy(), f'{attention_type}_attention_out_{i}')

                if attention_residual is None:
                    encoder_in = att_out
                elif attention_residual == ';':
                    # encoder_in = torch.cat([token_feats, att_out], dim=-1)
                    assert num_attention_layers == 1, 'Currently do not support residual = ";" with num_audio_attention_layers > 1'
                    encoder_in = torch.cat([encoder_in, att_out], dim=-1)
                elif attention_residual == '+':
                    encoder_in = encoder_in + att_out
                else:
                    raise ValueError('attention_residual "{}" not defined'.format(attention_residual))

                self._maybe_save(encoder_in.detach().cpu().numpy(), f'{attention_type}_attention_out_with_residual_{i}')
        else:
            # No attention layer
            pass

        return {
            "encodings": encoder_in,
            "att_map_logits": att_map_logits,
            "att_map": att_map
        }


    def _do_seq_encodings(self,
                        encoder_in,
                        mask):

        # Perform other s2s and s2v encodings 
        all_encoders = list(self.seq2seq_encoders)
        if self.seq2vec_encoder is not None:
            all_encoders.append(self.seq2vec_encoder)

        assert encoder_in.size(-1) == all_encoders[0].get_input_dim(), (encoder_in.size(), all_encoders[0].get_input_dim())
        for i in range(1, len(all_encoders)):
            assert all_encoders[i-1].get_output_dim() == all_encoders[i].get_input_dim(), \
                '{}: {} (output_dim: {}) || {} (input_dim: {})'.format(i, all_encoders[i-1], all_encoders[i-1].get_output_dim(), all_encoders[i], all_encoders[i].get_input_dim())
        
        for i, s2s_encoder in enumerate(self.seq2seq_encoders):
            encoder_in = s2s_encoder(encoder_in, mask)
            self._maybe_save(encoder_in.detach().cpu().numpy(), 'encoder_representation_{}'.format(i))
        
        encoder_seq_repr = encoder_in
        if self.seq2vec_encoder is not None:
            encoder_vec_out = self.seq2vec_encoder(encoder_in, mask)
        else:
            encoder_vec_out = None

        return {
            "vec_repr": encoder_vec_out,
            "seq_repr": encoder_seq_repr
        }

    def forward(self,
                token_feats,
                audio_feats=None,
                # attention_type=None,
                attention_feats_dict=None,
                mask=None):
        raise NotImplementedError()
        
    def get_output_dim(self):
        if self.seq2vec_encoder is not None:
            return self.seq2vec_encoder.get_output_dim()
        else:
            return self.seq2seq_encoders[-1].get_output_dim()


@SpeakQLEncoder.register('v1')
class SpeakQLEncoderV1(SpeakQLEncoder):
    ''' Back-compatible encoder. Can specify attention type or use deprecated audio/full_attention_layer'''

    def forward(self,
                token_feats,
                audio_feats=None,
                # attention_type=None,
                attention_feats_dict=None,
                mask=None):

        # if attention_type is None:
        #     # Infer from args
        #     attention_type = 'audio' if self.audio_attention_layer is not None else 'full'
        if attention_feats_dict is None:
            attention_feats_dict = {'audio': audio_feats, 'full': token_feats}

        att_out_dict = self._do_attentions(
            encoder_in=token_feats,
            # attention_type=attention_type,
            attention_feats_dict=attention_feats_dict,
            mask=mask)

        seq_enc_out_dict = self._do_seq_encodings(
            encoder_in=att_out_dict["encodings"],
            mask=mask)

        return {
            "vec_repr": seq_enc_out_dict["vec_repr"],
            "seq_repr": seq_enc_out_dict["seq_repr"],
            "att_map_logits": att_out_dict["att_map_logits"],
            "att_map": att_out_dict["att_map"]
        }


