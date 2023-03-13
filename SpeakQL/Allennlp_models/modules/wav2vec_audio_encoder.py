from typing import Iterator, List, Dict, Optional
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

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

from fairseq.models.wav2vec import Wav2VecModel



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


class SpeakQLAudioEncoder(torch.nn.Module, Registrable):
    pass

@SpeakQLAudioEncoder.register('wav2vec')
class Wav2vecAudioEncoder(SpeakQLAudioEncoder):
    def __init__(self,
                 wav2vec_model_path: str,
                 finetune_wav2vec: bool = False,
                 # split_size: int = None,
                 projected_output_size: int = None,
                 cpu: bool = False,
                 ) -> None:
        
        super().__init__()

        self.wav2vec_model_path = wav2vec_model_path
        if cpu:
            cp = torch.load(wav2vec_model_path, map_location=torch.device('cpu'))
        else:
            cp = torch.load(wav2vec_model_path)

        self.wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)
        self.wav2vec_model.load_state_dict(cp['model'])
        self.finetune_wav2vec = finetune_wav2vec

        self.projected_output_size = projected_output_size
        if projected_output_size is None:
            self.projection = None
        else:
            self.projection = torch.nn.Linear(in_features=512,
                                              out_features=projected_output_size)
        # model.eval()

        # Feeding everything once might make it too large
        # Therefore use split_size to split (on dim0) and feed, then combine
        # If None, do not split
        # For now, to make masking correct, fix split_size = 1
        # self.split_size = split_size

        # Configs of wav2vec-large; not sure how to get directly from model
        self.output_size = self.projected_output_size or 512
        self.min_input_length = 640     # 320 looks like the minimum, but might run into some unknown corner cases
        
        self.save_intermediate = False
        self.intermediates = dict()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.save_intermediate = save_intermediate

    def get_intermediates(self):
        return dict(self.intermediates)

    def forward(self,
                audio_tensor,
                audio_mask):

        # audio_tensor: (batch, audio_len, audio_dim=1)
        # audio_mask: (batch, audio_len)

        cuda_device_id = get_device_of(audio_tensor)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id
        
        # if self.split_size is None:
        #     z = self.wav2vec_model.feature_extractor(audio_tensor)
        #     c = self.wav2vec_model.feature_aggregator(z)
        # else:
        #     _audio_tensor_list = torch.split(audio_tensor, self.split_size, dim=0)
        #     _c_list = [self.wav2vec_model.feature_aggregator(self.wav2vec_model.feature_extractor(_audio_tensor))
        #         for _audio_tensor in _audio_tensor_list]
        #     c = torch.cat(_c_list, dim=0)

        audio_lens = audio_mask.sum(-1).detach().cpu().numpy().tolist()
        assert max(audio_lens) >= self.min_input_length, \
            f'max(audio_lens) = {max(audio_lens)}, but self.min_input_length = {self.min_input_length}'

        # audio_tensor_list = torch.split(audio_tensor, 1, dim=0)

        # c_list = List[_c]
        # _c: (out_len, out_dim=512)
        c_list = []

        with torch.set_grad_enabled(self.finetune_wav2vec):
            for i, _len in enumerate(audio_lens):
                _len = max(_len, self.min_input_length)

                _audio_tensor = audio_tensor[i, :_len].view(1, -1)
                # _z: (1, out_dim=512, out_len)
                _z = self.wav2vec_model.feature_extractor(_audio_tensor)
                # _c: (1, out_dim=512, out_len)
                _c = self.wav2vec_model.feature_aggregator(_z)
                # _c: (out_len, out_dim=512)
                _c = _c.squeeze(0).transpose(0, 1).contiguous()
                if self.projection is not None:
                    # _c = (out_len, out_dim=projected_out_dim)
                    _c = self.projection(_c)

                c_list.append(_c)

        out_audio_lens = [_c.size(0) for _c in c_list]
        # out_c: (batch, max_out_len, out_dim)
        out_c = pad_sequence(c_list, batch_first=True)
        # out_mask: (batch, max_out_len)
        out_mask = get_mask_from_sequence_lengths(torch.LongTensor(out_audio_lens), max_length=max(out_audio_lens)).to(device=cuda_device)

        assert out_c.size()[:2] == out_mask.size(), f'{out_c.size()} {out_mask.size()}'

        return out_c, out_mask

    def get_output_dim(self):
        return self.output_size













