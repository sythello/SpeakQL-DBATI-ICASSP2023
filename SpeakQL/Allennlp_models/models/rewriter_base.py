from typing import Iterator, List, Dict, Union, Optional, cast
import torch
import torch.optim as optim
from torch import nn
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn import ModuleList

import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField, MetadataField, ListField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
# from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.attention import Attention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention

from allennlp.modules.conditional_random_field import allowed_transitions, ConditionalRandomField

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    get_device_of, masked_softmax, masked_log_softmax, weighted_sum, \
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal

from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, MeanAbsoluteError, Average
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.dataloader import DataLoader
from allennlp.training.trainer import GradientDescentTrainer
# from allennlp.predictors import Predictor, Seq2SeqPredictor, SimpleSeq2SeqPredictor, SentenceTaggerPredictor
from allennlp.predictors import Predictor, SentenceTaggerPredictor
from allennlp.nn.activations import Activation
from allennlp.common.tqdm import Tqdm
from allennlp.common.params import Params
from allennlp.common.util import JsonDict, sanitize

from allennlp_models.generation.predictors import Seq2SeqPredictor
from allennlp_models.generation.models.simple_seq2seq import SimpleSeq2Seq
from allennlp_models.generation.modules.seq_decoders.seq_decoder import SeqDecoder
from allennlp_models.generation.modules.decoder_nets.decoder_net import DecoderNet


# from spacy.tokenizer import Tokenizer as SpacyTokenizer
# from spacy.lang.en import English
# nlp = English()
# Create a blank Tokenizer with just the English vocab
# tokenizer = Tokenizer(nlp.vocab)

from tqdm import tqdm

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

import os
import itertools
import json
from collections import defaultdict
from inspect import signature
import warnings
import pickle
from copy import copy, deepcopy
from overrides import overrides

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from utils.spider import process_sql, evaluation
# from utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema
from table_bert import TableBertModel

from modules.encoder import SpeakQLEncoder, SpeakQLEncoderV1
from modules.tabert_embedder import TaBERTEmbedder


PHONEME_VOCAB_NAMESPACE = "phonemes"
PHONEME_INDEXER_KEY = "phonemes"


class SpiderASRFixer_Base(Model):
    def __init__(self,
                 src_text_embedder: TextFieldEmbedder = None,
                 use_db_input: bool = True,     ## Ablation: -DB (implemented in reader, currently not needed for model)
                 use_audio: bool = True,        ## Ablation: -audio
                 use_tabert: bool = False,
                 tabert_model_path: str = None,
                 finetune_tabert: bool = False,
                 audio_seq2vec_encoder: Seq2VecEncoder = None,
                 encoder: SpeakQLEncoder = None,
                 using_gated_fusion: bool = False,      ## TODO: combine the code of using different encoders
                 using_ref_att_loss: bool = False,
                 ref_att_loss_coef: float = 1000.0,     ## TODO: schedule to shrink over epochs
                 ref_att_loss_coef_schedule: Dict = None,   ## keys: decay_rate, decay_value, min_value
                 concat_audio: bool = True,
                 using_modal_proj: bool = False,
                 token_proj_out_dim: int = 256,
                 audio_proj_out_dim: int = 128,
                 using_layer_norm: bool = False,
                 using_phoneme_input: bool = False,
                 ph2tok_audio_seq2vec_encoder: Seq2VecEncoder = None,
                 using_phoneme_labels: bool = False,        ## DIFFERENT from reader: this means using phoneme-level label loss
                 phoneme_loss_coef: float = 1.0,
                 using_phoneme_multilabels: bool = False,   ## DIFFERENT from reader: this means using token-level multilabel loss
                 phoneme_multilabel_loss_coef: float = 1.0,
                 ## NEW: copy mech
                 using_copy_mechanism: bool = False,    ## If true, need to use CopyNetSeqDecoder
                 aux_probe_configs: Dict = None,    # Keys: "utter_mention_schema[_coef]", "schema_dir_mentioned[_coef]", "schema_indir_mentioned[_coef]"
                 ## optional feats
                 align_tag_embedder: TokenEmbedder = None,
                 rewriter_tag_embedder: TokenEmbedder = None,
                 phoneme_tag_embedder: TextFieldEmbedder = None,    ## TextField to workaround the padding=-1 problem with LabelField
                 phoneme_tag_seq2vec_encoder: Seq2VecEncoder = None,
                 vocab: Vocabulary = None) -> None:
        super().__init__(vocab)
        self.src_text_embedder = src_text_embedder
        self.align_tag_embedder = align_tag_embedder
        self.rewriter_tag_embedder = rewriter_tag_embedder
        self.phoneme_tag_embedder = phoneme_tag_embedder
        self.phoneme_tag_seq2vec_encoder = phoneme_tag_seq2vec_encoder

        self.use_db_input = use_db_input
        self.use_audio = use_audio

        self.using_copy_mechanism = using_copy_mechanism

        self.use_tabert = use_tabert
        self.tabert_model_path = tabert_model_path
        self.finetune_tabert = finetune_tabert
        if use_tabert:
            # self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)
            self.tabert_embedder = TaBERTEmbedder(tabert_model_path, self.finetune_tabert)
        
        self.audio_seq2vec_encoder = audio_seq2vec_encoder
        self.encoder = encoder
        self.using_gated_fusion = using_gated_fusion
        self.using_ref_att_loss = using_ref_att_loss
        self.ref_att_loss_coef = ref_att_loss_coef
        self.ref_att_loss_coef_schedule = ref_att_loss_coef_schedule

        self.concat_audio = concat_audio

        # Whether use an extra separate projection module for different modals (text/schema, token/audio)
        if use_tabert:
            token_enc_dim = self.tabert_embedder.tabert_model.output_size
        else:
            token_enc_dim = self.src_text_embedder.get_output_dim()
        if use_audio:
            assert self.audio_seq2vec_encoder is not None
            audio_enc_dim = self.audio_seq2vec_encoder.get_output_dim()
        else:
            audio_enc_dim = None

        self.using_modal_proj = using_modal_proj
        self.using_layer_norm = using_layer_norm
        if using_modal_proj:
            self.token_proj_out_dim = token_proj_out_dim
            self.audio_proj_out_dim = audio_proj_out_dim

            if using_layer_norm:            
                self.text_token_proj = nn.Sequential(
                    nn.Linear(token_enc_dim, token_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    # nn.Linear(token_proj_out_dim, token_proj_out_dim),
                    nn.LayerNorm(token_proj_out_dim))
                self.schema_token_proj = nn.Sequential(
                    nn.Linear(token_enc_dim, token_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    # nn.Linear(token_proj_out_dim, token_proj_out_dim),
                    nn.LayerNorm(token_proj_out_dim))
                self.text_audio_proj = nn.Sequential(
                    nn.Linear(audio_enc_dim, audio_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    # nn.Linear(audio_proj_out_dim, audio_proj_out_dim),
                    nn.LayerNorm(audio_proj_out_dim))
                self.schema_audio_proj = nn.Sequential(
                    nn.Linear(audio_enc_dim, audio_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                    # nn.Linear(audio_proj_out_dim, audio_proj_out_dim),
                    nn.LayerNorm(audio_proj_out_dim))
            else:
                self.text_token_proj = nn.Sequential(
                    nn.Linear(token_enc_dim, token_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1))
                self.schema_token_proj = nn.Sequential(
                    nn.Linear(token_enc_dim, token_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1))
                self.text_audio_proj = nn.Sequential(
                    nn.Linear(audio_enc_dim, audio_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1))
                self.schema_audio_proj = nn.Sequential(
                    nn.Linear(audio_enc_dim, audio_proj_out_dim),
                    nn.LeakyReLU(negative_slope=0.1))

        else:
            ## No projection, seen as identity projection
            self.token_proj_out_dim = token_enc_dim
            self.audio_proj_out_dim = audio_enc_dim

            if using_layer_norm:
                self.token_norm = nn.LayerNorm(token_enc_dim)
                self.audio_norm = nn.LayerNorm(audio_enc_dim)


        self.using_phoneme_input = using_phoneme_input
        self.using_phoneme_labels = using_phoneme_labels
        self.ph2tok_audio_seq2vec_encoder = ph2tok_audio_seq2vec_encoder
        self.phoneme_loss_coef = phoneme_loss_coef
        self.using_phoneme_multilabels = using_phoneme_multilabels
        self.phoneme_multilabel_loss_coef = phoneme_multilabel_loss_coef

        if using_phoneme_labels:
            assert using_phoneme_input, "Must have phoneme-level input to use phoneme labels"
        if using_phoneme_labels or using_phoneme_multilabels:
            phonemes_size = vocab.get_vocab_size(namespace=PHONEME_VOCAB_NAMESPACE)  # Hard-coded
            self.phoneme_pred_head = nn.Linear(audio_enc_dim, phonemes_size)
        else:
            self.phoneme_pred_head = None

        ## Setting probes 
        self.aux_probe_configs = aux_probe_configs
        self.utter_mention_schema_head = None
        self.utter_mention_schema_coef = None
        self.schema_dir_mentioned_head = None
        self.schema_dir_mentioned_coef = None
        self.schema_indir_mentioned_head = None
        self.schema_indir_mentioned_coef = None
        if aux_probe_configs is not None:
            if aux_probe_configs['utter_mention_schema']:
                self.utter_mention_schema_head = nn.Linear(encoder.get_output_dim(), 1)
                self.utter_mention_schema_coef = aux_probe_configs['utter_mention_schema_coef']
            if aux_probe_configs['schema_dir_mentioned']:
                self.schema_dir_mentioned_head = nn.Linear(encoder.get_output_dim(), 1)
                self.schema_dir_mentioned_coef = aux_probe_configs['schema_dir_mentioned_coef']
            if aux_probe_configs['schema_indir_mentioned']:
                self.schema_indir_mentioned_head = nn.Linear(encoder.get_output_dim(), 1)
                self.schema_indir_mentioned_coef = aux_probe_configs['schema_indir_mentioned_coef']

        # Some sequence representation-based loss metrics
        self.att_KL = Average()
        self.ph_NLL = Average()
        self.ph_multi_NLL = Average()
        self.u_ms_NLL = Average()
        self.s_dm_NLL = Average()
        self.s_im_NLL = Average()

        self.save_intermediate = False
        self.intermediates = dict()

    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.encoder.set_save_intermediate(save_intermediate)
        self.save_intermediate = save_intermediate

    def get_intermediates(self):
        return {
            'encoder': self.encoder.intermediates,
            'rewriter_main': self.intermediates
        }

    def _basic_encoding(self,
                        sentence: Dict[str, torch.Tensor],
                        text_mask: torch.Tensor,
                        schema_mask: torch.Tensor,
                        schema_column_ids: torch.Tensor,    # For each schema token, which column it belongs to (starting at 1), 0 if not column
                        audio_feats: torch.Tensor,
                        audio_mask: torch.Tensor,
                        phoneme_audio_feats: torch.Tensor,
                        phoneme_labels: Dict[str, torch.Tensor],    # also from a TextField
                        phoneme_multilabels: torch.Tensor,
                        phoneme_audio_mask: torch.Tensor,
                        phoneme_label_mask: torch.Tensor,
                        metadata: List[Dict],
                        align_tags: torch.Tensor = None,
                        rewriter_tags: torch.Tensor = None) -> Dict:

        # audio_feats: (batch, seq_len, audio_len, audio_dim)
        # audio_mask: (batch, seq_len, audio_len)
        # text_mask: (batch, seq_len)
        # phoneme_audio_feats: (batch, seq_len, ph_len, ph_audio_len, audio_dim)
        # phoneme_labels: (batch, seq_len, ph_len)
        # phoneme_audio_mask: (batch, seq_len, ph_len, ph_audio_len)
        # phoneme_label_mask: (batch, seq_len, ph_len)

        # Not sure if this is the right way
        cuda_device_id = get_device_of(audio_feats)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        batch_size, seq_len, audio_len, audio_dim = audio_feats.size()
        assert batch_size == len(metadata)
        
        text_lens = [_meta['text_len'] for _meta in metadata]
        schema_lens = [_meta['schema_len'] for _meta in metadata]
        # rewrite_seq_lens = [_meta['rewrite_seq_len'] for _meta in metadata]
        # concat_lens = [text_lens[i] + 1 + schema_lens[i] for i in range(batch_size)]
        concat_lens = [_meta['concat_len'] for _meta in metadata]
        concat_mask = get_mask_from_sequence_lengths(torch.LongTensor(concat_lens), max_length=seq_len).to(device=cuda_device)
        text_mask = get_mask_from_sequence_lengths(torch.LongTensor(text_lens), max_length=seq_len).to(device=cuda_device)

        _non_schema_lens = [concat_lens[i] - schema_lens[i] for i in range(batch_size)]
        _non_schema_mask = get_mask_from_sequence_lengths(torch.LongTensor(_non_schema_lens), max_length=seq_len).to(device=cuda_device)
        schema_mask = torch.logical_xor(concat_mask, _non_schema_mask)
        
        # Get sentence (text + schema) mask
        mask = get_text_field_mask(sentence)
        # mask: (batch, seq_len)
        assert tensors_equal(concat_mask, mask), '{}\n{}'.format(concat_mask, mask)
        assert text_mask.size() == schema_mask.size() == concat_mask.size() == mask.size(), \
            '{}\n{}\n{}\n{}'.format(text_mask, schema_mask, concat_mask, mask)
        
        input_embeddings = []

        ## (Optional) using TaBERT embeddings
        if self.use_tabert:
            ## TaBERT encoding
            tabert_dim = self.tabert_embedder.tabert_model.output_size

            schema_len_padded = schema_column_ids.size(1)
            assert schema_len_padded == max(schema_lens), f'{schema_len_padded} | {schema_lens}'

            # tabert_sentence_embedding: (batch, seq_len, tabert_dim)
            tabert_sentence_embedding = self.tabert_embedder(schema_column_ids, metadata)

            assert tuple(tabert_sentence_embedding.size()) == (batch_size, seq_len, tabert_dim), \
                f'{tuple(tabert_sentence_embedding.size())} | {(batch_size, seq_len, tabert_dim)}'

            input_embeddings.append(tabert_sentence_embedding)

        ## (Optional) other embeddings than TaBERT
        if self.src_text_embedder is not None:
            # src_text_embedding: (batch, seq_len, emb_dim)
            try:
                src_text_embedding = self.src_text_embedder(sentence)
            except Exception as e:
                print('metadata:')
                print(metadata)
                print('sentence:')
                print(sentence)
                raise e

            input_embeddings.append(src_text_embedding)
        
        # word_embeddings: (batch, seq_len, emb_dim)
        word_embeddings = torch.cat(input_embeddings, dim=-1)
        self._maybe_save(word_embeddings.detach().cpu().numpy(), 'word_embeddings')
        if self.using_modal_proj:
            ## With proj, with/without layer norm
            word_embeddings = \
                self.text_token_proj(word_embeddings) * text_mask.unsqueeze(-1) + \
                self.schema_token_proj(word_embeddings) * schema_mask.unsqueeze(-1)
            self._maybe_save(word_embeddings.detach().cpu().numpy(), 'word_embeddings_projected')
        elif self.using_layer_norm:
            ## No proj, with layer norm
            word_embeddings = self.token_norm(word_embeddings)
            self._maybe_save(word_embeddings.detach().cpu().numpy(), 'word_embeddings_normalized')


        if not self.use_audio:
            # Ablation: not using audio
            audio_feats_encoded = None
            ph_audio_feats_encoded = None
        elif not self.using_phoneme_input:
            # Audio seq2vec encoding
            audio_feats_enc_in = audio_feats.view(batch_size * seq_len, audio_len, audio_dim)
            audio_mask_enc_in = audio_mask.view(batch_size * seq_len, audio_len)
            # audio_feats_enc_out: (batch_size * seq_len, audio_enc_out_dim)
            audio_feats_enc_out = self.audio_seq2vec_encoder(audio_feats_enc_in, audio_mask_enc_in)
            audio_feats_encoded = audio_feats_enc_out.view(batch_size, seq_len, -1)
            ph_audio_feats_encoded = None
        else:
            _batch_size, _seq_len, ph_len, ph_audio_len, audio_dim = phoneme_audio_feats.size()
            # print(_batch_size, _seq_len, ph_len, ph_audio_len, audio_dim)
            assert _batch_size == batch_size, (audio_feats.size(), phoneme_audio_feats.size())
            assert _seq_len == seq_len, (audio_feats.size(), phoneme_audio_feats.size())
            # Audio seq2vec encoding
            
            ## This (last line) might cause CUDA OOM
            # ph_audio_feats_enc_in = phoneme_audio_feats.view(batch_size * seq_len * ph_len, ph_audio_len, audio_dim)
            # ph_audio_mask_enc_in = phoneme_audio_mask.view(batch_size * seq_len * ph_len, ph_audio_len)
            # ph_audio_feats_enc_out = self.audio_seq2vec_encoder(ph_audio_feats_enc_in, ph_audio_mask_enc_in)

            ph_audio_feats_enc_in = phoneme_audio_feats.view(batch_size, seq_len * ph_len, ph_audio_len, audio_dim)
            ph_audio_mask_enc_in = phoneme_audio_mask.view(batch_size, seq_len * ph_len, ph_audio_len)

            _enc_outs = []
            for i in range(batch_size):
                _feats_enc_in = ph_audio_feats_enc_in[i]
                _mask_enc_in = ph_audio_mask_enc_in[i]
                # _enc_out: (1, seq_len * ph_len, audio_enc_out_dim)
                _enc_out = self.audio_seq2vec_encoder(_feats_enc_in, _mask_enc_in).unsqueeze(0)
                _enc_outs.append(_enc_out)
            
            # ph_audio_feats_enc_out: (batch_size, seq_len * ph_len, audio_enc_out_dim)
            ph_audio_feats_enc_out = torch.cat(_enc_outs, dim=0)
            ph_audio_feats_encoded = ph_audio_feats_enc_out.view(batch_size * seq_len, ph_len, -1)

            # ph_audio_mask_2: for ph2tok encoding; assuming phoneme pos 0 stands for token on/off
            ph_audio_mask_2 = phoneme_audio_mask.view(batch_size * seq_len, ph_len, ph_audio_len)[:, :, 0]
            audio_feats_enc_out = self.ph2tok_audio_seq2vec_encoder(ph_audio_feats_encoded, ph_audio_mask_2)
            audio_feats_encoded = audio_feats_enc_out.view(batch_size, seq_len, -1)

        if self.use_audio:
            self._maybe_save(audio_feats_encoded.detach().cpu().numpy(), 'audio_feats_encoded')
        if self.using_modal_proj:
            ## With proj, with/without layer norm
            audio_feats_encoded = \
                self.text_audio_proj(audio_feats_encoded) * text_mask.unsqueeze(-1) + \
                self.schema_audio_proj(audio_feats_encoded) * schema_mask.unsqueeze(-1)
            self._maybe_save(audio_feats_encoded.detach().cpu().numpy(), 'audio_feats_encoded_projected')
        elif self.using_layer_norm:
            ## No proj, with layer norm
            audio_feats_encoded = self.audio_norm(audio_feats_encoded)
            self._maybe_save(audio_feats_encoded.detach().cpu().numpy(), 'audio_feats_encoded_normalized')


        # Embed align tags if given
        if align_tags is not None:
            # print('align_tags:')
            # print(align_tags)
            # print(align_tags.size())
            align_tag_embeddings = self.align_tag_embedder(align_tags)
            # print('align_tag_embeddings:')
            # print(align_tag_embeddings.size())
        else:
            align_tag_embeddings = None

        # Embed rewriter tags if given
        if rewriter_tags is not None:
            # print('rewriter_tags:')
            # print(rewriter_tags)
            # print(rewriter_tags.size())
            rewriter_tag_embeddings = self.rewriter_tag_embedder(rewriter_tags)
            # print('rewriter_tag_embeddings:')
            # print(rewriter_tag_embeddings.size())
        else:
            rewriter_tag_embeddings = None

        # Embed phoneme labels if given and embedder specified
        if phoneme_labels is not None and self.phoneme_tag_embedder is not None:
            # (batch, seq_len, ph_len, emb_dim)
            try:
                ## TextFieldEmbedder (TextField to workaround the padding=-1 problem with LabelField)
                phoneme_tag_embeddings_single = self.phoneme_tag_embedder(phoneme_labels)
            except Exception as e:
                print('phoneme_labels:')
                print(phoneme_labels)
                print(phoneme_labels['phonemes'])
                print(phoneme_labels['phonemes'].size())
                print(self.vocab)
                raise e

            _batch_size, _seq_len, _ph_len, _emb_dim = phoneme_tag_embeddings_single.size()
            assert (_batch_size == batch_size) and (_seq_len == seq_len)
            phoneme_tag_embeddings_single_viewed = phoneme_tag_embeddings_single.view(_batch_size * _seq_len, _ph_len, _emb_dim)
            phoneme_label_mask_viewed = phoneme_label_mask.view(_batch_size * _seq_len, _ph_len)

            # (batch * seq_len, emb_dim)
            phoneme_tag_embeddings_viewed = self.phoneme_tag_seq2vec_encoder(phoneme_tag_embeddings_single_viewed, phoneme_label_mask_viewed)
            phoneme_tag_embeddings = phoneme_tag_embeddings_viewed.view(_batch_size, _seq_len, _emb_dim)
            self._maybe_save(phoneme_tag_embeddings.detach().cpu().numpy(), 'phoneme_tag_embeddings')
        else:
            phoneme_tag_embeddings = None

        
        return {
            "text_lens": text_lens,
            "schema_lens": schema_lens,
            "concat_lens": concat_lens,
            "mask": mask,
            "text_mask": text_mask,
            "schema_mask": schema_mask,
            "word_embeddings": word_embeddings,
            "audio_feats_encoded": audio_feats_encoded,
            "ph_audio_feats_encoded": ph_audio_feats_encoded,
            "align_tag_embeddings": align_tag_embeddings,
            "rewriter_tag_embeddings": rewriter_tag_embeddings,
            "phoneme_tag_embeddings": phoneme_tag_embeddings,
        }

