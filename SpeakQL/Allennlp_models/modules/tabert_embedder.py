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
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal, \
    batched_span_select

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

from table_bert import TableBertModel


class TaBERTEmbedder(torch.nn.Module):
    def __init__(self,
                 tabert_model_path: str,
                 finetune_tabert: bool):
        super().__init__()

        self.tabert_model_path = tabert_model_path
        self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)
        self.finetune_tabert = finetune_tabert

    def forward(self,
                schema_column_ids: torch.Tensor,
                metadata: List[Dict],
                seq_len: int = None):
        cuda_device_id = get_device_of(schema_column_ids)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        tabert_dim = self.tabert_model.output_size

        text_lens = [_meta['text_len'] for _meta in metadata]
        schema_lens = [_meta['schema_len'] for _meta in metadata]
        if seq_len is None:
            seq_len = max([_t + 1 + _s for _t, _s in zip(text_lens, schema_lens)])

        tabert_sentence_embeddings = []
        for idx, _meta in enumerate(metadata):
            # Enumerate over samples in batch; for each sample, encode (text, table) for each table and mean-pooling for text encoding
            _text_len = _meta['text_len']
            _schema_len = _meta['schema_len']

            _text_tokenized = _meta['text_tokenized']
            _offsets = _meta['text_offsets']

            # print(f"--- CP 1.{idx}.0: _text_tokenized = {' '.join(_text_tokenized)} -- {len(_text_tokenized)}")

            _tables = _meta['tabert_tables']
            _tables_list = list(_tables.values())
            for _tbl in _tables_list:
                assert _tbl.tokenized
                # print(f'Table: {_tbl.id}')
                # print(f'Header: {_tbl.header}')
                # print(f'Data:\n{_tbl.data}')
            
            # print(f'--- CP 1.{idx}.1: len(_tables) = {len(_tables)}, _text_len = {_text_len} _schema_len = {_schema_len}')
            # print(f"text tokens ({_text_len}): {_meta['source_tokens'][:_text_len]}")
            # print(f"schema tokens ({_schema_len}): {_meta['source_tokens'][_text_len + 1 : _schema_len]}")
            
            if self.finetune_tabert:
                # _text_encoding: (n_tables, text_piece_len, tabert_dim)
                # _column_encoding: (n_tables, n_columns, tabert_dim)
                _text_encoding, _column_encoding, _info_dict = self.tabert_model.encode(
                    contexts=[_text_tokenized] * len(_tables),
                    tables=_tables_list)
            else:
                with torch.no_grad():
                    _text_encoding, _column_encoding, _info_dict = self.tabert_model.encode(
                        contexts=[_text_tokenized] * len(_tables),
                        tables=_tables_list)

            # print(f'--- CP 1.{idx}.2: _text_encoding.size() = {_text_encoding.size()}, _column_encoding.size() = {_column_encoding.size()}')
            
            ## Copied from allennlp::pretrained_transformer_mismatched_embedder
            # _offsets_tensor: (n_tables, text_len, 2)
            _offsets_tensor = torch.LongTensor([_offsets] * len(_tables)).to(device=cuda_device)
            # _span_embeddings: (n_tables, text_len, max_span_len, tabert_dim)
            # _span_mask: (n_tables, text_len, max_span_len)
            _span_embeddings, _span_mask = batched_span_select(_text_encoding.contiguous(), _offsets_tensor)
            # print(f'_span_embeddings.size() = {_span_embeddings.size()}')
            # print(f'_span_mask.size() = {_span_mask.size()}')
            _span_mask = _span_mask.unsqueeze(-1)
            _span_embeddings *= _span_mask
            _span_embeddings_sum = _span_embeddings.sum(2)
            _span_embeddings_len = _span_mask.sum(2)
            _orig_embeddings = _span_embeddings_sum / torch.clamp_min(_span_embeddings_len, 1)
            _orig_embeddings[(_span_embeddings_len == 0).expand(_orig_embeddings.shape)] = 0

            # _text_encoding_span_pooled: (n_tables, text_len, tabert_dim)
            _text_encoding_span_pooled = _orig_embeddings
            # print(f'_text_encoding_span_pooled.size() = {_text_encoding_span_pooled.size()}')

            # _text_encoding_pooled: (text_len, tabert_dim)
            _text_encoding_pooled = _text_encoding_span_pooled.mean(dim=0)
            assert tuple(_text_encoding_pooled.size()) == (_text_len, tabert_dim), \
                f'{tuple(_text_encoding_pooled.size())} | {(_text_len, tabert_dim)}'

            # print(f'--- CP 1.{idx}.3: _text_encoding_pooled.size() = {_text_encoding_pooled.size()}')

            # print(_info_dict.keys())
            # print(_info_dict)

            # Need test!
            # _column_mask: (n_tables, n_columns)
            _column_mask = _info_dict['tensor_dict']['column_mask'].to(dtype=bool)
            # _all_columns_encoding: (n_all_columns, tabert_dim)
            _all_columns_encoding = _column_encoding[_column_mask, :]
            # _all_columns_encoding: (1 + n_all_columns, tabert_dim)
            _all_columns_encoding = torch.cat([torch.zeros([1, tabert_dim], dtype=torch.float32, device=cuda_device), _all_columns_encoding], dim=0)

            # schema_column_ids: (batch, schema_len_padded)
            # assert _all_columns_encoding.size(0) - 1 == schema_column_ids[idx].max() == sum([len(_tbl.header) for _tbl in _tables_list]), \
            #     f'{_all_columns_encoding.size(0) - 1}\t{schema_column_ids[idx].max()}\t{sum([len(_tbl.header) for _tbl in _tables_list])}'
            ## Assert not true, because long schema can be truncated, in which cases schema_column_ids[idx].max() will be smaller


            # _schema_column_ids_sliced: (schema_len,)
            _schema_column_ids_sliced = schema_column_ids[idx][:_schema_len]

            # _schema_encoding: (schema_len, tabert_dim)
            _schema_encoding = torch.index_select(_all_columns_encoding, dim=0, index=_schema_column_ids_sliced)
            assert tuple(_schema_encoding.size()) == (_schema_len, tabert_dim), \
                f'{tuple(_schema_encoding.size())} | {(_schema_len, tabert_dim)}'

            # tabert_text_embeddings.append(_text_encoding_pooled)
            # tabert_schema_embeddings.append(_schema_encoding)

            _padding_len = seq_len - (_text_len + 1 + _schema_len)

            # _tabert_sentence_embedding: (seq_len, tabert_dim)
            _tabert_sentence_embedding = torch.cat([
                _text_encoding_pooled,
                torch.zeros([1, tabert_dim], dtype=torch.float32, device=cuda_device),
                _schema_encoding,
                torch.zeros([_padding_len, tabert_dim], dtype=torch.float32, device=cuda_device)
            ], dim=0)

            assert tuple(_tabert_sentence_embedding.size()) == (seq_len, tabert_dim), \
                f'{tuple(_tabert_sentence_embedding.size())} | {(seq_len, tabert_dim)}'

            tabert_sentence_embeddings.append(_tabert_sentence_embedding)

            # print(f'--- CP 1.{idx}.4: _tabert_sentence_embedding.size() = {_tabert_sentence_embedding.size()}')

        # (batch, *_len, tabert_dim)
        # tabert_text_embedding = torch.stack(tabert_text_embeddings, dim=0)
        # tabert_schema_embedding = torch.stack(tabert_schema_embeddings, dim=0)

        # tabert_sentence_embedding: (batch, seq_len, tabert_dim)
        tabert_sentence_embedding = torch.stack(tabert_sentence_embeddings, dim=0)

        return tabert_sentence_embedding


