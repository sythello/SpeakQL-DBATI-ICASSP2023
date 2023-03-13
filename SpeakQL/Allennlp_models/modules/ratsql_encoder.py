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

from .encoder import SpeakQLEncoder

from ratsql.models.spider.spider_enc import SpiderEncoderV2, SpiderEncoderState, SpiderEncoderV2Preproc, preprocess_schema_uncached


@SpeakQLEncoder.register('ratsql_encoder')
class RatsqlEncoder(SpeakQLEncoder, SpiderEncoderV2):
    def __init__(self,
            device_name,
            preproc_config,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.,
            question_encoder=('emb', 'bilstm'),
            column_encoder=('emb', 'bilstm'),
            table_encoder=('emb', 'bilstm'),
            update_config={},
            include_in_memory=('question', 'column', 'table'),
            batch_encs_update=True,
            top_k_learnable=0):
        SpeakQLEncoder.__init__(self)

        device = torch.device(device_name)
        preproc = SpiderEncoderV2Preproc(**preproc_config)
        preproc.load()

        SpiderEncoderV2.__init__(self,
            device=device,
            preproc=preproc,
            word_emb_size=word_emb_size,
            recurrent_size=recurrent_size,
            dropout=dropout,
            question_encoder=question_encoder,
            column_encoder=column_encoder,
            table_encoder=table_encoder,
            update_config=update_config,
            include_in_memory=include_in_memory,
            batch_encs_update=batch_encs_update,
            top_k_learnable=top_k_learnable)














