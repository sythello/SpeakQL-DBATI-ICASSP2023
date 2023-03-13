from typing import Iterator, List, Dict, Optional, cast
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
    get_device_of, masked_softmax, weighted_sum, \
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal, \
    batched_span_select

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
from modules.copynet_decoder import CopyNetSeqDecoder
from modules.tabert_embedder import TaBERTEmbedder
from modules import SpeakQLAudioEncoder, Wav2vecAudioEncoder

from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem, load_tables
from ratsql.utils import registry
from ratsql.models.spider.spider_enc import SpiderEncoderState, SpiderEncoderV2Preproc, preprocess_schema_uncached
from ratsql.models.nl2code.decoder import NL2CodeDecoderPreprocItem, NL2CodeDecoderPreproc, NL2CodeDecoder
from ratsql.models.spider.spider_beam_search import beam_search_with_heuristics_for_speakql

class SQLDecoder(Registrable):
    pass

@SQLDecoder.register("ratsql_decoder")
class RatsqlSQLDecoder(SQLDecoder, NL2CodeDecoder):
    def __init__(self,
            device_name,
            preproc_config,
            rule_emb_size=128,
            node_embed_size=64,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=256,
            recurrent_size=256,
            dropout=0.,
            desc_attn='bahdanau',
            copy_pointer=None,
            multi_loss_type='logsumexp',
            sup_att=None,
            use_align_mat=False,
            use_align_loss=False,
            enumerate_order=False,
            loss_type="softmax"):
        SQLDecoder.__init__(self)

        device = torch.device(device_name)
        preproc = NL2CodeDecoderPreproc(**preproc_config)
        preproc.load()
        
        NL2CodeDecoder.__init__(self,
            device=device,
            preproc=preproc,
            rule_emb_size=rule_emb_size,
            node_embed_size=node_embed_size,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=enc_recurrent_size,
            recurrent_size=recurrent_size,
            dropout=dropout,
            desc_attn=desc_attn,
            copy_pointer=copy_pointer,
            multi_loss_type=multi_loss_type,
            sup_att=sup_att,
            use_align_mat=use_align_mat,
            use_align_loss=use_align_loss,
            enumerate_order=enumerate_order,
            loss_type=loss_type)


















