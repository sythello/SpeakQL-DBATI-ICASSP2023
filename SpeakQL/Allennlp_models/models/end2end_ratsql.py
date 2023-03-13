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
from modules.sql_decoder import SQLDecoder, RatsqlSQLDecoder
from modules.tabert_embedder import TaBERTEmbedder
from modules import SpeakQLAudioEncoder, Wav2vecAudioEncoder

from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem, load_tables
from ratsql.utils import registry
from ratsql.models.spider.spider_enc import SpiderEncoderState, SpiderEncoderV2Preproc, preprocess_schema_uncached
from ratsql.models.nl2code.decoder import NL2CodeDecoderPreprocItem, NL2CodeDecoderPreproc, NL2CodeDecoder
from ratsql.models.spider.spider_beam_search import beam_search_with_heuristics_for_speakql

@Model.register('ratsql_end2end_model')
class RatsqlEnd2endModel(Model):
    def __init__(self,
                 encoder: SpeakQLEncoder,   # SpiderEncoderV2 from ratsql
                 sql_decoder: SQLDecoder,   # NL2CodeDecoder from ratsql
                 vocab: Vocabulary = None) -> None:
        super().__init__(vocab)

        self.encoder = encoder
        self.sql_decoder = sql_decoder

        # Metrics
        self.mle_loss_metrics = Average()
        self.align_loss_metrics = Average()

        self.save_intermediate = False
        self.intermediates = dict()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        # self.encoder.set_save_intermediate(save_intermediate)
        # self.rewrite_decoder.set_save_intermediate(save_intermediate)
        self.save_intermediate = save_intermediate
    
    @overrides
    def forward(self,
                **kwargs) -> Dict[str, torch.Tensor]:
        '''
        Expected kwargs:
            sentence: Dict[str, torch.Tensor],
            source_token_ids: torch.Tensor,
            source_to_target: torch.Tensor,
            text_mask: torch.Tensor,
            schema_mask: torch.Tensor,
            schema_column_ids: torch.Tensor, # For each schema token, which column it belongs to (starting at 1), 0 if not column
            audio_feats: torch.Tensor,
            audio_mask: torch.Tensor,
            metadata: List[Dict],
            align_tags: torch.Tensor = None

        In this model (end2end_ratsql), only need metadata["ratsql_items"]
        '''
        
        # sentence = kwargs.get("sentence", None)
        # source_token_ids = kwargs.get("source_token_ids", None)
        # source_to_target = kwargs.get("source_to_target", None)
        # text_mask = kwargs.get("text_mask", None)
        # schema_mask = kwargs.get("schema_mask", None)
        # schema_column_ids = kwargs.get("schema_column_ids", None)
        # audio_feats = kwargs.get("audio_feats", None)
        # audio_mask = kwargs.get("audio_mask", None)
        metadata = kwargs.get("metadata", None)
        # align_tags = kwargs.get("align_tags", None)

        # ratsql_spider_item, ratsql_enc_item, ratsql_dec_item = metadata[i]["ratsql_items"]
        
        
        # Not sure if this is the right way
        # cuda_device_id = get_device_of(audio_feats)
        # cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        enc_inputs = [_meta['ratsql_items'][1] for _meta in metadata]
        enc_states_nl2code = self.encoder(enc_inputs)

        output = {
            'enc_states_nl2code': enc_states_nl2code
        }
        
        losses = []
        for enc_state, _meta in zip(enc_states_nl2code, metadata):
            spider_item, enc_preproc_item, dec_preproc_item = _meta['ratsql_items']

            _loss_ok = True
            try:
                loss = self.sql_decoder.compute_loss(enc_preproc_item, dec_preproc_item, enc_state, debug=False)
            except KeyError as e:
                _err_rule = e.args[0]
                print("\nIgnored sample with unseen rule:", _err_rule)
                _loss_ok = False
            
            if _loss_ok:
                losses.append(loss)

        total_loss = torch.mean(torch.stack(losses, dim=0), dim=0)
        output['loss'] = total_loss
        
        return output
        

    # @overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     metrics_dict = dict()

    #     metrics_dict["L_mle"] = self.mle_loss_metrics.get_metric(reset)
    #     metrics_dict["L_align"] = self.align_loss_metrics.get_metric(reset)

    #     return metrics_dict


    def begin_inference(self, speakql_input):
        metadata = speakql_input["metadata"]
        assert len(metadata) == 1
        orig_item, enc_preproc_item, dec_preproc_item = metadata[0]['ratsql_items']
        
        forward_output = self.forward(**speakql_input)
        enc_state = forward_output['enc_states_nl2code'][0]
        
        return self.sql_decoder.begin_inference(enc_state, orig_item)


