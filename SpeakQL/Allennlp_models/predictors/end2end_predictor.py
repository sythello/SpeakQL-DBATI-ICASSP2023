from typing import Iterator, List, Dict, Optional, cast
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn import ModuleList

import numpy as np
from allennlp.data import Instance
from allennlp.data.batch import Batch
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
    batched_span_select, move_to_device

from allennlp.training.metrics import CategoricalAccuracy, MeanAbsoluteError
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
from utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema

from modules.encoder import SpeakQLEncoderV1

from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem, load_tables
from ratsql.utils import registry
from ratsql.models.spider.spider_enc import SpiderEncoderState, SpiderEncoderV2Preproc, preprocess_schema_uncached
from ratsql.models.nl2code.decoder import NL2CodeDecoderPreprocItem, NL2CodeDecoderPreproc, NL2CodeDecoder
from ratsql.models.spider.spider_beam_search import beam_search_with_heuristics_for_speakql


@Predictor.register('speakql_end2end_predictor')
class SpeakQLEnd2EndPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 beam_size=4,
                 max_steps=1000):
        super().__init__(model, dataset_reader)
        
        self.beam_size = beam_size
        self.max_steps = max_steps
        
        self.save_intermediate = False

    def set_save_intermediate(self, save_intermediate: bool):
        self._model.set_save_intermediate(save_intermediate)
        self.save_intermediate = save_intermediate
    
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        # Input instance has gold rewrite_seq_s2s  
        
        outputs = dict()
        metadata = instance.fields['metadata']
        outputs['question'] = ' '.join(metadata['source_tokens'][:metadata['text_len']])
        outputs['original_id'] = metadata['original_id']
        outputs['gold_sql'] = metadata['target_written_sql']
        
        ## In eval mode, model will forward with beam_search even if gold is provided 
        # _instance = Instance(instance.fields.copy())
        # del _instance.fields['rewrite_seq_s2s']
        
        with torch.no_grad():
            # cuda_device = self._model._get_prediction_device()
            _batch = Batch([instance])
            _batch.index_instances(self._model.vocab)
            # model_input = move_to_device(_batch.as_tensor_dict(), self.cuda_device)
            model_input = _batch.as_tensor_dict()
            
            sql_beam_search_outputs = beam_search_with_heuristics_for_speakql(
                model=self._model,
                speakql_input=model_input,
                orig_item=metadata['ratsql_items'][0],
                preproc_item=metadata['ratsql_items'][1],
                beam_size=self.beam_size,
                max_steps=self.max_steps,
            )
            # outputs = self.make_output_human_readable(self(**model_input))
        
        pred_sql = ''
        if len(sql_beam_search_outputs) > 0:
            # pred_sql = sql_beam_search_outputs[0]['inferred_code']
            beam = sql_beam_search_outputs[0]
            model_output, inferred_code = beam.inference_state.finalize()
            pred_sql = inferred_code
        
        outputs['pred_sql'] = pred_sql

        if self.save_intermediate:
            outputs['intermediates'] = self._model.get_intermediates()
        
        return sanitize(outputs)









