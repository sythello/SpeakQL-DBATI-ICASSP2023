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
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal

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



@Predictor.register('spider_ASR_rewriter_predictor_tagger_ILM')
class SpiderASRRewriterPredictor_Tagger_ILM(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)
        self.save_intermediate = False

    def set_save_intermediate(self, save_intermediate: bool):
        self._model.set_save_intermediate(save_intermediate)
        self.save_intermediate = save_intermediate

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        # Input instance has gold tags and rewrite_seq 
        
        outputs = dict()
        outputs['question'] = ' '.join([str(tok) for tok in instance.fields['sentence']]).split(' [SEP] ')[0]
        outputs['original_id'] = instance.fields['metadata']['original_id']
        outputs['gold_tags'] = list(instance.fields['tags'])
        outputs['gold_rewrite_seq'] = list(instance.fields['rewrite_seq'])
        
        ## (1) Full-fledged prediction: do not use gold tags and rewrite_seq.
        ## First predict tags, then predict the rewrite_seq 
        _instance = Instance(instance.fields.copy())
        _output = self._model.forward_on_instance(_instance)
        
        # Get tags prediction 
        tags_prediction_readable = _output['tag_pred_readable']
        outputs['tags_prediction'] = tags_prediction_readable

        _metrics = self._model.get_metrics(reset=True)
        outputs['tags_NLL'] = _metrics['tag_NLL']
        outputs['tags_accuracy'] = _metrics['tag_accuracy']
        
        if self.save_intermediate:
            outputs['tag_prediction_intermediates'] = self._model.get_intermediates()

        ## Reconstruct the tags field; predict the rewrite_seq based on predicted tags
        schema_len = instance.fields['metadata']['schema_len']
        tag_tokens_padded = tags_prediction_readable + ['O' for _ in range(schema_len + 1)]

        tags_field = SequenceLabelField(labels=tag_tokens_padded,
                                        sequence_field=instance.fields['sentence'],
                                        label_namespace='rewriter_tags')
        _instance.add_field('tags', tags_field, vocab=self._model.vocab)
        _output = self._model.forward_on_instance(_instance)
        
        # Get rewrite_seq prediction 
        rewrite_seq_predictions = _output['rewrite_seq_preds_readable']
        outputs['rewrite_seq_prediction'] = rewrite_seq_predictions[0]
        outputs['rewrite_seq_prediction_cands'] = rewrite_seq_predictions
        
        _metrics = self._model.get_metrics(reset=True)
        outputs['rewrite_seq_NLL'] = _metrics['rewrite_seq_NLL']

        if self.save_intermediate:
            outputs['rewrite_seq_prediction_intermediates'] = self._model.get_intermediates()
        
        ## (2) Only rewrite_seq prediction: use gold tags to predict rewrite_seq 
        _instance = Instance(instance.fields.copy())
        # del _instance.fields['rewrite_seq']
        _output = self._model.forward_on_instance(_instance)
        
        oracle_tags_rewrite_seq_predictions = _output['rewrite_seq_preds_readable']
        outputs['oracle_tags_rewrite_seq_prediction'] = oracle_tags_rewrite_seq_predictions[0]
        outputs['oracle_tags_rewrite_seq_prediction_cands'] = oracle_tags_rewrite_seq_predictions
        # outputs['oracle_tags_rewrite_seq_prediction_LLs'] = _output['rewrite_seq_LLs']

        _metrics = self._model.get_metrics(reset=True)
        outputs['oracle_tags_rewrite_seq_NLL'] = _metrics['rewrite_seq_NLL']
        
        if self.save_intermediate:
            outputs['oracle_tags_rewrite_seq_prediction_intermediates'] = self._model.get_intermediates()
        
        return sanitize(outputs)




