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
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
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

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from utils.spider import process_sql

# Import custom modules
from modules.encoder import SpeakQLEncoder, SpeakQLEncoderV1
from modules.tabert_embedder import TaBERTEmbedder

# torch.manual_seed(1)

@Model.register('spider_ASR_reranker_siamese')
class SpiderASRReranker_Siamese(Model):
    '''
    Siamese:
    Wrapper of a regression model.
    Take a pair of input, feed each to underlying model, output margin loss.
    '''

    def __init__(self,
                 regression_model: Model,
                 margin: float) -> None:
        super().__init__(regression_model.vocab)
        self.regression_model = regression_model
        self.margin = margin
    
    def forward(self, **kwargs):
        # Assume that fields for sample_1 ends with '_1', and fields for sample_2 ends with '_2'
        # (A "sample" can be a batch)
        # Assume that sample_1 is the correct one 
        
        # [:-2] to remove '_1' and '_2'
        kwargs_1 = {k[:-2] : kwargs[k] for k in kwargs if isinstance(k, str) and k.endswith('_1')}
        kwargs_2 = {k[:-2] : kwargs[k] for k in kwargs if isinstance(k, str) and k.endswith('_2')}
        if len(kwargs_1) * 2 == len(kwargs_2) * 2 == len(kwargs):
            siamese = True  # Training 
        elif len(kwargs_1) == len(kwargs) and len(kwargs_2) == 0:
            siamese = False # Test
        else:
            raise ValueError
        
        if siamese:
            # Training 
            output_1 = self.regression_model(**kwargs_1)
            output_2 = self.regression_model(**kwargs_2)
        
            output = {
                'score_preds_1': output_1['score_preds'],
                'score_preds_2': output_2['score_preds'],
            }
            
            target = torch.ones_like(output_1['score_preds'])
            # print('Input1:', output_1['score_preds'].size())
            # print('Input2:', output_2['score_preds'].size())
            # print('Target:', target.size())
            output['loss'] = F.margin_ranking_loss(
                input1=output_1['score_preds'],
                input2=output_2['score_preds'],
                target=target,
                margin=self.margin)

            for k, v in output_1.items():
                if k.startswith('aux_loss:'):
                    ## an auxiliary loss, including ref-att-loss, ph-mlabel-loss, probing losses, etc.
                    output['loss'] += v
        
            for k, v in output_2.items():
                if k.startswith('aux_loss:'):
                    ## an auxiliary loss, including ref-att-loss, ph-mlabel-loss, probing losses, etc.
                    output['loss'] += v

        else:
            # Test
            output_1 = self.regression_model(**kwargs1)
            
            output = {
                'score_preds': output_1['score_preds'],
            }
        
        return output



