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
from utils.bert import utils_bert_pa

# torch.manual_seed(1)

@Predictor.register('spider_ASR_reranker_predictor')
class SpiderASRRerankerPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs['question'] = ' '.join([str(tok) for tok in instance.fields['sentence']]).split(' [SEP] ')[0]
        outputs['original_id'] = instance.fields['metadata']['original_id']

        ## For debug / analysis purpose, add the processed input seq (utter + DB)
        try:
            outputs['input_seq'] = instance.fields['metadata']['source_tokens']
        except KeyError:
            # compatible with older versions 
            pass

        return sanitize(outputs)


@Predictor.register('spider_ASR_reranker_predictor_siamese')
class SpiderASRRerankerPredictor_Siamese(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.regression_model.forward_on_instance(instance)
        outputs['question'] = ' '.join([str(tok) for tok in instance.fields['sentence']]).split(' [SEP] ')[0]
        outputs['original_id'] = instance.fields['metadata']['original_id']

        ## For debug / analysis purpose, add the processed input seq (utter + DB)
        try:
            outputs['input_seq'] = instance.fields['metadata']['source_tokens']
        except KeyError:
            # compatible with older versions 
            pass
            
        return sanitize(outputs)





