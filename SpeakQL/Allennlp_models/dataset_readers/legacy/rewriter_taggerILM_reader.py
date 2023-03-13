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

from allennlp.common.util import START_SYMBOL, END_SYMBOL

# from spacy.tokenizer import Tokenizer as SpacyTokenizer
# from spacy.lang.en import English
# nlp = English()
# Create a blank Tokenizer with just the English vocab
# tokenizer = Tokenizer(nlp.vocab)

# from tqdm import tqdm

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

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema
from .reader_utils import extractAudioFeatures, extractAudioFeatures_NoPooling, dbToTokens

AUDIO_DIM = 136
AUDIO_DIM_NO_POOLING = 68


@DatasetReader.register('spider_ASR_rewriter_reader_tagger_ILM')
class SpiderASRRewriterReader_Tagger_ILM(DatasetReader):
    '''
    Sequence tagging + ILM rewrite generation
    '''
    def __init__(self,
                 tables_json_fname: str,
                 dataset_dir: str,
                 src_token_indexers: Dict[str, TokenIndexer] = None,
                 tgt_token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_len: int = 300,
                 include_gold_tags: bool = True,
                 include_gold_rewrite_seq: bool = True,
                 debug: bool = False) -> None:
        super().__init__(lazy=False)
        self.tables_json_fname = tables_json_fname
        self.dataset_dir = dataset_dir
        self.src_token_indexers = src_token_indexers # Can use BERT, word-level, char-level, etc.
        self.tgt_token_indexers = tgt_token_indexers # Should only be word-level 
        self.debug = debug
        self.max_sequence_len = max_sequence_len
        
        self.include_gold_tags = include_gold_tags
        self.include_gold_rewrite_seq = include_gold_rewrite_seq
    
    def text_to_instance(self,
                         original_id: int,
                         text_tokens: List[Token],
                         schema_tokens: List[Token],
                         text_audio_feats: List[np.ndarray],
                         schema_audio_feats: List[np.ndarray],
                         tags: List[str],
                         rewrite_tokens: List[Token]) -> Instance:
        
        concat_tokens = text_tokens + [Token('[SEP]')] + schema_tokens
    
        if len(concat_tokens) > self.max_sequence_len:
            excess_len = len(concat_tokens) - self.max_sequence_len
            concat_tokens = concat_tokens[:-excess_len]
            schema_tokens = schema_tokens[:-excess_len]
            schema_audio_feats = schema_audio_feats[:-excess_len]
        else:
            excess_len = 0
        
        concat_audio_feats = text_audio_feats + [np.zeros((1, AUDIO_DIM_NO_POOLING))] + schema_audio_feats
        assert len(concat_audio_feats) == len(concat_tokens)
        audio_lens = [_audio_feats.shape[0] for _audio_feats in concat_audio_feats]
        audio_lens_max = max(audio_lens)
        
        text_mask = np.array([1] * len(text_tokens) + [0] * (len(concat_tokens) - len(text_tokens)), dtype=np.int)
        schema_mask = np.array([0] * (len(concat_tokens) - len(schema_tokens)) + [1] * len(schema_tokens), dtype=np.int)
        audio_mask = np.array([
            [1] * a_len + [0] * (audio_lens_max - a_len) for a_len in audio_lens
        ], dtype=np.int)
        
        meta_field = MetadataField({'original_id': original_id,
                                    'text_len': len(text_tokens),
                                    'schema_len': len(schema_tokens),
                                    'rewrite_seq_len': len(rewrite_tokens)
                                   })
        concat_sentence_field = TextField(concat_tokens, self.src_token_indexers)
        text_mask_field = ArrayField(text_mask, dtype=np.int)
        schema_mask_field = ArrayField(schema_mask, dtype=np.int)
        audio_mask_field = ArrayField(audio_mask, dtype=np.int)
        
        # text_audio_field = MetadataField(text_audio_feats) # List[np.array(steps, audio_dim)]
        # schema_audio_field = MetadataField(schema_audio_feats)
        concat_audio_field = ListField([ArrayField(_token_audio_feats) for _token_audio_feats in concat_audio_feats])
        
        fields = {"sentence": concat_sentence_field,
                  "text_mask": text_mask_field,
                  "schema_mask": schema_mask_field,
                  # "text_audio_feats": text_audio_field,
                  # "schema_audio_feats": schema_audio_field,
                  "audio_feats": concat_audio_field,
                  "audio_mask": audio_mask_field,
                  "metadata": meta_field}

        if tags is not None:
            tag_tokens_padded = tags + ['O' for _ in range(len(schema_tokens) + 1)]            
            tags_field = SequenceLabelField(labels=tag_tokens_padded,
                                            sequence_field=concat_sentence_field,
                                            label_namespace='rewriter_tags')
            
            # tag_tokens_padded = [Token(t) for t in tags] + [Token('O') for _ in range(len(schema_tokens) + 1)]
            # tags_field = TextField(tag_tokens_padded, {'tokens' : SingleIdTokenIndexer()})

            assert len(tag_tokens_padded) == len(concat_tokens)
            fields["tags"] = tags_field
        if rewrite_tokens is not None:
            rewrite_tokens.insert(0, Token(START_SYMBOL))
            rewrite_tokens.append(Token(END_SYMBOL))

            rewrite_seq_field = TextField(rewrite_tokens, self.tgt_token_indexers)
            fields["rewrite_seq"] = rewrite_seq_field
            
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        # file_path: dataset split, e.g. train, dev, test.
        
        ds = file_path  # dataset split
        sub_dir = ('dev' if ds == 'test' else ds)
        
        dataset_json_path = os.path.join(self.dataset_dir, sub_dir, '{}_rewriter.json'.format(ds))
        
        databases = read_dataset_schema(self.tables_json_fname)
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        if self.debug:
            dataset_json = dataset_json[::200]
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for i, cand_list in enumerate(dataset_json):
            
            if len(cand_list) == 0:
                continue
            
            o_id = cand_list[0]['original_id']
            
            db_id = cand_list[0]['db_id'] # Maybe should combine the common fields for all candidates, like original_id, db_id, etc. 
            db_schema = databases[db_id]
            
            schema_sentence = dbToTokens(db_schema)
            schema_tokens = [Token(tok) for tok in schema_sentence]
            schema_audio_feats_list = []
            for token in schema_sentence:
                if token in ',.:':
                    schema_audio_feats_list.append(np.zeros((1, AUDIO_DIM_NO_POOLING)))
                else:
                    feats_fname = os.path.join(self.dataset_dir, 'db', 'speech_feats_array', '{}.pkl'.format(token))
                    with open(feats_fname, 'rb') as f:
                        feats_array = pickle.load(f)
                    schema_audio_feats_list.append(feats_array.T)  # In pkl files, shapes are (68, audio_len)

            assert len(schema_audio_feats_list) == len(schema_sentence)

            for cand in cand_list:
                text_tokens = [Token(word) for word in cand['question_toks']]
                audio_fname = os.path.join(self.dataset_dir, sub_dir, 'speech_wav', '{}.wav'.format(o_id))
                text_audio_feats_list = extractAudioFeatures_NoPooling(audio_fname, cand['span_ranges'])
                
                if self.include_gold_tags:
                    tags = cand['rewriter_tags']
                else:
                    tags = None

                if self.include_gold_rewrite_seq:
                    # rewrite_list = ['[START_RWT]']
                    rewrite_list = []
                    for _edit in cand['rewriter_edits']:
                        rewrite_list.extend(_edit['tgt_text'].split(' '))
                        rewrite_list.append('[ANS]')
                    # rewrite_list.append('[END_RWT]')
                    rewrite_tokens = [Token(w) for w in rewrite_list]
                else:
                    rewrite_tokens = None
                
                yield self.text_to_instance(original_id=o_id,
                                            text_tokens=text_tokens,
                                            schema_tokens=schema_tokens,
                                            text_audio_feats=text_audio_feats_list,
                                            schema_audio_feats=schema_audio_feats_list,
                                            tags=tags,
                                            rewrite_tokens=rewrite_tokens)




