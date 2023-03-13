from typing import Iterator, List, Dict, Union, Optional, cast
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
import random

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from fairseq.models.wav2vec import Wav2VecModel

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema

import table_bert
from table_bert import TableBertModel
from .reader_utils import extractAudioFeatures, extractAudioFeatures_NoPooling, \
    extractRawAudios, extractAudioFeatures_NoPooling_Wav2vec, \
    dbToTokens, dbToTokensWithColumnIndexes, read_DB, Get_align_tags

AUDIO_DIM = 136
AUDIO_DIM_NO_POOLING = 68
AUDIO_DIM_WAV2VEC_PROJ = 64


@DatasetReader.register('spider_ASR_rewriter_reader_tagger_tabert')
class SpiderASRRewriterReader_Tagger_TaBERT(DatasetReader):
    '''
    Sequence tagging + ILM rewrite generation, tagger part
    '''
    def __init__(self,
                 tables_json_fname: str,
                 dataset_dir: str,
                 databases_dir: str,
                 tabert_model_path: str,
                 specify_full_path: bool = False,   # For self._read(), whether file_path includes a full path (split:full_path) or split only
                 ## query_audio_dir:
                 # If str: the dir containing {id}.wav files
                 # If dict: Dict[ds, audio_dir], ds in ['train', 'dev', 'test']
                 # If None, use default ({dataset_dir}/{sub_dir}/speech_wav)
                 query_audio_dir: Union[str, dict] = None,
                 src_token_indexers: Dict[str, TokenIndexer] = None,
                 # tgt_token_indexers: Dict[str, TokenIndexer] = None,
                 audio_feats_type: str = 'mfcc',            # ['mfcc', 'wav2vec', 'raw']
                 wav2vec_model_path: str = None,
                 max_sequence_len: int = 300,
                 include_align_tags: bool = False,  # Include align tags (token match/mismatch within other cands, [SAME]/[DIFF-n])
                 include_gold_tags: bool = True,
                 prototype_dict_path: str = None,
                 tabert_column_type_source: str = 'spider', # ['spider', 'sqlite', 'sqlite_legacy']
                 samples_limit: int = None,                 # Only using this number of samples, for analysis; None = use all
                 debug: bool = False,
                 cpu: bool = False,
                 shuffle: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tables_json_fname = tables_json_fname
        self.dataset_dir = dataset_dir
        self.databases_dir = databases_dir
        self.tabert_model_path = tabert_model_path
        self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)

        self.specify_full_path = specify_full_path

        self.query_audio_dir = query_audio_dir

        self.include_align_tags = include_align_tags

        self.src_token_indexers = src_token_indexers # Can use BERT, word-level, char-level, etc.
        # self.tgt_token_indexers = tgt_token_indexers # Should only be word-level 
        self.max_sequence_len = max_sequence_len
        self.include_gold_tags = include_gold_tags
        self.debug = debug
        self.cpu = cpu
        self.samples_limit = samples_limit


        assert audio_feats_type in ['mfcc', 'wav2vec', 'raw']
        self.audio_feats_type = audio_feats_type
        if audio_feats_type == 'mfcc':
            self.audio_feats_dim = AUDIO_DIM_NO_POOLING
        elif audio_feats_type == 'wav2vec':
            self.audio_feats_dim = AUDIO_DIM_WAV2VEC_PROJ
        elif audio_feats_type == 'raw':
            self.audio_feats_dim = 1

        self.wav2vec_model_path = wav2vec_model_path
        if self.audio_feats_type == 'wav2vec':
            assert wav2vec_model_path is not None
            if cpu:
                _cp = torch.load(wav2vec_model_path, map_location=torch.device('cpu'))
            else:
                _cp = torch.load(wav2vec_model_path)
            self.w2v_model = Wav2VecModel.build_model(_cp['args'], task=None)
            self.w2v_model.load_state_dict(_cp['model'])
            self.w2v_model.eval()


        self.prototype_dict_path = prototype_dict_path
        if prototype_dict_path is not None:
            with open(prototype_dict_path, 'rb') as f:
                self.prototype_dict = pickle.load(f)
        else:
            self.prototype_dict = None

        assert tabert_column_type_source in ['spider', 'sqlite', 'sqlite_legacy']
        self.tabert_column_type_source = tabert_column_type_source

        self._tabert_tables_cache = dict()  # Dict[db_id, Dict[table_name, table_bert.Table]]
    
    def text_to_instance(self,
                         original_id: int,
                         text_tokens: List[Token],
                         schema_tokens: List[Token],
                         schema_column_ids: List[int],      # To retreive column encodings from TaBERT
                         tabert_tables: Dict[str, table_bert.Table],
                         text_audio_feats: List[np.ndarray],
                         schema_audio_feats: List[np.ndarray],
                         align_tags: List[str],
                         rewriter_tags: List[str]) -> Instance:
        
        concat_tokens = text_tokens + [Token('[SEP]')] + schema_tokens
    
        if len(concat_tokens) > self.max_sequence_len:
            excess_len = len(concat_tokens) - self.max_sequence_len
            concat_tokens = concat_tokens[:-excess_len]
            schema_tokens = schema_tokens[:-excess_len]
            schema_column_ids = schema_column_ids[:-excess_len]
            schema_audio_feats = schema_audio_feats[:-excess_len]
        else:
            excess_len = 0
        
        concat_audio_feats = text_audio_feats + [np.zeros((1, self.audio_feats_dim))] + schema_audio_feats
        assert len(concat_audio_feats) == len(concat_tokens)
        audio_lens = [_audio_feats.shape[0] for _audio_feats in concat_audio_feats]
        audio_lens_max = max(audio_lens)
        
        text_mask = np.array([1] * len(text_tokens) + [0] * (len(concat_tokens) - len(text_tokens)), dtype=np.int)
        schema_mask = np.array([0] * (len(concat_tokens) - len(schema_tokens)) + [1] * len(schema_tokens), dtype=np.int)
        audio_mask = np.array([
            [1] * a_len + [0] * (audio_lens_max - a_len) for a_len in audio_lens
        ], dtype=np.int)
        
        # For TaBERT: Need to do tokenize for each word, and keep their offsets; later use these to reconstruct word encodings
        text_tokenized = []
        offsets = []
        for _tok in text_tokens:
            _tok_pieces = self.tabert_model.tokenizer.tokenize(_tok.text)
            _st = len(text_tokenized)
            _ed = _st + len(_tok_pieces)
            offsets.append((_st, _ed))
            text_tokenized.extend(_tok_pieces)


        meta_field = MetadataField({
            "original_id": original_id,
            "text_len": len(text_tokens),
            "schema_len": len(schema_tokens),
            # "source_tokens": [t.text for t in concat_tokens],
            # TaBERT
            "tabert_tables": tabert_tables,
            "text_tokenized": text_tokenized,
            "text_offsets": offsets
        })

        concat_sentence_field = TextField(concat_tokens, self.src_token_indexers)
        text_mask_field = ArrayField(text_mask, dtype=np.int)
        schema_mask_field = ArrayField(schema_mask, dtype=np.int)
        audio_mask_field = ArrayField(audio_mask, dtype=np.int)
        
        schema_column_id_field = ArrayField(np.array(schema_column_ids), dtype=np.int)  # TaBERT

        concat_audio_field = ListField([ArrayField(_token_audio_feats) for _token_audio_feats in concat_audio_feats])
        
        fields = {"sentence": concat_sentence_field,
                  "text_mask": text_mask_field,
                  "schema_mask": schema_mask_field,
                  "schema_column_ids": schema_column_id_field,  # TaBERT
                  "audio_feats": concat_audio_field,
                  "audio_mask": audio_mask_field,
                  "metadata": meta_field}

        if align_tags is not None:
            align_tag_tokens_padded = align_tags + ['[O]' for _ in range(len(schema_tokens) + 1)]            
            align_tags_field = SequenceLabelField(labels=align_tag_tokens_padded,
                                            sequence_field=concat_sentence_field,
                                            label_namespace='align_tags')

            assert len(align_tag_tokens_padded) == len(concat_tokens)
            fields["align_tags"] = align_tags_field


        if rewriter_tags is not None:
            rewriter_tag_tokens_padded = rewriter_tags + ['O' for _ in range(len(schema_tokens) + 1)]            
            rewriter_tags_field = SequenceLabelField(labels=rewriter_tag_tokens_padded,
                                            sequence_field=concat_sentence_field,
                                            label_namespace='rewriter_tags')
            
            assert len(rewriter_tag_tokens_padded) == len(concat_tokens)
            fields["rewriter_tags"] = rewriter_tags_field
            
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        # file_path: if self.specify_full_path, this is "split:full_path";
        #   otherwise, this is only the dataset split, e.g. train, dev, test.
        
        if self.specify_full_path:
            ds, dataset_json_path = file_path.split(':')
            sub_dir = ('dev' if ds == 'test' else ds)
        else:
            ds = file_path  # dataset split
            sub_dir = ('dev' if ds == 'test' else ds)
            dataset_json_path = os.path.join(self.dataset_dir, sub_dir, '{}_rewriter.json'.format(ds))
        
        databases = read_dataset_schema(self.tables_json_fname)
        
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)

        if self.samples_limit is not None:
            assert 0 < self.samples_limit <= len(dataset_json), \
                f'samples_limit should be in (0, {len(dataset_json)}], but got {self.samples_limit}'

            _all_ids = list(range(len(dataset_json)))
            _sample_ids = random.sample(_all_ids, k=self.samples_limit)
            dataset_json = [dataset_json[i] for i in _sample_ids]
        if self.debug:
            print('warning: argument "debug" is deprecated in speakql dataset_readers')
            dataset_json = dataset_json[:5] + dataset_json[5::400]
        
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for i, cand_list in enumerate(dataset_json):
            
            if len(cand_list) == 0:
                continue
            
            o_id = cand_list[0]['original_id']
            
            db_id = cand_list[0]['db_id'] # Maybe should combine the common fields for all candidates, like original_id, db_id, etc. 
            db_schema = databases[db_id]
            
            # Schema processing
            schema_sentence, schema_column_ids = dbToTokensWithColumnIndexes(db_schema)
            schema_tokens = [Token(tok) for tok in schema_sentence]
            schema_audio_feats_list = []
            for token in schema_sentence:
                if token in ',.:':
                    schema_audio_feats_list.append(np.zeros((1, self.audio_feats_dim)))
                else:
                    # Actual word
                    if self.audio_feats_type == 'mfcc':
                        # Use pre-saved feats
                        feats_fname = os.path.join(self.dataset_dir, 'db', 'speech_feats_array', '{}.pkl'.format(token))
                        with open(feats_fname, 'rb') as f:
                            feats_array = pickle.load(f)
                        schema_audio_feats_list.append(feats_array.T)  # In pkl files, shapes are (68, audio_len)
                    elif self.audio_feats_type == 'wav2vec':
                        # Use pre-saved feats
                        feats_fname = os.path.join(self.dataset_dir, 'db', 'speech_feats_array_wav2vec', '{}.pkl'.format(token))
                        with open(feats_fname, 'rb') as f:
                            feats_array = pickle.load(f)
                        schema_audio_feats_list.append(feats_array)    # In pkl files, shapes are (audio_len, 64)
                    elif self.audio_feats_type == 'raw':
                        audio_fname = os.path.join(self.dataset_dir, 'db', 'speech_wav', '{}.wav'.format(token))
                        _, feats_array = audioBasicIO.read_audio_file(audio_fname)
                        schema_audio_feats_list.append(feats_array.reshape(-1, 1))

            assert len(schema_audio_feats_list) == len(schema_sentence)


            # Collect TaBERT tables
            if db_id in self._tabert_tables_cache:
                tabert_tables = self._tabert_tables_cache[db_id]
            else:
                tabert_tables = read_DB(db_id=db_id,
                                        db_dir=self.databases_dir,
                                        db_schema=db_schema,
                                        prototype_dict=self.prototype_dict,
                                        column_type_source=self.tabert_column_type_source)

                for _tbl in tabert_tables.values():
                    _tbl.tokenize(self.tabert_model.tokenizer)

                    # print(f'Table: {_tbl.id}')
                    # print(f'Header ({len(_tbl.header)}):\n')
                    # for _col in _tbl.header:
                    #     print(f'\tColumn {_col.name} | {_col.type} | {_col.sample_value}')
                    #     assert _col.sample_value is not None

                self._tabert_tables_cache[db_id] = tabert_tables


            # Get align tags if needed
            if self.include_align_tags:
                Get_align_tags(cand_list)


            # Decide _audio_dir based on self.query_audio_dir
            if self.query_audio_dir is None:
                _audio_dir = os.path.join(self.dataset_dir, sub_dir, 'speech_wav')
            elif isinstance(self.query_audio_dir, str):
                _audio_dir = self.query_audio_dir
            elif isinstance(self.query_audio_dir, dict):
                _audio_dir = self.query_audio_dir[ds]

            # Get instances from text_to_instance()
            for cand in cand_list:
                text_tokens = [Token(word) for word in cand['question_toks']]
                # audio_fname = os.path.join(self.dataset_dir, sub_dir, 'speech_wav', '{}.wav'.format(o_id))
                audio_fname = os.path.join(_audio_dir, '{}.wav'.format(o_id))

                # text_audio_feats_list = extractAudioFeatures_NoPooling(audio_fname, cand['span_ranges'])
                if self.audio_feats_type == 'mfcc':
                    text_audio_feats_list = extractAudioFeatures_NoPooling(audio_fname, cand['span_ranges'])
                elif self.audio_feats_type == 'wav2vec':
                    text_audio_feats_list = extractAudioFeatures_NoPooling_Wav2vec(audio_fname, cand['span_ranges'], self.w2v_model)
                elif self.audio_feats_type == 'raw':
                    text_audio_feats_list = extractRawAudios(audio_fname, cand['span_ranges'])
                
                
                if self.include_align_tags:
                    align_tags = cand['align_tags']
                else:
                    align_tags = None

                if self.include_gold_tags:
                    rewriter_tags = cand['rewriter_tags']
                else:
                    rewriter_tags = None
                
                yield self.text_to_instance(original_id=o_id,
                                            text_tokens=text_tokens,
                                            schema_tokens=schema_tokens,
                                            schema_column_ids=schema_column_ids, # TaBERT
                                            tabert_tables=tabert_tables,
                                            text_audio_feats=text_audio_feats_list,
                                            schema_audio_feats=schema_audio_feats_list,
                                            align_tags=align_tags,
                                            rewriter_tags=rewriter_tags)




