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
import pickle
import random

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from SpeakQL.Allennlp_models.utils.spider import process_sql
from SpeakQL.Allennlp_models.utils.schema_gnn import spider_utils
from SpeakQL.Allennlp_models.utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema

import table_bert
from table_bert import TableBertModel

from .reader_utils import extractAudioFeatures, extractAudioFeatures_NoPooling, dbToTokens, dbToTokensWithColumnIndexes, \
    read_DB, Get_align_tags
    
# torch.manual_seed(1)

AUDIO_DIM_POOLING = 136
AUDIO_DIM_NO_POOLING = 68

@DatasetReader.register('spider_ASR_reranker_reader_v2_siamese_tabert')
class SpiderASRRerankerReaderV2_Siamese_TaBERT(DatasetReader):
    '''
    V2: Audio features are passed as sequences, not avg/max pooled vectors.
    Siamese: Each instance include fields for a pair of inputs, fields_1 for sample 1 and fields_2 for sample 2
    '''
    
    def __init__(self,
                 tables_json_fname: str,
                 dataset_dir: str,
                 databases_dir: str,
                 tabert_model_path: str,
                 specify_full_path: bool = False,   # For self._read(), whether file_path includes a full path (split:full_path) or split only
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_len: int = 300,
                 include_align_tags: bool = False,  # Include align tags (token match/mismatch within other cands, [SAME]/[DIFF-n])
                 prototype_dict_path: str = None,
                 tabert_column_type_source: str = 'spider',  # ['spider', 'sqlite', 'sqlite_legacy']
                 for_training: bool = True,         # True for training, False for inference
                 samples_limit: int = None,         # Only using this number of samples, for analysis; None = use all
                 debug: bool = False) -> None:
        super().__init__(lazy=False)
        self.tables_json_fname = tables_json_fname
        self.dataset_dir = dataset_dir
        self.databases_dir = databases_dir
        self.tabert_model_path = tabert_model_path
        self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)

        self.specify_full_path = specify_full_path

        self.include_align_tags = include_align_tags
        
        self.token_indexers = token_indexers
        self.max_sequence_len = max_sequence_len
        self.for_training = for_training
        self.debug = debug
        self.samples_limit = samples_limit

        self.prototype_dict_path = prototype_dict_path
        if prototype_dict_path is not None:
            with open(prototype_dict_path, 'rb') as f:
                self.prototype_dict = pickle.load(f)
        else:
            self.prototype_dict = None

        assert tabert_column_type_source in ['spider', 'sqlite', 'sqlite_legacy']
        self.tabert_column_type_source = tabert_column_type_source

        self._tabert_tables_cache = dict()  # Dict[db_id, Dict[table_name, table_bert.Table]]
    
    
    def _single_sample_to_dict(self,
                         original_id: int,
                         text_tokens: List[Token],
                         schema_tokens: List[Token],
                         schema_column_ids: List[int],      # To retreive column encodings from TaBERT
                         tabert_tables: Dict[str, table_bert.Table],
                         text_audio_feats: List[np.ndarray],
                         schema_audio_feats: List[np.ndarray],
                         align_tags: List[str],
                         score: float = None) -> Dict:
        
        concat_tokens = text_tokens + [Token('[SEP]')] + schema_tokens
    
        if len(concat_tokens) > self.max_sequence_len:
            excess_len = len(concat_tokens) - self.max_sequence_len
            concat_tokens = concat_tokens[:-excess_len]
            schema_tokens = schema_tokens[:-excess_len]
            schema_column_ids = schema_column_ids[:-excess_len]
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

        # Need to do tokenize for each word, and keep their offsets; later use these to reconstruct word encodings
        text_tokenized = []
        offsets = []
        for _tok in text_tokens:
            _tok_pieces = self.tabert_model.tokenizer.tokenize(_tok.text)
            _st = len(text_tokenized)
            _ed = _st + len(_tok_pieces)
            offsets.append((_st, _ed))
            text_tokenized.extend(_tok_pieces)
        
        metadata = {"original_id": original_id,
                    "text_len": len(text_tokens),
                    "schema_len": len(schema_tokens),
                    "tabert_tables": tabert_tables,
                    "text_tokenized": text_tokenized,
                    "text_offsets": offsets}
        meta_field = MetadataField(metadata)

        # meta_field = MetadataField({'original_id': original_id,
        #                             'text_len': len(text_tokens),
        #                             'schema_len': len(schema_tokens),
        #                            })
        concat_sentence_field = TextField(concat_tokens, self.token_indexers)
        text_mask_field = ArrayField(text_mask, dtype=np.int)
        schema_mask_field = ArrayField(schema_mask, dtype=np.int)
        audio_mask_field = ArrayField(audio_mask, dtype=np.int)
        
        schema_column_id_field = ArrayField(np.array(schema_column_ids), dtype=np.int)

        concat_audio_field = ListField([ArrayField(_token_audio_feats) for _token_audio_feats in concat_audio_feats])
        
        fields = {"sentence": concat_sentence_field,
                  "text_mask": text_mask_field,
                  "schema_mask": schema_mask_field,
                  "schema_column_ids": schema_column_id_field,
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

        return fields
    
    def pair_to_instance(self,
                         param_dict_1: Dict,
                         param_dict_2: Dict) -> Instance:
        assert param_dict_1.keys() == param_dict_2.keys()
        
        _fields_1 = self._single_sample_to_dict(**param_dict_1)
        _fields_2 = self._single_sample_to_dict(**param_dict_2)
        
        fields = dict([(k + '_1', v) for k, v in _fields_1.items()] + [(k + '_2', v) for k, v in _fields_2.items()])
        
        return Instance(fields)

    def single_to_instance(self, param_dict: Dict) -> Instance:
        fields = self._single_sample_to_dict(**param_dict)

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
            dataset_json_path = os.path.join(self.dataset_dir, sub_dir, '{}_reranker.json'.format(ds))
        
        # dataset_json_path = os.path.join(self.dataset_dir, sub_dir, '{}_reranker.json'.format(ds))
        
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
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for i, cand_list in enumerate(dataset_json):
            
            if len(cand_list) == 0:
                # no cands
                continue

            if len(cand_list) == 1 and self.for_training:
                # 1 cand, not rankable, but should keep during inference
                continue

            score_list = [cand['ratsql_pred_score'] for cand in cand_list]
            max_score = max(score_list)
            min_score = min(score_list)
            if np.isclose(max_score, min_score) and self.for_training:
                # all cands tie, not rankable, but should keep during inference
                continue
            

            o_id = cand_list[0]['original_id']
            
            db_id = cand_list[0]['db_id'] # Maybe should combine the common fields for all candidates, like original_id, db_id, etc. 
            db_schema = databases[db_id]
            
            ## Read schema, get schema tokens & audio feats
            schema_sentence, schema_column_ids = dbToTokensWithColumnIndexes(db_schema)
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

            ## Get tabert tables
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

                self._tabert_tables_cache[db_id] = tabert_tables

            # Get align tags if needed
            if self.include_align_tags:
                Get_align_tags(cand_list)

            if self.for_training:
                ## Siamese, yield paired instances 

                ## Find the best & non-best cands 
                best_cands = []
                non_best_cands = []
                for i in range(len(score_list)):
                    if np.isclose(score_list[i], max_score):
                        best_cands.append(cand_list[i])
                    else:
                        non_best_cands.append(cand_list[i])
                assert len(best_cands) + len(non_best_cands) == len(cand_list)
                assert len(best_cands) > 0 and len(non_best_cands) > 0
                
                n_best = len(best_cands)
                n_non_best = len(non_best_cands)
                eff_n = max(n_best, n_non_best)
                
                cand_pairs = [(best_cands[i % n_best], non_best_cands[i % n_non_best]) for i in range(eff_n)]

                ## Yield instances
                for cand_pair in cand_pairs:
                    cand_param_dicts = []
                    for cand in cand_pair:
                        text_tokens = [Token(word) for word in cand['question_toks']]
                        audio_fname = os.path.join(self.dataset_dir, sub_dir, 'speech_wav', '{}.wav'.format(o_id))
                        text_audio_feats_list = extractAudioFeatures_NoPooling(audio_fname, cand['span_ranges'])

                        if self.include_align_tags:
                            align_tags = cand['align_tags']
                        else:
                            align_tags = None

                        cand_param_dict = {
                            'original_id': o_id,
                            'text_tokens': text_tokens,
                            'schema_tokens': schema_tokens,
                            'schema_column_ids': schema_column_ids,
                            'tabert_tables': tabert_tables,
                            'text_audio_feats': text_audio_feats_list,
                            'schema_audio_feats': schema_audio_feats_list,
                            'align_tags': align_tags,
                        }
                        cand_param_dicts.append(cand_param_dict)
                    
                    assert len(cand_param_dicts) == 2
                    
                    yield self.pair_to_instance(cand_param_dicts[0], cand_param_dicts[1])
            else:
                ## Not Siamese, yield single instances

                for cand in cand_list:
                    text_tokens = [Token(word) for word in cand['question_toks']]
                    audio_fname = os.path.join(self.dataset_dir, sub_dir, 'speech_wav', '{}.wav'.format(o_id))
                    text_audio_feats_list = extractAudioFeatures_NoPooling(audio_fname, cand['span_ranges'])

                    if self.include_align_tags:
                        align_tags = cand['align_tags']
                    else:
                        align_tags = None

                    cand_param_dict = {
                        'original_id': o_id,
                        'text_tokens': text_tokens,
                        'schema_tokens': schema_tokens,
                        'schema_column_ids': schema_column_ids,
                        'tabert_tables': tabert_tables,
                        'text_audio_feats': text_audio_feats_list,
                        'schema_audio_feats': schema_audio_feats_list,
                        'align_tags': align_tags,
                    }

                    yield self.single_to_instance(cand_param_dict)






