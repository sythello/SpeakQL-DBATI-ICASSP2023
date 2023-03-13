from typing import Iterator, List, Dict, Union, Optional, cast
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn import ModuleList

import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, MultiLabelField, SequenceLabelField, ArrayField, MetadataField, ListField
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
import string
import re
import editdistance
import pdb

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer

from fairseq.models.wav2vec import Wav2VecModel

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema
from SpeakQL.Allennlp_models.utils.misc_utils import Load_CMU_Dict, Load_DB_Tok2phs_Dict, \
    WordPronDist, WordPronSimilarity, ConstructWordSimMatrix, \
    tokens_to_phonemes, strip_stress

import table_bert
from table_bert import TableBertModel
from .reader_utils import extractAudioFeatures, extractAudioFeatures_NoPooling, \
    extractRawAudios, extractAudioFeatures_NoPooling_Wav2vec, \
    dbToTokens, dbToTokensWithColumnIndexes, dbToTokensWithAddCells, \
    read_DB, Get_align_tags, load_DB_content, collect_DB_toks_dict, text_cell_to_toks



AUDIO_DIM = 136
AUDIO_DIM_NO_POOLING = 68
AUDIO_DIM_WAV2VEC_PROJ = 64

PHONEME_VOCAB_NAMESPACE = "phonemes"
PHONEME_INDEXER_KEY = "phonemes"

class SpiderASRFixerReader_Base(DatasetReader):
    '''
    The Base class for Rewriter reader (tagger, ILM, s2s)
    Now is also used for non-siamese reranker. But need to refactor (create separate sub-classes) for siamese reranker
    '''
    def __init__(self,
                 tables_json_fname: str,
                 dataset_dir: str,
                 databases_dir: str,
                 use_db_input: bool = True,             # If false, do ablation w/o db input 
                 use_db_cells: str = None,              # [None, 'gold', 'gold,noise={k}' 'K={k}', 'debug']
                 db_cells_in_bracket: bool = False,
                 db_cells_split: bool = False,          # If true, split cells in input seq using text_cell_to_toks
                 use_schema_audio: bool = True,         # use the (synthesized) audio of schema tokens; overrided by "use_db_cells" (for back-compat)
                 use_tabert: bool = False,
                 tabert_model_path: str = None,
                 use_phoneme_inputs: bool = False,      # include phoneme-level audio slicing input
                 use_phoneme_labels: bool = False,      # include phoneme labels (including mlabels)
                 default_phoneme_slices: int = 4,       # Slice non-phoneme token audios into how many parts
                 specify_full_path: bool = False,       # For self._read(), whether file_path includes a full path (split:full_path) or split only
                 ## query_audio_dir:
                 # If str: the dir containing {id}.wav files
                 # If dict: Dict[ds, audio_dir], ds in ['train', 'dev', 'test']
                 # If None, use default ({dataset_dir}/{sub_dir}/speech_wav)
                 query_audio_dir: Union[str, dict] = None,
                 src_token_indexers: Dict[str, TokenIndexer] = None,
                 tgt_token_indexers: Dict[str, TokenIndexer] = None,
                 ph_token_indexers: Dict[str, TokenIndexer] = None,
                 audio_feats_type: str = 'mfcc',            # ['mfcc', 'wav2vec', 'raw']
                 wav2vec_model_path: str = None,
                 pronun_dict_path: str = None,              # Path to cmudict
                 db_tok2phs_dict_path: str = None,          # Path to db_tok2phs.json
                 max_sequence_len: int = 300,
                 max_tokenized_sequence_len: int = 500,     # Limit for bert
                 include_align_tags: bool = False,  # Include align tags (token match/mismatch within other cands, [SAME]/[DIFF-n])
                 ## probes
                 aux_probes: Dict[str, bool] = None,        # Keys: 'utter_mention_schema', 'schema_dir_mentioned', 'schema_indir_mentioned'
                 prototype_dict_path: str = None,
                 tabert_column_type_source: str = 'spider', # ['spider', 'sqlite', 'sqlite_legacy']
                 samples_limit: int = None,                 # Only using this number of samples, for analysis; None = use all
                 cands_limit: int = None,                   # Only using this number of cands per sample (e.g. for test can set to 1)
                 debug: bool = False,
                 debug_db_ids: List[str] = None,        # only test for a list of problematic dbs
                 cpu: bool = False,
                 shuffle: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.tables_json_fname = tables_json_fname
        self.dataset_dir = dataset_dir
        self.databases_dir = databases_dir
        self.use_db_input = use_db_input
        self.use_db_cells = use_db_cells
        self.db_cells_in_bracket = db_cells_in_bracket
        self.db_cells_split = db_cells_split
        self.use_schema_audio = use_schema_audio
        if use_db_cells is not None:
            self._db_content_cache = dict()     # Dict[db_id, Dict[table_name, Dict[column_name, List[cells]]]]
        if use_db_cells is not None and use_schema_audio:
            print(f'WARNING: use_db_cells = {use_db_cells} and use_schema_audio = {use_schema_audio} are both enabled.')
            print(f'Will keep use_db_cells = {use_db_cells} and set use_schema_audio to False.')
            self.use_schema_audio = False

        self.use_tabert = use_tabert
        self.tabert_model_path = tabert_model_path
        if use_tabert:
            assert tabert_model_path is not None
            self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)

            assert tabert_column_type_source in ['spider', 'sqlite', 'sqlite_legacy']
            self.tabert_column_type_source = tabert_column_type_source

            self._tabert_tables_cache = dict()  # Dict[db_id, Dict[table_name, table_bert.Table]]

        # TODO: use these
        # If use phoneme input, then include phoneme labels, even if not using labels in model
        # (for implementation simplicity)
        self.use_phoneme_inputs = use_phoneme_inputs    # Whether using phoneme-level input
        self.use_phoneme_labels = use_phoneme_labels    # Whether using phoneme input and/or mlabels
        # if use_phoneme_inputs:
        #     assert use_phoneme_labels
        self.default_phoneme_slices = default_phoneme_slices

        self.specify_full_path = specify_full_path

        self.query_audio_dir = query_audio_dir

        self.include_align_tags = include_align_tags

        self.src_token_indexers = src_token_indexers # Can use BERT, word-level, char-level, etc.
        self.tgt_token_indexers = tgt_token_indexers # Should only be word-level 
        self.ph_token_indexers = ph_token_indexers   # Should only be word-level (SingleId)
        ## Back-compatible
        if ph_token_indexers is None:
            self.ph_token_indexers = {PHONEME_INDEXER_KEY: SingleIdTokenIndexer(namespace=PHONEME_VOCAB_NAMESPACE)}

        self.max_sequence_len = max_sequence_len
        self.max_tokenized_sequence_len = max_tokenized_sequence_len
        self.debug = debug
        self.debug_db_ids = debug_db_ids
        self.cpu = cpu
        self.samples_limit = samples_limit
        self.cands_limit = cands_limit


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
        
        # self.include_gold_rewrite_seq = include_gold_rewrite_seq
        # self.use_tagger_prediction = use_tagger_prediction

        # probes
        if aux_probes is not None:
            self.probe_utter_mention_schema = aux_probes['utter_mention_schema']
            self.probe_schema_dir_mentioned = aux_probes['schema_dir_mentioned']
            self.probe_schema_indir_mentioned = aux_probes['schema_indir_mentioned']
        else:
            self.probe_utter_mention_schema = False
            self.probe_schema_dir_mentioned = False
            self.probe_schema_indir_mentioned = False


        self.prototype_dict_path = prototype_dict_path
        if prototype_dict_path is not None:
            with open(prototype_dict_path, 'rb') as f:
                self.prototype_dict = pickle.load(f)
        else:
            self.prototype_dict = None

        self.pronun_dict_path = pronun_dict_path
        self.db_tok2phs_dict_path = db_tok2phs_dict_path
        if pronun_dict_path is not None:
            self.word2pron, self.pron2word = Load_CMU_Dict(pronun_dict_path)
        else:
            self.word2pron = defaultdict(set) # all possible prons, Dict[str, Set[Tuple[str]]]
            self.pron2word = defaultdict(set) # all possible words, Dict[Tuple[str], Set[str]]

        if db_tok2phs_dict_path is not None:
            _w2p, _p2w = Load_DB_Tok2phs_Dict(db_tok2phs_dict_path)
            for tok, phs in _w2p.items():
                if tok not in self.word2pron:
                    self.word2pron[tok].update(phs)
                ## for pron2word: not used anyway, skipped for now 

        print(f'Joint word2pron size = {len(self.word2pron)}')
        print(list(self.word2pron.items())[50000:100001:10000])


    @staticmethod
    def _token_to_ids(tokens: List[Union[Token, str]]) -> List[int]:
        ids = dict()
        out = []
        for token in tokens:
            if isinstance(token, Token):
                token = token.text
            out.append(ids.setdefault(token, len(ids)))
        return out

    @staticmethod
    def _maybe_lower(t: str):
        ## Possibly need to manually update when adding special tokens

        # if len(t) > 0 and t[0] in '[@':
        if t in {'[ANS]', '[CLS]', '[SEP]', '[MASK]', '@@PADDING@@', '@@UNKNOWN@@', START_SYMBOL, END_SYMBOL}:
            ## special, not lower
            pass
        else:
            t = t.lower()
        return t

    @staticmethod
    def _rewriter_tags_to_span_ids(rewriter_tags):
        source_span_ids = []
        _curr_span = []
        for i, _tag in enumerate(rewriter_tags):
            if _tag.endswith('EDIT'):
                _curr_span.append(i)
            if (_tag in {'L-EDIT', 'U-EDIT'} or (not _tag.endswith('EDIT'))) and len(_curr_span) > 0:
                # end the span
                source_span_ids.append(list(_curr_span))
                _curr_span = []

        if len(_curr_span) > 0:
            print(f'WARNING: unclosed edit span: {rewriter_tags}')

        return source_span_ids


    def _token_to_phonemes(self, token):
        # Return the first pronunciation phonemes from self.word2pron
        # return list(self.word2pron[token.lower()])[0]
        return sorted(list(self.word2pron[token.lower()]))[0]

    def text_to_instance(self):
        ## In sub-classes, should override
        ## should call self._building_basic_tensors_and_fields() and add task/model-specific fields
        raise NotImplementedError

    def _building_basic_tensors_and_fields(self,
                         original_id: int,
                         text_tokens: List[Token],
                         schema_tokens: List[Token],
                         schema_column_ids: List[int],      # To retreive column encodings from TaBERT
                         sql_query_tokens: List[Token],
                         tabert_tables: Dict[str, table_bert.Table],
                         text_audio_feats: List[np.ndarray],
                         schema_audio_feats: List[np.ndarray],
                         text_token_phonemes: List[List[str]],
                         text_token_phoneme_feats: List[List[np.ndarray]],
                         schema_token_phonemes: List[List[str]],
                         schema_token_phoneme_feats: List[List[np.ndarray]],
                         align_tags: List[str]) -> Instance:

        if self.use_db_input:
            concat_tokens = text_tokens + [Token('[SEP]')] + schema_tokens
            concat_audio_feats = text_audio_feats + [np.zeros((1, self.audio_feats_dim))] + schema_audio_feats
        else:
            # Ablation for -db
            concat_tokens = text_tokens
            concat_audio_feats = text_audio_feats

        ## Truncate schema parts if concat is too long: decide the excess_len to trunk off
        if len(concat_tokens) > self.max_sequence_len:
            excess_len = len(concat_tokens) - self.max_sequence_len
        else:
            excess_len = 0

        ## May still be too long due to tokenization; further truncate if so
        while True:
            length_ok = True
            for indexer_key, indexer in self.src_token_indexers.items():
                if indexer_key != 'bert':
                    continue
                ## only need to check for bert
                ## bert indexed_dict.keys() = dict_keys(['token_ids', 'mask', 'type_ids', 'offsets', 'wordpiece_mask'])
                _temp_concat_seq = ' '.join([t.text for t in concat_tokens[:-excess_len]])
                _tokenized_seq = indexer._tokenizer.tokenize(_temp_concat_seq)
                _tokenized_len = len(_tokenized_seq)
                if _tokenized_len > self.max_tokenized_sequence_len:
                    length_ok = False
                    break
            if length_ok:
                break
            else:
                excess_len += 10
                # print(f'** Tokenized_len = {_tokenized_len}')

        ## After determining the excess_len, do the truncation
        if excess_len > 0:
            concat_tokens = concat_tokens[:-excess_len]
            schema_tokens = schema_tokens[:-excess_len]
            schema_column_ids = schema_column_ids[:-excess_len]
            schema_audio_feats = schema_audio_feats[:-excess_len]
            concat_audio_feats = concat_audio_feats[:-excess_len]
            if schema_token_phonemes is not None:
                schema_token_phonemes = schema_token_phonemes[:-excess_len]
            if schema_token_phoneme_feats is not None:
                schema_token_phoneme_feats = schema_token_phoneme_feats[:-excess_len]


        
        assert len(concat_audio_feats) == len(concat_tokens), f"text:{len(text_tokens)}, text_audio:{len(text_audio_feats)}, " + \
            f"schema:{len(schema_tokens)}, schema_audio:{len(schema_audio_feats)}, " + \
            f"concat_tokens:{len(concat_tokens)}, concat_audio_feats:{len(concat_audio_feats)}"
        audio_lens = [_audio_feats.shape[0] for _audio_feats in concat_audio_feats]
        audio_lens_max = max(audio_lens)
        
        text_mask = np.array([1] * len(text_tokens) + [0] * (len(concat_tokens) - len(text_tokens)), dtype=np.int)
        schema_mask = np.array([0] * (len(concat_tokens) - len(schema_tokens)) + [1] * len(schema_tokens), dtype=np.int)
        audio_mask = np.array([
            [1] * a_len + [0] * (audio_lens_max - a_len) for a_len in audio_lens
        ], dtype=np.int)

        metadata = {
            "original_id": original_id,
            "text_len": len(text_tokens),
            "schema_len": len(schema_tokens),
            "concat_len": len(concat_tokens),
            "text_tokens": [str(t) for t in text_tokens],
            "schema_tokens": [str(t) for t in schema_tokens],
            "concat_tokens": [str(t) for t in concat_tokens],
        }

        if self.use_tabert:
            # For TaBERT: Need to do tokenize for each word, and keep their offsets; later use these to reconstruct word encodings
            text_tokenized = []
            offsets = []
            for _tok in text_tokens:
                _tok_pieces = self.tabert_model.tokenizer.tokenize(_tok.text)
                _st = len(text_tokenized)
                _ed = _st + len(_tok_pieces)
                offsets.append((_st, _ed))
                text_tokenized.extend(_tok_pieces)
        
            metadata["tabert_tables"] = tabert_tables
            metadata["text_tokenized"] = text_tokenized
            metadata["text_offsets"] = offsets

        
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
                  "audio_mask": audio_mask_field}

        ## START phonemes
        if self.use_phoneme_labels:
            # This is DIFFERENT from legacy: we always give phoneme labels and multilabels

            # List[MultiLabelField]
            token_phoneme_multilabel_fields = []
            # List[ListField[LabelField]]
            token_phoneme_label_fields = []
            # List[ArrayField(ph_len,)]
            token_phoneme_label_mask_fields = []

            if self.use_db_input:
                concat_token_phonemes = text_token_phonemes + [['[NONE]']] + schema_token_phonemes
            else:
                concat_token_phonemes = text_token_phonemes

            for i, _tok_phs in enumerate(concat_token_phonemes):
                ## _tok_phs: List[str] or None

                if _tok_phs is None:
                    _tok_ph_mlabel_field = MultiLabelField([], label_namespace=PHONEME_VOCAB_NAMESPACE)
                    ## TextField to workaround the padding=-1 problem with LabelField
                    _tok_ph_label_field = TextField([Token('[NONE]')], self.ph_token_indexers)
                    _tok_ph_label_mask_field = ArrayField(np.zeros((1,))) 
                else:
                    # if len(_tok_phs) > 50:
                    #     print(f"WARNING: long pronunciation: token = {concat_tokens[i]}, _tok_phs = {_tok_phs}")
                    _tok_ph_mlabel_field = MultiLabelField(list(set(_tok_phs)), label_namespace=PHONEME_VOCAB_NAMESPACE)
                    _tok_ph_label_field = TextField([Token(_ph) for _ph in _tok_phs], self.ph_token_indexers)
                    _tok_ph_label_mask_field = ArrayField(np.ones((len(_tok_phs),))) 
                    
                token_phoneme_multilabel_fields.append(_tok_ph_mlabel_field)
                token_phoneme_label_fields.append(_tok_ph_label_field)
                token_phoneme_label_mask_fields.append(_tok_ph_label_mask_field)
                
            # phoneme_multilabels [bs * seq_len]
            phoneme_multilabels_field = ListField(token_phoneme_multilabel_fields)
            # phoneme_labels [bs * seq_len * ph_len]
            phoneme_labels_field = ListField(token_phoneme_label_fields)
            # phoneme_labels_mask [bs * seq_len * ph_len]
            phoneme_label_mask_field = ListField(token_phoneme_label_mask_fields)
            
            fields['phoneme_multilabels'] = phoneme_multilabels_field
            fields['phoneme_labels'] = phoneme_labels_field
            fields['phoneme_label_mask'] = phoneme_label_mask_field


        if self.use_phoneme_inputs:
            # This is DIFFERENT from legacy: this is only for phoneme-based audio slicing feats 

            # List[ListField[ArrayField(ph_audio_len, audio_dim)]]
            token_phoneme_audio_fields = []
            # List[ListField[ArrayField(ph_audio_len,)]
            token_phoneme_audio_mask_fields = []

            if self.use_db_input:
                concat_token_phoneme_feats = text_token_phoneme_feats + [[np.zeros((1, 1))]] + schema_token_phoneme_feats
            else:
                concat_token_phoneme_feats = text_token_phoneme_feats
            assert len(text_token_phonemes) == len(text_token_phoneme_feats) == len(text_tokens), \
                (len(text_token_phonemes), len(text_token_phoneme_feats), len(text_tokens))
            assert len(schema_token_phonemes) == len(schema_token_phoneme_feats) == len(schema_tokens), \
                (len(schema_token_phonemes), len(schema_token_phoneme_feats), len(schema_tokens))
            assert len(concat_token_phonemes) == len(concat_token_phoneme_feats) == len(concat_tokens), \
                (len(concat_token_phonemes), len(concat_token_phoneme_feats), len(concat_tokens))

            for _tok_phs, _tok_ph_feats in zip(concat_token_phonemes, concat_token_phoneme_feats):
                ## _tok_phs: List[str] or None
                ## _tok_ph_feats: List[array(ph_a_len, a_dim)] (list.len=self.default_phoneme_slices if _tok_phs=None)

                if _tok_phs is None:
                    assert len(_tok_ph_feats) == self.default_phoneme_slices, (len(_tok_ph_feats), [_ph_feats.shape for _ph_feats in _tok_ph_feats])
                    _tok_ph_audio_field = ListField([ArrayField(_ph_feats) for _ph_feats in _tok_ph_feats])
                    _tok_ph_audio_mask_field = ListField([ArrayField(np.ones((len(_ph_feats),))) for _ph_feats in _tok_ph_feats])
                else:
                    assert len(_tok_phs) == len(_tok_ph_feats), (len(_tok_phs), len(_tok_ph_feats), _tok_phs, _tok_ph_feats)
                    _tok_ph_audio_field = ListField([ArrayField(_ph_feats) for _ph_feats in _tok_ph_feats])
                    _tok_ph_audio_mask_field = ListField([ArrayField(np.ones((len(_ph_feats),))) for _ph_feats in _tok_ph_feats])
                    
                # print([lf.label for lf in _tok_ph_label_field.field_list])
                # print(_tok_ph_audio_field.sequence_length())
                token_phoneme_audio_fields.append(_tok_ph_audio_field)
                token_phoneme_audio_mask_fields.append(_tok_ph_audio_mask_field)

            # phoneme_audio_field =  [bs * seq_len * phon_len * audio_len * audio_dim]
            # phoneme_mask (same shape as phoneme labels, for which has/doesn’t have label)
            phoneme_audio_field = ListField(token_phoneme_audio_fields)
            phoneme_audio_mask_field = ListField(token_phoneme_audio_mask_fields)

            fields['phoneme_audio_feats'] = phoneme_audio_field
            fields['phoneme_audio_mask'] = phoneme_audio_mask_field
        ## END phonemes


        ## START probes
        stemmer = PorterStemmer()
        if self.probe_utter_mention_schema:
            _labels = []

            _utter_tokens_stem = [stemmer.stem(str(_t)) for _t in text_tokens]
            _schema_tokens_stem = [stemmer.stem(str(_t)) for _t in schema_tokens if str(_t) not in string.punctuation]
            for _ut in _utter_tokens_stem:
                if _ut in _schema_tokens_stem:
                    _labels.append(1)
                else:
                    _labels.append(0)

            _labels = _labels + [0] * (len(concat_tokens) - len(text_tokens))
            fields['utter_mention_schema_labels'] = ArrayField(np.array(_labels, dtype=int))

        if self.probe_schema_dir_mentioned:
            _labels = []
        
            _utter_tokens_stem = [stemmer.stem(str(_t)) for _t in text_tokens]
            _schema_tokens_stem = [stemmer.stem(str(_t)) for _t in schema_tokens]
            for _ut in _schema_tokens_stem:
                if _ut in string.punctuation:
                    _labels.append(0)
                elif _ut in _utter_tokens_stem:
                    _labels.append(1)
                else:
                    _labels.append(0)

            _labels = [0] * (len(concat_tokens) - len(schema_tokens)) + _labels
            fields['schema_dir_mentioned_labels'] = ArrayField(np.array(_labels, dtype=int))

        if self.probe_schema_indir_mentioned:
            schema_id2names = defaultdict(str)
            _tmp_ids = []
            _tmp_toks = []
            for i, _tok in enumerate(schema_tokens):
                _tok = str(_tok)
                if _tok in ',.:':
                    # end of name 
                    _name = '_'.join(_tmp_toks)
                    for _idx in _tmp_ids:
                        schema_id2names[_idx] = _name
                    
                    _tmp_ids = []
                    _tmp_toks = []
                else:
                    _tmp_ids.append(i)
                    _tmp_toks.append(_tok)
            # assert _tmp_ids == _tmp_toks == []    # Might not be true due to truncating 
            
            sql_schema_names = []
            for i, q_tok in enumerate(sql_query_tokens):
                q_tok = str(q_tok)
                _toks = re.split(r'^[Tt]\d+\.', q_tok)
                if len(_toks) > 1:
                    assert len(_toks) == 2 and _toks[0] == '', q_tok
                for _tok in _toks:
                    if _tok.isupper() or _tok in string.punctuation:
                        # "T1", ",", "", etc.
                        continue
                    assert ' ' not in _tok, q_tok
                    sql_schema_names.append(_tok.lower()) # can have '_' in name 
        
            _labels = []
            for i in range(len(schema_tokens)):
                if schema_id2names[i] in sql_schema_names:
                    _labels.append(1)
                else:
                    _labels.append(0)

            _labels = [0] * (len(concat_tokens) - len(schema_tokens)) + _labels
            fields['schema_indir_mentioned_labels'] = ArrayField(np.array(_labels, dtype=int))
        ## END probes

        if align_tags is not None:
            align_tag_tokens_padded = align_tags + ['[O]' for _ in range(len(concat_tokens) - len(text_tokens))]            
            align_tags_field = SequenceLabelField(labels=align_tag_tokens_padded,
                                            sequence_field=concat_sentence_field,
                                            label_namespace='align_tags')

            assert len(align_tag_tokens_padded) == len(concat_tokens)
            fields["align_tags"] = align_tags_field


        if self.pronun_dict_path is not None:
            ## Add pron map as "reference" attention map
            try:
                pron_sim_map = ConstructWordSimMatrix(
                    sen=[str(t) for t in concat_tokens],
                    word2pron=self.word2pron,
                    sim_func=WordPronSimilarity,
                    default_val=1e-4,
                    skip_punct=True)
            except Exception as e:
                print(original_id)
                print(concat_tokens)
                raise e

            ## Normalize by sum (TODO: other ways, like softmax?)
            # _row_sum = pron_sim_map.sum(axis=-1)
            # _row_nonzero = _row_sum > 0
            # pron_sim_map[_row_nonzero] /= np.expand_dims(_row_sum[_row_nonzero], -1)
            pron_sim_map /= pron_sim_map.sum(axis=-1)

            fields["ref_att_map"] = ArrayField(pron_sim_map)

        # fields["metadata"] = MetadataField(metadata)

        return fields, metadata
    

    def _text_to_instance_specific_proc(self, cand, text_to_instance_kwargs, cand_list=None) -> bool:
        ## cand: Dict in dataset (dataset = List[spider_example:List[cand:Dict]])
        ## In sub-classes, should modify(add) kwargs to 'text_to_instance_kwargs'
        pass

    def _schema_processing_with_audio(self, cand, preloaded_databases):
        ''' The early version, with schema token audio feats. Not supporting DB content cells. '''
        ''' Now schema audio feats is toggled with "use_schema_audio" arg, could rename to "_schema_processing_wo_content_cells". '''

        db_id = cand['db_id']
        db_schema = preloaded_databases[db_id]['schema']

        sql_query_tokens = [Token(tok) for tok in cand['query_toks']]

        # Schema processing
        if self.use_db_input:
            if self.use_db_cells is None:
                schema_sentence, schema_column_ids = dbToTokensWithColumnIndexes(db_schema)
            elif self.use_db_cells == 'debug':
                add_cells = dict()
                for table_name, table in db_schema.items():
                    for column in table.columns[:-1]:
                        add_cells[(table_name, column.name)] = [column.text]
                schema_sentence, schema_column_ids = dbToTokensWithAddCells(db_schema, add_cells=add_cells, cells_in_bracket=self.db_cells_in_bracket)
                print('#'+'='*30+'#')
                print(db_id)
                print(schema_sentence)
            else:
                raise NotImplementedError(self.use_db_cells)

            schema_tokens = [Token(tok) for tok in schema_sentence]

            if self.use_schema_audio:
                schema_audio_feats_list = []
                for token in schema_sentence:
                    if token in ',.:;':
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
            else:
                schema_audio_feats_list = [np.zeros((1, self.audio_feats_dim)) for tok in schema_tokens]

            ## phonemes (Schema)
            if self.use_phoneme_labels:
                assert self.audio_feats_type == 'mfcc'  # now labels can go alone as tags, this can be removed

                schema_phoneme_json = os.path.join(self.dataset_dir, 'db', 'schema_phonemes.json')
                with open(schema_phoneme_json, 'r') as f:
                    schema_phoneme_dict = json.load(f)

                schema_token_phonemes = []      ## List[List[str] or None]

                for token in schema_sentence:
                    if token in ',.:':  # why ";" is not there??
                        schema_token_phonemes.append(None)
                    elif token not in schema_phoneme_dict:
                        ## Words with no available phonemes
                        schema_token_phonemes.append(None)
                    else:
                        _tok_phs = schema_phoneme_dict[token]['phonemes']
                        schema_token_phonemes.append(_tok_phs)
            else:
                schema_token_phonemes = None

            # phoneme feats
            if self.use_phoneme_inputs:
                assert self.use_phoneme_labels
                schema_token_phoneme_feats = [] ## List[List[np.array(ph_audio_len, audio_dim)]]

                for token in schema_sentence:
                    if token in ',.:;':
                        schema_token_phoneme_feats.append([np.zeros((1, self.audio_feats_dim)) for _ in range(self.default_phoneme_slices)])
                    elif token not in schema_phoneme_dict:
                        ## Words with no available phonemes
                        # Use full token; can use pre-saved feats
                        feats_fname = os.path.join(self.dataset_dir, 'db', 'speech_feats_array', f'{token}.pkl')
                        with open(feats_fname, 'rb') as f:
                            feats_array = pickle.load(f).T      ## In pkl files, shapes are (68, a_len); T->(a_len, 68)
                        feats_array_splits = np.array_split(feats_array, self.default_phoneme_slices, axis=0)
                        schema_token_phoneme_feats.append(feats_array_splits)
                    else:
                        ## Has available phonemes (TODO: pre-compute these feats?)
                        _tok_ph_spans = schema_phoneme_dict[token]['phoneme_spans']
                        audio_fname = os.path.join(self.dataset_dir, 'db', 'speech_wav', f'{token}.wav')
                        # print(_tok_ph_spans)
                        _ph_feats = extractAudioFeatures_NoPooling(audio_fname, _tok_ph_spans)
                        schema_token_phoneme_feats.append(_ph_feats)
            else:
                schema_token_phoneme_feats = None
        else:
            ## Ablation: no db schema
            schema_column_ids = []
            schema_tokens = []
            schema_audio_feats_list = []
            schema_token_phonemes = []
            schema_token_phoneme_feats = []

        return {
            'schema_column_ids': schema_column_ids,
            'schema_tokens': schema_tokens,
            'schema_audio_feats_list': schema_audio_feats_list,
            'schema_token_phonemes': schema_token_phonemes,
            'schema_token_phoneme_feats': schema_token_phoneme_feats,
        }


    def _schema_processing_with_content_cells(self, cand, preloaded_databases):
        ''' The new version, with DB content cells. Not supporting DB token audio feats. '''

        db_id = cand['db_id']

        # Dict[table_name, Table]: schema only
        db_schema = preloaded_databases[db_id]['schema']
        # Dict[table_name, Dict[col_name, List[cells]]]
        db_tables_dict = preloaded_databases[db_id]['content']
        # Dict[tok, List[Tuple(table_name, column_name, cell)]]
        db_content_toks_dict = preloaded_databases[db_id]['content_toks']

        # sql_query_tokens = [Token(tok.lower()) for tok in cand['query_toks']]
        sql_query_tokens = [Token(tok) for tok in cand['query_toks']]

        # Schema processing
        if self.use_db_input:
            if self.use_db_cells is None:
                schema_sentence, schema_column_ids = dbToTokensWithColumnIndexes(db_schema)
            elif self.use_db_cells == 'debug':
                add_cells = dict()
                for table_name, table in db_schema.items():
                    for column in table.columns[:-1]:
                        add_cells[(table_name, column.name)] = [column.text]
                schema_sentence, schema_column_ids = dbToTokensWithAddCells(db_schema, add_cells=add_cells, cells_in_bracket=self.db_cells_in_bracket)
                print('#'+'='*30+'#')
                print(db_id)
                print(schema_sentence)
            elif self.use_db_cells.startswith('gold'):
                ## Adding cells with gold toks
                # db_tables_dict = load_DB_content(db_id=db_id, db_dir=self.databases_dir)
                # db_content_toks_dict = collect_DB_toks_dict(db_tables_dict)

                db_content_toks = set(db_content_toks_dict.keys())
                
                ## Find toks in edit tgt
                all_hit_toks = set()
                
                for edit_d in cand['rewriter_edits']:
                    hit_toks = (set(edit_d['tgt_text'].lower().split(' ')) & db_content_toks) - set(stopwords.words("english"))
                    if len(hit_toks) > 0:
                        # collect this edit 
                        all_hit_toks.update(hit_toks)

                add_cells = defaultdict(set)
                for tok in all_hit_toks:
                    ## Sometimes a tok appears in a lot of big cells (e.g. URLs); try to only use 1 here to avoid too long seq
                    for (table_name, column_name, cell) in db_content_toks_dict[tok][:1]:
                        ## lower() here is to be make sql column names compatible with tables.json
                        add_cells[(table_name, column_name.lower())].add(cell)

                if self.use_db_cells.startswith('gold,noise'):
                    ## Add some cells for noise
                    noise_k = int(self.use_db_cells.split('=')[-1])
                    noise_cell_cands = set()
                    for table_name, col_cells_dict in db_tables_dict.items():
                        for column_name, col_cells in col_cells_dict.items():
                            for _c in col_cells:
                                noise_cell_cands.add((table_name, column_name, _c))

                    if len(noise_cell_cands) < noise_k:
                        print(f'WARNING: not enough cells: db_id = {db_id}, cells = {noise_cell_cands}')
                        noise_cells = list(noise_cell_cands)
                    else:
                        noise_cells = random.sample(noise_cell_cands, k=noise_k)
                    for table_name, column_name, _c in noise_cells:
                        add_cells[(table_name, column_name.lower())].add(_c)

                # map back to Dict[List]
                add_cells = {k: list(v) for k, v in add_cells.items()}

                schema_sentence, schema_column_ids = dbToTokensWithAddCells(db_schema, add_cells=add_cells, cells_in_bracket=self.db_cells_in_bracket)

            elif self.use_db_cells.startswith('K='):
                ## Add the phoneme-retrieved cells based on source tokens in edit span
                ## Only works with ILM, where the rewriter tags are available (TODO: refactor, putting this under ILM)
                retrieve_k = int(self.use_db_cells.split('=')[-1])
                try:
                    _use_tagger_prediction = self.use_tagger_prediction
                except AttributeError as e:
                    print(f'The class of reader is "{type(self)}"; does not have attribute "use_tagger_prediction".')
                    raise e

                if _use_tagger_prediction:
                    # use predicted (evaling)
                    rewriter_tags = cand['tagger_predicted_rewriter_tags']
                    source_span_ids = self._rewriter_tags_to_span_ids(rewriter_tags)
                else:
                    # use gold (training)
                    rewriter_tags = cand['rewriter_tags']
                    source_span_ids = [edit_d['src_span'] for edit_d in cand['rewriter_edits']]
                    assert source_span_ids == self._rewriter_tags_to_span_ids(rewriter_tags), \
                        f"{rewriter_tags} || {source_span_ids}"

                db_content_toks = set(db_content_toks_dict.keys())

                # sel_db_toks = set()
                sel_db_toks = []
                # for edit_d in cand['rewriter_edits']:
                    # q_span = ' '.join([cand['question_toks'][i].lower() for i in edit_d['src_span']])
                for src_span in source_span_ids:
                    q_span = ' '.join([cand['question_toks'][i].lower() for i in src_span])
                    if q_span in stopwords.words("english"):
                        continue

                    # assert text_cell_to_toks(q_span) == [t for t in q_span.split(' ') if (t not in string.punctuation and not ''.join(t.split(',')).isnumeric())], q_span  ## assertion untrue, can have punct inside the word
                    q_span_phs = [ph for q_tok in text_cell_to_toks(q_span) for ph in self._token_to_phonemes(q_tok)]
                    if len(q_span_phs) == 0:
                        continue

                    db_tok_cand_list = []  # List[Tuple(tok, sim, char_sim)]
                    for db_tok_cand in db_content_toks:
                        _max_sim = 0
                        for db_tok_cand_phs in self.word2pron[db_tok_cand]:
                            ph_edist = editdistance.eval(q_span_phs, db_tok_cand_phs)   ## Not using WordPronSimilarity() since using span, not single word
                            ph_sim = 1 - float(ph_edist) / max(len(q_span_phs), len(db_tok_cand_phs))
                            _max_sim = max(_max_sim, ph_sim)

                        char_edist = editdistance.eval(q_span, db_tok_cand)
                        char_sim = 1 - float(char_edist) / max(len(q_span), len(db_tok_cand))
                        db_tok_cand_list.append((db_tok_cand, _max_sim, char_sim))

                    db_tok_cand_list.sort(key=lambda x: (-x[1], -x[2], x[0]))   # order: (high ph_sim, high char_sim, alphabet)
                    # sel_db_toks.update([x[0] for x in db_tok_cand_list[:retrieve_k]])
                    sel_db_toks.extend([x[0] for x in db_tok_cand_list[:retrieve_k] if x[0] not in sel_db_toks])

                add_cells = defaultdict(list)
                for tok in sel_db_toks:
                    ## Sometimes a tok appears in a lot of big cells (e.g. URLs); try to only use 1 here to avoid too long seq
                    for (table_name, column_name, cell) in db_content_toks_dict[tok][:1]:
                        # add_cells[(table_name, column_name.lower())].add(cell)
                        _cell_str = ' '.join(text_cell_to_toks(cell)) if self.db_cells_split else cell
                        # if _cell_str != cell:
                        #     print('Cell split:', cell, _cell_str)

                        ## lower() here is to be make sql column names compatible with tables.json
                        if _cell_str not in add_cells[(table_name, column_name.lower())]:
                            add_cells[(table_name, column_name.lower())].append(_cell_str)


                # map back to Dict[List]
                # add_cells = {k: list(v) for k, v in add_cells.items()}

                schema_sentence, schema_column_ids = dbToTokensWithAddCells(db_schema, add_cells=add_cells, cells_in_bracket=self.db_cells_in_bracket)

            else:
                raise NotImplementedError(self.use_db_cells)

            schema_tokens = [Token(tok) for tok in schema_sentence]
            schema_audio_feats_list = [np.zeros((1, self.audio_feats_dim)) for tok in schema_tokens]

            ## phonemes (Schema)
            ## Notice that this is DIFFERENT from the previous version!
            ## Previous version is based on audio alignment, including "sil", etc., and based on CMU dict only
            ## This version is simply phonemes, and phonemes are generated for OOV tokens 
            if self.use_phoneme_labels:
                # schema_phoneme_json = os.path.join(self.dataset_dir, 'db', 'db_tok2phs.json')
                # with open(schema_phoneme_json, 'r') as f:
                #     schema_phoneme_dict = json.load(f)

                schema_token_phonemes = []      ## List[List[str] or None]

                for token in schema_sentence:
                    if token in ',.:;()':
                        schema_token_phonemes.append(None)
                    else:
                        # split to deal with column names with digits, etc.
                        token_phs = []
                        for tok in text_cell_to_toks(token):
                            _tok_phs = self._token_to_phonemes(tok)
                            token_phs.extend(_tok_phs)
                        schema_token_phonemes.append(token_phs)
            else:
                schema_token_phonemes = None

            # phoneme feats
            schema_token_phoneme_feats = None
        else:
            ## Ablation: no db schema
            schema_column_ids = []
            schema_tokens = []
            schema_audio_feats_list = []
            schema_token_phonemes = []
            schema_token_phoneme_feats = []

        return {
            'schema_column_ids': schema_column_ids,
            'schema_tokens': schema_tokens,
            'schema_audio_feats_list': schema_audio_feats_list,
            'schema_token_phonemes': schema_token_phonemes,
            'schema_token_phoneme_feats': schema_token_phoneme_feats,
        }


    def _read(self, file_path: str) -> Iterator[Instance]:
        # split file_path by ';', then feed to _read_one_path (which is originally _read())
        file_path_list = file_path.split(';')
        sample_iters_list = [self._read_one_path(p) for p in file_path_list]
        return itertools.chain(*sample_iters_list)

    def _read_one_path(self, file_path: str) -> Iterator[Instance]:
        # file_path: if self.specify_full_path, this is "split:full_path";
        #   otherwise, this is only the dataset split, e.g. train, dev, test.
        if self.specify_full_path:
            ds, dataset_json_path = file_path.split(':')
            sub_dir = ('dev' if ds == 'test' else ds)
        else:
            ds = file_path  # dataset split
            sub_dir = ('dev' if ds == 'test' else ds)
            dataset_json_path = os.path.join(self.dataset_dir, sub_dir, '{}_rewriter+phonemes.json'.format(ds))
        
        databases = read_dataset_schema(self.tables_json_fname)     ## Dict[db_id, Dict[table_name, Table]]: schema only
        all_db_content_table_dicts = dict()     ## Dict[db_id, Dict[table_name, Dict[col_name, List[cells]]]]
        all_db_content_toks_dicts = dict()      ## Dict[db_id, Dict[tok, List[Tuple(table_name, column_name, cell)]]]
        for db_id in databases.keys():
            db_content_tables_dict = load_DB_content(db_id=db_id, db_dir=self.databases_dir)
            db_content_toks_dict = collect_DB_toks_dict(db_content_tables_dict)
            all_db_content_table_dicts[db_id] = db_content_tables_dict
            all_db_content_toks_dicts[db_id] = db_content_toks_dict
        preloaded_databases = {db_id : {'schema': databases[db_id],
                                        'content': all_db_content_table_dicts[db_id],
                                        'content_toks': all_db_content_toks_dicts[db_id]}
                            for db_id in databases.keys()}


        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        
        if self.debug_db_ids is not None:
            ## Only test these dbs
            _dataset_json = [cand_list for cand_list in dataset_json if len(cand_list) > 0 and cand_list[0]['db_id'] in self.debug_db_ids]
            if len(_dataset_json) == 0:
                _db_id = dataset_json[0][0]['db_id']
                print(f'Warning: no samples found with db_id in {self.debug_db_ids}. Using db_id = {_db_id} instead')
                _dataset_json = [cand_list for cand_list in dataset_json if len(cand_list) > 0 and cand_list[0]['db_id'] == _db_id]
            dataset_json = _dataset_json

        if self.samples_limit is not None:
            assert 0 < self.samples_limit <= len(dataset_json), \
                f'samples_limit should be in (0, {len(dataset_json)}], but got {self.samples_limit}'

            _all_ids = list(range(len(dataset_json)))
            _sample_ids = random.sample(_all_ids, k=self.samples_limit)
            dataset_json = [dataset_json[i] for i in _sample_ids]
        if self.debug:
            print('warning: argument "debug" is deprecated in speakql dataset_readers')
            dataset_json = dataset_json[:5] + dataset_json[5::400]
        
        ## Generate phonemes for oov words in question
        _question_oov_tokens = set()
        for i, cand_list in enumerate(dataset_json):
            for cand in cand_list:
                for tok in cand['question_toks']:
                    for t in text_cell_to_toks(tok):
                        if t.lower() not in self.word2pron:
                            _question_oov_tokens.add(t.lower())
        _question_oov_tokens = list(_question_oov_tokens)
        _question_oov_token_phs = tokens_to_phonemes(_question_oov_tokens)
        print(f'Question OOV: {len(_question_oov_tokens)}, {list(zip(_question_oov_tokens, _question_oov_token_phs))[:10]}')
        for t, phs in zip(_question_oov_tokens, _question_oov_token_phs):
            # notice that word2pron is Dict[tok:str, Set[pron:List[str]]]
            phs = tuple([strip_stress(_phone) for _phone in phs])
            self.word2pron[t].add(phs)
        
        for i, cand_list in enumerate(dataset_json):
            
            if len(cand_list) == 0:
                continue
            
            o_id = cand_list[0]['original_id']
            
            db_id = cand_list[0]['db_id'] # Maybe should combine the common fields for all candidates, like original_id, db_id, etc. 
            db_schema = databases[db_id]

            sql_query_tokens = [Token(tok) for tok in cand_list[0]['query_toks']]
            
            if self.use_db_cells is None:
                ## DB sentence is the same for cands, can compute together
                schema_processed_dict = self._schema_processing_with_audio(cand_list[0], preloaded_databases=preloaded_databases)
            
            if self.use_tabert:
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
            else:
                # Not using tabert
                tabert_tables = None

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
            if self.cands_limit is not None:
                cand_list = cand_list[:self.cands_limit]
            for cand in cand_list:
                if self.use_db_cells is None:
                    ## Already processed
                    assert isinstance(schema_processed_dict, dict)
                else:
                    schema_processed_dict = self._schema_processing_with_content_cells(cand, preloaded_databases=preloaded_databases)

                schema_column_ids = schema_processed_dict['schema_column_ids']
                schema_tokens = schema_processed_dict['schema_tokens']
                schema_audio_feats_list = schema_processed_dict['schema_audio_feats_list']
                schema_token_phonemes = schema_processed_dict['schema_token_phonemes']
                schema_token_phoneme_feats = schema_processed_dict['schema_token_phoneme_feats']

                ## For the sake of matching src-tgt, need to lower; but for some PLM need to keep case
                ## Leave for the instance maker
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

                ## phonemes (Utterance)
                if self.use_phoneme_labels:
                    if self.use_phoneme_inputs:
                        ## need to align with phoneme input
                        text_token_phonemes = [] ## List[List[str] or None]
                        for tok_phs in cand['token_phonemes']:
                            if tok_phs is None:
                                text_token_phonemes.append(tok_phs)
                            else:
                                text_token_phonemes.append([strip_stress(_ph) for _ph in tok_phs])
                    else:
                        ## not phoneme-level input - should align with how DB token phonemes are obtained (cmu dict - espeak inferer)
                        text_token_phonemes = []
                        for _tok in cand['question_toks']:
                            _tok_phs = []
                            for _t in text_cell_to_toks(_tok):
                                _tok_phs.extend(self._token_to_phonemes(_t))
                            text_token_phonemes.append(_tok_phs)

                        # print(cand['question_toks'])
                        # print('Tok-to-phs:', text_token_phonemes)
                        # print('From file:', cand['token_phonemes'])
                        # print('Stripped:', [(None if tok_phs is None else [strip_stress(_ph) for _ph in tok_phs]) for tok_phs in cand['token_phonemes']])
                        # print()
                else:
                    text_token_phonemes = None

                if self.use_phoneme_inputs:
                    text_token_phonemes_all_feats = []           ## List[List[np.array(ph_audio_len, audio_dim)]]
                    for _tok_ph_spans, _tok_span in zip(cand['token_phoneme_spans'], cand['span_ranges']):
                        _tok_st, _tok_ed = _tok_span
                        _tok_st = float(_tok_st)
                        _tok_ed = float(_tok_ed)

                        if _tok_ph_spans is None:
                            # Use full token audio; (later) not using for phoneme loss
                            # split timespan into N slices (N = self.default_phoneme_slices)
                            _times = np.linspace(_tok_st, _tok_ed, self.default_phoneme_slices + 1)
                            _ph_spans = [(_times[i], _times[i+1]) for i in range(self.default_phoneme_slices)]
                        else:
                            # move _tok_ph_spans according to token's own start time
                            _ph_spans = [(_tok_st + _ph_st, _tok_st + _ph_ed) for _ph_st, _ph_ed in _tok_ph_spans]

                        if self.audio_feats_type == 'mfcc':
                            _ph_feats = extractAudioFeatures_NoPooling(audio_fname, _ph_spans)
                        elif self.audio_feats_type == 'wav2vec':
                            _ph_feats = extractAudioFeatures_NoPooling_Wav2vec(audio_fname, _ph_spans, self.w2v_model)
                        elif self.audio_feats_type == 'raw':
                            _ph_feats = extractRawAudios(audio_fname, _ph_spans)
                        text_token_phonemes_all_feats.append(_ph_feats)
                else:
                    text_token_phonemes_all_feats = None
                
                if self.include_align_tags:
                    align_tags = cand['align_tags']
                else:
                    align_tags = None


                text_to_instance_kwargs = {
                    'original_id': o_id,
                    'text_tokens': text_tokens,
                    'schema_tokens': schema_tokens,
                    'schema_column_ids': schema_column_ids, # TaBERT
                    'sql_query_tokens': sql_query_tokens,
                    'tabert_tables': tabert_tables,
                    'text_audio_feats': text_audio_feats_list,
                    'schema_audio_feats': schema_audio_feats_list,
                    'text_token_phonemes': text_token_phonemes,
                    'text_token_phoneme_feats': text_token_phonemes_all_feats,
                    'schema_token_phonemes': schema_token_phonemes,
                    'schema_token_phoneme_feats': schema_token_phoneme_feats,
                    'align_tags': align_tags,
                }

                ## Modify the kwargs based on specific task/model
                _keep = self._text_to_instance_specific_proc(cand, text_to_instance_kwargs, cand_list)
                if _keep == False:
                    ## Sometimes we don't want to keep certain samples (e.g. for reranker, all cands equally good)
                    continue

                # if self.use_tagger_prediction:
                #     rewriter_tags = cand['tagger_predicted_rewriter_tags']
                # else:
                #     rewriter_tags = cand['rewriter_tags']

                # if self.include_gold_rewrite_seq:
                #     # rewrite_list = ['[START_RWT]']
                #     rewrite_list = []
                #     for _edit in cand['rewriter_edits']:
                #         rewrite_list.extend(_edit['tgt_text'].split(' '))
                #         rewrite_list.append('[ANS]')
                #     # rewrite_list.append('[END_RWT]')
                #     rewrite_tokens = [Token(w) for w in rewrite_list]
                # else:
                #     rewrite_tokens = None
                
                yield self.text_to_instance(**text_to_instance_kwargs)


if __name__ == '__main__':
    # expected: [[10]]
    sample_rewriter_tags_1 = ['O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'O-KEEP', 'U-EDIT']

    # expected: [[1,2,3], [4], [6], [10,11]]
    sample_rewriter_tags_2 = ["O-KEEP", "B-EDIT", "I-EDIT", "L-EDIT", "U-EDIT", "O-KEEP", "U-EDIT", "O-KEEP", "B-DEL", "L-DEL", "B-EDIT", "L-EDIT", "U-DEL", "O-KEEP"]

    _span_ids = SpiderASRFixerReader_Base._rewriter_tags_to_span_ids(sample_rewriter_tags_1)

    print(_span_ids)










