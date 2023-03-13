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
from SpeakQL.Allennlp_models.utils.misc_utils import Load_CMU_Dict, WordPronDist, WordPronSimilarity, ConstructWordSimMatrix

import table_bert
from table_bert import TableBertModel
from .reader_utils import extractAudioFeatures, extractAudioFeatures_NoPooling, \
    extractRawAudios, extractAudioFeatures_NoPooling_Wav2vec, \
    dbToTokens, dbToTokensWithColumnIndexes, read_DB, Get_align_tags

from dataset_readers.rewriter_reader_base import SpiderASRFixerReader_Base


AUDIO_DIM = 136
AUDIO_DIM_NO_POOLING = 68
AUDIO_DIM_WAV2VEC_PROJ = 64

PHONEME_VOCAB_NAMESPACE = "phonemes"


@DatasetReader.register('spider_ASR_rewriter_reader_tagger_comb_new')
class SpiderASRRewriterReader_Tagger_Combined_new(SpiderASRFixerReader_Base):
    '''
    Sequence tagging + ILM rewrite generation, tagger part
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
                 use_phoneme_inputs: bool = False,      # include phoneme-level input
                 use_phoneme_labels: bool = False,      # include phoneme labels (including mlabels)
                 default_phoneme_slices: int = 4,       # Slice non-phoneme token audios into how many parts
                 specify_full_path: bool = False,   # For self._read(), whether file_path includes a full path (split:full_path) or split only
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
                 include_gold_tags: bool = True,
                 use_tagger_prediction: bool = False,
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

        super().__init__(
                 tables_json_fname=tables_json_fname,
                 dataset_dir=dataset_dir,
                 databases_dir=databases_dir,
                 use_db_input=use_db_input,
                 use_db_cells=use_db_cells,
                 db_cells_in_bracket=db_cells_in_bracket,
                 db_cells_split=db_cells_split,
                 use_schema_audio=use_schema_audio,
                 use_tabert=use_tabert,
                 tabert_model_path=tabert_model_path,
                 use_phoneme_inputs=use_phoneme_inputs,
                 use_phoneme_labels=use_phoneme_labels,
                 default_phoneme_slices=default_phoneme_slices,
                 specify_full_path=specify_full_path,
                 query_audio_dir=query_audio_dir,
                 src_token_indexers=src_token_indexers,
                 tgt_token_indexers=tgt_token_indexers,
                 ph_token_indexers=ph_token_indexers,
                 audio_feats_type=audio_feats_type,
                 wav2vec_model_path=wav2vec_model_path,
                 pronun_dict_path=pronun_dict_path,
                 db_tok2phs_dict_path=db_tok2phs_dict_path,
                 max_sequence_len=max_sequence_len,
                 include_align_tags=include_align_tags,
                 aux_probes=aux_probes,
                 prototype_dict_path=prototype_dict_path,
                 tabert_column_type_source=tabert_column_type_source,
                 samples_limit=samples_limit,
                 cands_limit=cands_limit,
                 debug=debug,
                 debug_db_ids=debug_db_ids,
                 cpu=cpu,
                 shuffle=shuffle,
                 lazy=lazy)

        self.include_gold_tags = include_gold_tags

    @overrides
    def text_to_instance(self,
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
                         align_tags: List[str],
                         ## START: extra args
                         rewriter_tags: List[str]) -> Instance:

        fields, metadata = self._building_basic_tensors_and_fields(
                    original_id=original_id,
                    text_tokens=text_tokens,
                    schema_tokens=schema_tokens,
                    schema_column_ids=schema_column_ids,
                    sql_query_tokens=sql_query_tokens,
                    tabert_tables=tabert_tables,
                    text_audio_feats=text_audio_feats,
                    schema_audio_feats=schema_audio_feats,
                    text_token_phonemes=text_token_phonemes,
                    text_token_phoneme_feats=text_token_phoneme_feats,
                    schema_token_phonemes=schema_token_phonemes,
                    schema_token_phoneme_feats=schema_token_phoneme_feats,
                    align_tags=align_tags)

        text_len = metadata["text_len"]
        schema_len = metadata["schema_len"]
        concat_len = metadata["concat_len"]
        concat_sentence_field = fields["sentence"]

        concat_tokens_lower = [Token(self._maybe_lower(t)) for t in metadata["concat_tokens"]]
        metadata["source_tokens"] = concat_tokens_lower

        if rewriter_tags is not None:
            rewriter_tag_tokens_padded = rewriter_tags + ['O' for _ in range(concat_len - text_len)]            
            rewriter_tags_field = SequenceLabelField(labels=rewriter_tag_tokens_padded,
                                            sequence_field=concat_sentence_field,
                                            label_namespace='rewriter_tags')

            assert len(rewriter_tag_tokens_padded) == concat_len
            fields["rewriter_tags"] = rewriter_tags_field

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
    
    @overrides
    def _text_to_instance_specific_proc(self, cand, text_to_instance_kwargs, cand_list=None) -> bool:
        if self.include_gold_tags:
            rewriter_tags = cand['rewriter_tags']
        else:
            rewriter_tags = None

        text_to_instance_kwargs['rewriter_tags'] = rewriter_tags
        return True


