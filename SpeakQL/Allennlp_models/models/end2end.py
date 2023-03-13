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

@Model.register('speakql_end2end_model')
class SpeakQLEnd2endModel(Model):
    def __init__(self,
                 src_text_embedder: TextFieldEmbedder,
                 tabert_model_path: str,
                 finetune_tabert: bool,
                 audio_seq2vec_encoder: Seq2VecEncoder,
                 encoder: SpeakQLEncoder,
                 sql_decoder: SQLDecoder,  # NL2CodeDecoder from ratsql
                 concat_audio: bool,
                 raw_audio_encoder: SpeakQLAudioEncoder = None,  # wav2vec
                 finetune_raw_audio_encoder: bool = False,
                 align_tag_embedder: TokenEmbedder = None,
                 vocab: Vocabulary = None) -> None:
        super().__init__(vocab)
        self.src_text_embedder = src_text_embedder
        self.align_tag_embedder = align_tag_embedder

        self.tabert_model_path = tabert_model_path
        # self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)
        self.finetune_tabert = finetune_tabert
        self.tabert_embedder = TaBERTEmbedder(self.tabert_model_path, self.finetune_tabert)
        
        self.raw_audio_encoder = raw_audio_encoder
        # self.finetune_raw_audio_encoder = finetune_raw_audio_encoder
        self.audio_seq2vec_encoder = audio_seq2vec_encoder
        self.encoder = encoder
        # self.rewrite_decoder = rewrite_decoder  # Already using BLEU as tensor_based_metric by default
        self.sql_decoder = sql_decoder
        
        self.concat_audio = concat_audio

        # Metrics
        self.mle_loss_metrics = Average()
        self.align_loss_metrics = Average()

        self.save_intermediate = False
        self.intermediates = dict()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.encoder.set_save_intermediate(save_intermediate)
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
        '''
        
        sentence = kwargs.get("sentence", None)
        source_token_ids = kwargs.get("source_token_ids", None)
        source_to_target = kwargs.get("source_to_target", None)
        text_mask = kwargs.get("text_mask", None)
        schema_mask = kwargs.get("schema_mask", None)
        schema_column_ids = kwargs.get("schema_column_ids", None)
        audio_feats = kwargs.get("audio_feats", None)
        audio_mask = kwargs.get("audio_mask", None)
        metadata = kwargs.get("metadata", None)
        align_tags = kwargs.get("align_tags", None)
        
        
        # audio_feats: (batch, seq_len, audio_len, audio_dim)
        # audio_mask: (batch, seq_len, audio_len)
        # text_mask: (batch, seq_len)
        
        # print([_meta['source_tokens'] for _meta in metadata])

        # Not sure if this is the right way
        cuda_device_id = get_device_of(audio_feats)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        # _audio_len and _audio_dim are temp, may be changed by raw_audio_encoder (wav2vec)
        batch_size, seq_len, _audio_len, _audio_dim = audio_feats.size()
        assert batch_size == len(metadata)
        
        text_lens = [_meta['text_len'] for _meta in metadata]
        schema_lens = [_meta['schema_len'] for _meta in metadata]
        # rewrite_seq_lens = [_meta['rewrite_seq_s2s_len'] for _meta in metadata]
        concat_lens = [text_lens[i] + 1 + schema_lens[i] for i in range(batch_size)]
        concat_mask = get_mask_from_sequence_lengths(torch.LongTensor(concat_lens), max_length=seq_len).to(device=cuda_device)
        text_mask = get_mask_from_sequence_lengths(torch.LongTensor(text_lens), max_length=seq_len).to(device=cuda_device)
        assert seq_len == max(concat_lens), f'{seq_len} | {concat_lens}'
        
        ## Get sentence (text + schema) mask
        mask = get_text_field_mask(sentence)
        # mask: (batch, seq_len)
        assert tensors_equal(concat_mask, mask), '{}\n{}'.format(concat_mask, mask)
        assert text_mask.size() == concat_mask.size() == mask.size(), f'{text_mask}\n{concat_mask}\n{mask}'
        
        ## TaBERT encoding
        tabert_dim = self.tabert_embedder.tabert_model.output_size

        schema_len_padded = schema_column_ids.size(1)
        assert schema_len_padded == max(schema_lens), f'{schema_len_padded} | {schema_lens}'

        # tabert_sentence_embedding: (batch, seq_len, tabert_dim)
        tabert_sentence_embedding = self.tabert_embedder(schema_column_ids, metadata)

        assert tuple(tabert_sentence_embedding.size()) == (batch_size, seq_len, tabert_dim), \
            f'{tuple(tabert_sentence_embedding.size())} | {(batch_size, seq_len, tabert_dim)}'

        ## (Optional) other embeddings than TaBERT
        if self.src_text_embedder is not None:
            # word_embeddings: (batch, seq_len, emb_dim)
            word_embeddings = self.src_text_embedder(sentence)
            word_embeddings = torch.cat([tabert_sentence_embedding, word_embeddings], dim=-1)
        else:
            word_embeddings = tabert_sentence_embedding
        

        ## Audio encoding

        # Raw audio encoding (if using)
        if self.raw_audio_encoder is not None:
            assert _audio_dim == 1, "Raw audio encoder is supposed to work on raw audio features, but got MFCC"

            _raw_audio_feats_in = audio_feats.view(batch_size * seq_len, _audio_len, _audio_dim)
            _raw_audio_mask_in = audio_mask.view(batch_size * seq_len, _audio_len)
            # with torch.set_grad_enabled(self.finetune_raw_audio_encoder):
            _raw_audio_feats_out, _raw_audio_mask_out = self.raw_audio_encoder(_raw_audio_feats_in, _raw_audio_mask_in)

            _, audio_len, audio_dim = _raw_audio_feats_out.size()   # dim0 is batch_size * seq_len

            # print('MASK SUM (should be proportional):')
            # print(_raw_audio_mask_out.view(batch_size, seq_len, audio_len).sum(2).sum(1))
            # print(audio_mask.sum(2).sum(1))
            # print()
            
            audio_feats = _raw_audio_feats_out.view(batch_size, seq_len, audio_len, audio_dim)
            audio_mask = _raw_audio_mask_out.view(batch_size, seq_len, audio_len)

        else:
            audio_len = _audio_len
            audio_dim = _audio_dim

        # audio_feats: (batch_size, seq_len, audio_len, audio_dim)

        # Audio seq2vec encoding
        audio_feats_enc_in = audio_feats.view(batch_size * seq_len, audio_len, audio_dim)
        audio_mask_enc_in = audio_mask.view(batch_size * seq_len, audio_len)
        audio_feats_enc_out = self.audio_seq2vec_encoder(audio_feats_enc_in, audio_mask_enc_in)
        # audio_feats_enc_out: (batch_size * seq_len, audio_enc_out_dim)
        audio_feats_encoded = audio_feats_enc_out.view(batch_size, seq_len, -1)
        
        # Audio concatenation
        if self.concat_audio:
            token_feats = torch.cat([word_embeddings, audio_feats_encoded], dim=-1)
        else:
            token_feats = word_embeddings

        # Align tags concatenation
        if align_tags is not None:
            # Using align tags
            align_tag_embeddings = self.align_tag_embedder(align_tags)
            token_feats = torch.cat([token_feats, align_tag_embeddings], dim=-1)

        encoder_output = self.encoder(token_feats, audio_feats_encoded, mask)
        
        encoder_seq_repr = encoder_output['seq_repr']
        encoder_att_map = encoder_output['att_map']

        # print(f'--- CP 3: encoder_seq_repr.size() = {encoder_seq_repr.size()}')

        enc_states_nl2code = []
        # losses_batch = []
        mle_losses_batch = []
        align_losses_batch = []
        
        for i in range(batch_size):
            sample_seq_repr = encoder_seq_repr[i]
            sample_att_map = encoder_att_map[i]
            sample_q_tokens = metadata[i]['source_tokens'][:text_lens[i]]
            
            q_enc_item = sample_seq_repr[:text_lens[i]].unsqueeze(0)
            
            c_pointer_spans = metadata[i]['pointer_spans']['column']
            t_pointer_spans = metadata[i]['pointer_spans']['table']
            scm_offset = text_lens[i] + 1
            
            _Nq = text_lens[i]
            _Nc = len(c_pointer_spans) + 1  # include '*'
            _Nt = len(t_pointer_spans)
            _orig_len, _mem_dim = sample_seq_repr.size()
            # _orig_len is the batch-padded full seq length

            # print(f'_Nq = {_Nq}')
            # print(f'_Nc = {_Nc}')
            # print(f'_Nt = {_Nt}')
            # print(f'_orig_len = {_orig_len}')
            # print(f'_mem_dim = {_mem_dim}')
            
            c_enc_list = [torch.zeros(1, _mem_dim, device=cuda_device)]  # zeros for '*'
            for col_idx, col_range in enumerate(c_pointer_spans):
                st, ed = col_range
                st += scm_offset
                ed += scm_offset
                if st < _orig_len:
                    c_enc = sample_seq_repr[st : ed].mean(dim=0).unsqueeze(0)
                    assert not torch.isnan(c_enc).any()
                else:
                    c_enc = torch.zeros(1, _mem_dim, device=cuda_device)
                c_enc_list.append(c_enc)
            # (1, _Nc, _mem_dim)
            c_enc_item = torch.cat(c_enc_list, dim=0).unsqueeze(0)
            
            t_enc_list = []
            for tbl_idx, tbl_range in enumerate(t_pointer_spans):
                st, ed = tbl_range
                st += scm_offset
                ed += scm_offset
                if st < _orig_len:
                    t_enc = sample_seq_repr[st : ed].mean(dim=0).unsqueeze(0)
                    assert not torch.isnan(t_enc).any()
                else:
                    t_enc = torch.zeros(1, _mem_dim, device=cuda_device)
                t_enc_list.append(t_enc)
            # (1, _Nt, _mem_dim)
            t_enc_item = torch.cat(t_enc_list, dim=0).unsqueeze(0)
            
            nl2code_memory = torch.cat([q_enc_item, c_enc_item, t_enc_item], dim=1)
            assert nl2code_memory.size() == (1, _Nq + _Nc + _Nt, _mem_dim), f'nl2code_memory.size() = {nl2code_memory.size()}'
            
            pointer_memories = {
                'column': c_enc_item,
                'table': torch.cat([c_enc_item, t_enc_item], dim=1)
            }
            
            # pointer_maps = {
            #     'column': {j : list(range(cps[0] + scm_offset, cps[1] + scm_offset)) for j, cps in enumerate(c_pointer_spans)},
            #     'table': {j : list(range(tps[0] + scm_offset, tps[1] + scm_offset)) for j, tps in enumerate(t_pointer_spans)},
            # }
            
            # BiLSTM-summ will merge the encoding of multiple tokens of a col/table into one, also merging the
            # boundaries, therefore col/table spans all go to singletons 
            pointer_maps = {
                'column': {j : [j] for j in range(len(c_pointer_spans) + 1)},  # +1 for '*'
                'table': {j : [j] for j in range(len(t_pointer_spans))},
            }
            
            ## align_mat: [q, c, t] to c/t
            
            # source side filtering & pooling 
            _att_list = []
            _att_list.append(sample_att_map[:text_lens[i]])
            _att_list.append(torch.zeros(1, _orig_len, device=cuda_device))  # zeros for '*'
            
            for col_idx, col_range in enumerate(c_pointer_spans):
                st, ed = col_range
                st += scm_offset
                ed += scm_offset
                _c_att = sample_att_map[st : ed].sum(dim=0).unsqueeze(0)
                _att_list.append(_c_att)
                
            for tbl_idx, tbl_range in enumerate(t_pointer_spans):
                st, ed = tbl_range
                st += scm_offset
                ed += scm_offset
                _t_att = sample_att_map[st : ed].sum(dim=0).unsqueeze(0)
                _att_list.append(_t_att)
            
            _att_map = torch.cat(_att_list, dim=0)
            assert _att_map.size() == (_Nq + _Nc + _Nt, _orig_len), f'_att_map.size() = {_att_map.size()}'

            # target side filtering & pooling, for m2c and m2t separately 
            m2c_align_list = [torch.zeros(_Nq + _Nc + _Nt, 1, device=cuda_device)]  # zeros for '*'
            for col_idx, col_range in enumerate(c_pointer_spans):
                st, ed = col_range
                st += scm_offset
                ed += scm_offset
                m2c = _att_map[:, st : ed].sum(dim=1).unsqueeze(1)
                m2c_align_list.append(m2c)
            m2c_align_mat = torch.cat(m2c_align_list, dim=1)
            
            m2t_align_list = []
            for tbl_idx, tbl_range in enumerate(t_pointer_spans):
                st, ed = tbl_range
                st += scm_offset
                ed += scm_offset
                m2t = _att_map[:, st : ed].sum(dim=1).unsqueeze(1)
                m2t_align_list.append(m2t)
            m2t_align_mat = torch.cat(m2t_align_list, dim=1)
            
            assert m2c_align_mat.size() == (_Nq + _Nc + _Nt, _Nc), f'm2c_align_mat.size() = {m2c_align_mat.size()}'
            assert m2t_align_mat.size() == (_Nq + _Nc + _Nt, _Nt), f'm2t_align_mat.size() = {m2t_align_mat.size()}'

            enc_state_nl2code = SpiderEncoderState(
                state=None,
                memory=nl2code_memory,
                question_memory=q_enc_item,
                schema_memory=torch.cat([c_enc_item, t_enc_item], dim=1),
                words=sample_q_tokens,
                pointer_memories=pointer_memories,
                pointer_maps=pointer_maps,
                m2c_align_mat=m2c_align_mat,
                m2t_align_mat=m2t_align_mat
            )
            
            enc_states_nl2code.append(enc_state_nl2code)
            
            # TODO: rat-sql enc_input, dec_input: see if easy to get.
            #   If not, see if possible to modify the decoder (try not to modify torch params) to work without them 

            spider_item, enc_preproc_item, dec_preproc_item = metadata[i]['ratsql_items']
        
            if spider_item.code is not None:
                # (Assumed) when gold code is provided, loss is computable; necessary for val
                # if self.training:
                # loss = self.sql_decoder.compute_loss(enc_preproc_item, dec_preproc_item, enc_state_nl2code, debug=False)
                _loss_ok = True
                
                try:
                    mle_loss = self.sql_decoder.compute_mle_loss(enc_preproc_item, dec_preproc_item, enc_state_nl2code, debug=False)
                    # mle_loss.size() = (1,)
                    assert not torch.isnan(mle_loss)
                except KeyError as e:
                    _err_rule = e.args[0]
                    print("\nIgnored sample with unseen rule:", _err_rule)
                    _loss_ok = False
                except Exception as e:
                    print("[metadata]:")
                    print(metadata[i])
                    print()
                    print("[enc_state_nl2code]:")
                    print(enc_state_nl2code)
                    print()
                    print("** mle_loss error")
                    raise e

                try:
                    align_loss = self.sql_decoder.compute_align_loss(enc_state_nl2code, dec_preproc_item)
                    # align_loss.size() = (,)
                    assert not torch.isnan(align_loss)
                except Exception as e:
                    print("[metadata]:")
                    print(metadata[i])
                    print()
                    print("[enc_state_nl2code]:")
                    print(enc_state_nl2code)
                    print()
                    print("** align_loss error")
                    raise e

                if _loss_ok:
                    mle_losses_batch.append(mle_loss.view(1))
                    align_losses_batch.append(align_loss.view(1))
                else:
                    mle_losses_batch.append(torch.tensor([0.0], device=cuda_device))
                    align_losses_batch.append(torch.tensor([0.0], device=cuda_device))
        
        # if self.training:
        #     assert len(mle_losses_batch) == len(align_losses_batch) == len(enc_states_nl2code) == batch_size, \
        #         f"{len(mle_losses_batch)}, {len(align_losses_batch)}, {len(enc_states_nl2code)}, {batch_size}"
        
        output = {
            'enc_states_nl2code': enc_states_nl2code
        }
        
        # if self.training:
        #     assert len(mle_losses_batch) > 0
        if len(mle_losses_batch) > 0:
            total_mle_loss = torch.mean(torch.stack(mle_losses_batch, dim=0), dim=0)
            total_align_loss = torch.mean(torch.stack(align_losses_batch, dim=0), dim=0)
            total_loss = total_mle_loss + total_align_loss
            output['loss'] = total_loss
            # output['L_mle'] = total_mle_loss
            # output['L_align'] = total_align_loss

            self.mle_loss_metrics(total_mle_loss.item())
            self.align_loss_metrics(total_align_loss.item())

            # print()
            # print('> mle_loss:', total_mle_loss.item())
            # print('> align_loss:', total_align_loss.item())
            # print('> total_loss:', total_loss.item())
        
        return output
        

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = dict()

        metrics_dict["L_mle"] = self.mle_loss_metrics.get_metric(reset)
        metrics_dict["L_align"] = self.align_loss_metrics.get_metric(reset)

        return metrics_dict


    def begin_inference(self, speakql_input):
        metadata = speakql_input["metadata"]
        assert len(metadata) == 1
        orig_item, enc_preproc_item, dec_preproc_item = metadata[0]['ratsql_items']
        
        forward_output = self.forward(**speakql_input)
        enc_state = forward_output['enc_states_nl2code'][0]
        
        return self.sql_decoder.begin_inference(enc_state, orig_item)


