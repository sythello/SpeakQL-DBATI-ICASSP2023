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
    get_device_of, masked_softmax, masked_log_softmax, weighted_sum, \
    get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask, tensors_equal

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
from modules.tabert_embedder import TaBERTEmbedder
from models.rewriter_base import SpiderASRFixer_Base


@Model.register('spider_ASR_rewriter_tagger_comb_new')
class SpiderASRRewriter_Tagger_Combined_new(SpiderASRFixer_Base):
    def __init__(self,
                 src_text_embedder: TextFieldEmbedder,
                 # tgt_text_embedder: TextFieldEmbedder, # Only needed in decoder; no need if decoder is given 
                 use_db_input: bool = True,     ## Ablation: -DB (implemented in reader, currently not needed for model)
                 use_audio: bool = True,        ## Ablation: -audio
                 use_tabert: bool = False,
                 tabert_model_path: str = None,
                 finetune_tabert: bool = False,
                 audio_seq2vec_encoder: Seq2VecEncoder = None,
                 encoder: SpeakQLEncoder = None,
                 using_gated_fusion: bool = False,      ## TODO: combine the code of using different encoders
                 using_ref_att_loss: bool = False,
                 ref_att_loss_coef: float = 1000.0,
                 ref_att_loss_coef_schedule: Dict = None,
                 concat_audio: bool = True,
                 using_modal_proj: bool = False,
                 token_proj_out_dim: int = 256,
                 audio_proj_out_dim: int = 128,
                 using_layer_norm: bool = False,
                 using_phoneme_input: bool = False,
                 ph2tok_audio_seq2vec_encoder: Seq2VecEncoder = None,
                 using_phoneme_labels: bool = False,
                 phoneme_loss_coef: float = 1.0,
                 using_phoneme_multilabels: bool = False,
                 phoneme_multilabel_loss_coef: float = 1.0,
                 aux_probe_configs: Dict = None,    # Keys: "utter_mention_schema[_coef]", "schema_dir_mentioned[_coef]", "schema_indir_mentioned[_coef]"
                 ## optional feats
                 align_tag_embedder: TokenEmbedder = None,
                 phoneme_tag_embedder: TextFieldEmbedder = None,    ## TextField to workaround the padding=-1 problem with LabelField
                 phoneme_tag_seq2vec_encoder: Seq2VecEncoder = None,
                 ## classifier specific 
                 ff_dimension: int = 64,
                 vocab: Vocabulary = None) -> None:

        super().__init__(
            src_text_embedder=src_text_embedder,
            use_db_input=use_db_input,
            use_audio=use_audio,
            use_tabert=use_tabert,
            tabert_model_path=tabert_model_path,
            finetune_tabert=finetune_tabert,
            audio_seq2vec_encoder=audio_seq2vec_encoder,
            encoder=encoder,
            using_gated_fusion=using_gated_fusion,
            using_ref_att_loss=using_ref_att_loss,
            ref_att_loss_coef=ref_att_loss_coef,
            ref_att_loss_coef_schedule=ref_att_loss_coef_schedule,
            concat_audio=concat_audio,
            using_modal_proj=using_modal_proj,
            token_proj_out_dim=token_proj_out_dim,
            audio_proj_out_dim=audio_proj_out_dim,
            using_layer_norm=using_layer_norm,
            using_phoneme_input=using_phoneme_input,
            ph2tok_audio_seq2vec_encoder=ph2tok_audio_seq2vec_encoder,
            using_phoneme_labels=using_phoneme_labels,
            phoneme_loss_coef=phoneme_loss_coef,
            using_phoneme_multilabels=using_phoneme_multilabels,
            phoneme_multilabel_loss_coef=phoneme_multilabel_loss_coef,
            aux_probe_configs=aux_probe_configs,
            align_tag_embedder=align_tag_embedder,
            phoneme_tag_embedder=phoneme_tag_embedder,
            phoneme_tag_seq2vec_encoder=phoneme_tag_seq2vec_encoder,
            rewriter_tag_embedder=None,
            vocab=vocab,
        )

        # Tagger specific
        self.ff_dimension = ff_dimension

        self.num_rewriter_tags = vocab.get_vocab_size('rewriter_tags')
        self.tag_projection_layer_1 = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=ff_dimension)
        self.tag_projection_layer_2 = torch.nn.Linear(
            in_features=ff_dimension,
            out_features=self.num_rewriter_tags)
        self.constraints = allowed_transitions(constraint_type='BIOUL',
                                               labels=vocab.get_index_to_token_vocabulary('rewriter_tags'))
        self.crf = ConditionalRandomField(
            self.num_rewriter_tags,
            self.constraints,
            include_start_end_transitions=True
        )

        # Metrics
        self.tag_accuracy = BooleanAccuracy()
        self.tag_NLL = Average()
        self.att_KL = Average()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.encoder.set_save_intermediate(save_intermediate)
        self.save_intermediate = save_intermediate

    def get_intermediates(self):
        return {
            'encoder': self.encoder.intermediates,
            'main': self.intermediates
        }

    @overrides
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                text_mask: torch.Tensor,
                schema_mask: torch.Tensor,
                schema_column_ids: torch.Tensor,    # For each schema token, which column it belongs to (starting at 1), 0 if not column
                audio_feats: torch.Tensor,
                audio_mask: torch.Tensor,
                metadata: List[Dict],
                phoneme_audio_feats: torch.Tensor = None,
                phoneme_labels: torch.Tensor = None,
                phoneme_multilabels: torch.Tensor = None,
                phoneme_audio_mask: torch.Tensor = None,
                phoneme_label_mask: torch.Tensor = None,
                utter_mention_schema_labels: torch.Tensor = None,
                schema_dir_mentioned_labels: torch.Tensor = None,
                schema_indir_mentioned_labels: torch.Tensor = None,
                ref_att_map: torch.Tensor = None,
                align_tags: torch.Tensor = None,
                rewriter_tags: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # audio_feats: (batch, seq_len, audio_len, audio_dim)
        # audio_mask: (batch, seq_len, audio_len)
        # text_mask: (batch, seq_len)

        cuda_device_id = get_device_of(audio_feats)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        batch_size, seq_len, audio_len, audio_dim = audio_feats.size()
        assert batch_size == len(metadata)

        basic_encoding_output = self._basic_encoding(
            sentence=sentence,
            text_mask=text_mask,
            schema_mask=schema_mask,
            schema_column_ids=schema_column_ids,
            audio_feats=audio_feats,
            audio_mask=audio_mask,
            phoneme_audio_feats=phoneme_audio_feats,
            phoneme_labels=phoneme_labels,
            phoneme_multilabels=phoneme_multilabels,
            phoneme_audio_mask=phoneme_audio_mask,
            phoneme_label_mask=phoneme_label_mask,
            metadata=metadata,
            align_tags=align_tags,
            rewriter_tags=None,     ## This is ground truth for tagger, so never provide as input
        )

        text_lens = basic_encoding_output['text_lens']
        schema_lens = basic_encoding_output['schema_lens']
        concat_lens = basic_encoding_output['concat_lens']
        mask = basic_encoding_output['mask']
        text_mask = basic_encoding_output['text_mask']
        schema_mask = basic_encoding_output['schema_mask']
        word_embeddings = basic_encoding_output['word_embeddings']
        audio_feats_encoded = basic_encoding_output['audio_feats_encoded']
        ph_audio_feats_encoded = basic_encoding_output['ph_audio_feats_encoded']
        align_tag_embeddings = basic_encoding_output['align_tag_embeddings']
        phoneme_tag_embeddings = basic_encoding_output['phoneme_tag_embeddings']

        tag_embeddings_list = []
        if align_tag_embeddings is not None:
            tag_embeddings_list.append(align_tag_embeddings)
        if phoneme_tag_embeddings is not None:
            tag_embeddings_list.append(phoneme_tag_embeddings)
        # tag_embeddings_list.append(rewriter_tag_embeddings)
        if len(tag_embeddings_list) > 0:
            tag_embeddings = torch.cat(tag_embeddings_list, dim=-1)
        else:
            tag_embeddings = None

        # Encoding
        if self.using_gated_fusion:
            # encoder version should be 'gated_fusion' (deprecated) or 'v2'
            encoder_output = self.encoder(word_embeddings, audio_feats_encoded, align_tag_embeddings, mask)
        else:
            # encoder version should be 'v1', i.e. non-gated fusion

            token_feats = word_embeddings
            if self.concat_audio and self.use_audio:
                token_feats = torch.cat([token_feats, audio_feats_encoded], dim=-1)
            if tag_embeddings is not None:
                token_feats = torch.cat([token_feats, tag_embeddings], dim=-1)

            # # encoder v1 has updated to take these kwargs
            # encoder_output = self.encoder(token_feats=token_feats, audio_feats=audio_feats_encoded, mask=mask)

            attention_feats_dict = {
                'audio': audio_feats_encoded,
                'full': token_feats,
                'phoneme': phoneme_tag_embeddings,
            }
            encoder_output = self.encoder(
                token_feats=token_feats,
                # audio_feats=audio_feats_encoded,
                attention_feats_dict=attention_feats_dict,
                mask=mask)

        encoder_seq_repr = encoder_output['seq_repr']
        self._maybe_save(encoder_seq_repr.detach().cpu().numpy(), 'encoder_seq_representation')


        output = dict()
        
        tag_logits = F.leaky_relu(self.tag_projection_layer_1(encoder_seq_repr), negative_slope=0.01)
        tag_logits = self.tag_projection_layer_2(tag_logits)
        best_path = self.crf.viterbi_tags(tag_logits, text_mask)
        # best_path: [([preds], nll_score)]
        # preds are already padded to full length of concat_sentence_field 
        tag_pred = [p[0] for p in best_path]    # iterating on batch dim; List[List[int]]; len(tag_pred[i]) == text_lens[i]
        
        output['tag_logits'] = tag_logits
        output['mask'] = mask
        output['text_mask'] = text_mask
        output['tag_pred'] = tag_pred
        
        if rewriter_tags is not None:
            ## Train
            # tag_losses = []
            # for i in range(batch_size):
            #     log_likelihood = self.crf(tag_logits[i:i+1], tags[i:i+1], text_mask[i:i+1])  # [i:i+1]: keep the batch dimension 
            #     _loss = -log_likelihood
            #     self.tag_NLL(_loss.item() / text_lens[i])
            #     # print(torch.LongTensor(tag_pred[i]).size())
            #     # print(tags[i, :text_lens[i]].size())
            #     # print(text_lens[i])
            #     self.tag_accuracy(torch.tensor(tag_pred[i], dtype=torch.long, device=cuda_device), tags[i, :text_lens[i]])

            #     tag_losses.append(_loss)

            # output['tag_loss'] = torch.stack(tag_losses).mean()

            log_likelihood = self.crf(tag_logits, rewriter_tags, text_mask)
            _loss = -log_likelihood
            output['tag_loss'] = _loss

            self.tag_NLL(_loss.item())
            for i in range(batch_size):
                self.tag_accuracy(torch.tensor(tag_pred[i], dtype=torch.long, device=cuda_device), rewriter_tags[i, :text_lens[i]])
        
            output['loss'] = output['tag_loss']

            if self.using_ref_att_loss and (ref_att_map is not None):
                assert encoder_output['att_map_logits'] is not None
                # assert ref_att_map is not None        ## Not need to be true during prediction
                log_att_map = masked_log_softmax(encoder_output['att_map_logits'], mask)    ## TODO: is this mask really working properly? Since it's 1 dim less than att_map
                att_kl_loss = F.kl_div(log_att_map, ref_att_map, reduction='batchmean')

                self.att_KL(att_kl_loss.item())
                output['att_kl_loss'] = att_kl_loss
                output['loss'] += self.ref_att_loss_coef * att_kl_loss

            ## Phoneme losses
            if self.using_phoneme_labels:
                assert ph_audio_feats_encoded is not None
                _, ph_len, audio_enc_dim = ph_audio_feats_encoded.size() # (batch_size * seq_len, ph_len, audio_enc_dim)
                # (batch * seq_len * ph_len, ph_size)
                ph_pred_logits = self.phoneme_pred_head(ph_audio_feats_encoded.view(batch_size * seq_len * ph_len, audio_enc_dim))
                # (batch * seq_len * ph_len)
                xent_tensor = F.cross_entropy(ph_pred_logits, phoneme_labels.view(-1), reduction='none', ignore_index=-1)
                # (batch, seq_len, ph_len)
                xent_tensor = xent_tensor.view(batch_size, seq_len, ph_len)

                phoneme_loss = torch.mean(xent_tensor * phoneme_label_mask)

                self.ph_NLL(phoneme_loss.item())
                output['ph_loss'] = phoneme_loss
                output['loss'] += self.phoneme_loss_coef * phoneme_loss
                output['ph_preds'] = ph_pred_logits.argmax(dim=-1).view(batch_size, seq_len, ph_len).detach().cpu().numpy()

            if self.using_phoneme_multilabels:
                # print('='*30)
                # print(phoneme_multilabels.size())
                # print(phoneme_label_mask.size())
                # print(phoneme_multilabels)
                # print(phoneme_label_mask)

                _, _, audio_enc_dim = audio_feats_encoded.size() # (batch_size, seq_len, audio_enc_dim)
                # (batch, seq_len, ph_size)
                ph_pred_logits = self.phoneme_pred_head(audio_feats_encoded)
                # (batch, seq_len, ph_size)
                xent_tensor = F.binary_cross_entropy_with_logits(ph_pred_logits, phoneme_multilabels.float(), reduction='none')
                # (batch, seq_len)
                xent_tensor = xent_tensor.mean(dim=-1)
                # (batch, seq_len)
                phoneme_multilabel_mask = phoneme_label_mask[:, :, 0]   # pos 0 stands for token phonemes on/off
                phoneme_multilabel_loss = torch.mean(xent_tensor * phoneme_multilabel_mask)

                # print(ph_pred_logits.size())
                # print(xent_tensor.size())
                # print(phoneme_multilabel_mask.size())
                # print(ph_pred_logits)
                # print(xent_tensor)
                # print(phoneme_multilabel_mask)
                # print('='*30)

                self.ph_multi_NLL(phoneme_multilabel_loss.item())
                output['ph_multi_loss'] = phoneme_multilabel_loss
                output['loss'] += self.phoneme_multilabel_loss_coef * phoneme_multilabel_loss
                output['ph_multi_preds'] = ph_pred_logits.ge(0).detach().cpu().numpy()

            ## Probing losses
            if self.utter_mention_schema_head is not None:
                # (batch, seq_len, 1)
                utter_mention_schema_logits = self.utter_mention_schema_head(encoder_seq_repr).squeeze(-1)
                xent_tensor = F.binary_cross_entropy_with_logits(
                    utter_mention_schema_logits,
                    utter_mention_schema_labels,
                    reduction='none')
                utter_mention_schema_loss = torch.mean(xent_tensor * text_mask)

                self.u_ms_NLL(utter_mention_schema_loss.item())
                output['u_ms_loss'] = utter_mention_schema_loss
                output['loss'] += self.utter_mention_schema_coef * utter_mention_schema_loss
                output['u_ms_preds'] = utter_mention_schema_logits.ge(0).detach().cpu().numpy()

            if self.schema_dir_mentioned_head is not None:
                # (batch, seq_len, 1)
                schema_dir_mentioned_logits = self.schema_dir_mentioned_head(encoder_seq_repr).squeeze(-1)
                xent_tensor = F.binary_cross_entropy_with_logits(
                    schema_dir_mentioned_logits,
                    schema_dir_mentioned_labels,
                    reduction='none')
                schema_dir_mentioned_loss = torch.mean(xent_tensor * schema_mask)

                self.s_dm_NLL(schema_dir_mentioned_loss.item())
                output['s_dm_loss'] = schema_dir_mentioned_loss
                output['loss'] += self.schema_dir_mentioned_coef * schema_dir_mentioned_loss
                output['s_dm_preds'] = schema_dir_mentioned_logits.ge(0).detach().cpu().numpy()

            if self.schema_indir_mentioned_head is not None:
                # (batch, seq_len, 1)
                schema_indir_mentioned_logits = self.schema_indir_mentioned_head(encoder_seq_repr).squeeze(-1)
                xent_tensor = F.binary_cross_entropy_with_logits(
                    schema_indir_mentioned_logits,
                    schema_indir_mentioned_labels,
                    reduction='none')
                schema_indir_mentioned_loss = torch.mean(xent_tensor * schema_mask)

                self.s_im_NLL(schema_indir_mentioned_loss.item())
                output['s_im_loss'] = schema_indir_mentioned_loss
                output['loss'] += self.schema_indir_mentioned_coef * schema_indir_mentioned_loss
                output['s_im_preds'] = schema_indir_mentioned_logits.ge(0).detach().cpu().numpy()

            
        return output
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = {
            "tag_NLL": self.tag_NLL.get_metric(reset),
            "tag_accuracy": self.tag_accuracy.get_metric(reset),
            # "rewrite_seq_NLL": self.rewrite_seq_NLL.get_metric(reset)
        }

        if self.using_ref_att_loss:
            metrics_dict["att_KL"] = self.att_KL.get_metric(reset)

        if self.using_phoneme_labels:
            metrics_dict["ph_NLL"] = self.ph_NLL.get_metric(reset)
        
        if self.using_phoneme_multilabels:
            metrics_dict["ph_m_NLL"] = self.ph_multi_NLL.get_metric(reset)

        if self.utter_mention_schema_head is not None:
            metrics_dict["u_ms"] = self.u_ms_NLL.get_metric(reset)

        if self.schema_dir_mentioned_head is not None:
            metrics_dict["s_dm"] = self.s_dm_NLL.get_metric(reset)

        if self.schema_indir_mentioned_head is not None:
            metrics_dict["s_im"] = self.s_im_NLL.get_metric(reset)

        return metrics_dict
    
    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # This is batched! 
        
        if 'tag_pred' in output_dict:
            _tag_pred_readable = []
            # _tag_NLL = []
            for _tag_pred in output_dict['tag_pred']:
                _tag_pred_readable.append(self.make_tags_prediction_readable(_tag_pred))
                # _tag_NLL.append(_tag_pred_score)
                
            output_dict['tag_pred_readable'] = _tag_pred_readable
            # output_dict['tag_NLL'] = _tag_NLL

        return output_dict
        
    def make_tags_prediction_readable(self, prediction) -> List[str]:
        return [self.vocab.get_token_from_index(int(idx), namespace='rewriter_tags') for idx in prediction]


