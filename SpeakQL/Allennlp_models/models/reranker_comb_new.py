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

from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, MeanAbsoluteError, Average, BLEU
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



@Model.register('spider_ASR_reranker_v2_comb_new')
class SpiderASRRerankerV2_Combined_new(SpiderASRFixer_Base):
    '''
    V2: Audio features are passed as sequences, not avg/max pooled vectors.
    new: using base class SpiderASRFixer_Base; using NLL clf error, not siamese
    '''
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
                 ## reranker specific 
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
            vocab=vocab,
        )

        self.ff1 = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                   out_features=ff_dimension)
        self.ff2 = torch.nn.Linear(in_features=ff_dimension,
                                   out_features=1)

        self.concat_audio = concat_audio

        # self.accuracy = MeanAbsoluteError()     # Not directly reported
        self.oracle_label_NLL = Average()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.encoder.set_save_intermediate(save_intermediate)
        ## TODO: other modules
        self.save_intermediate = save_intermediate

    def get_intermediates(self):
        return {
            'encoder': self.encoder.get_intermediates(),
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
                phoneme_labels: Dict[str, torch.Tensor] = None,
                phoneme_multilabels: torch.Tensor = None,
                phoneme_audio_mask: torch.Tensor = None,
                phoneme_label_mask: torch.Tensor = None,
                utter_mention_schema_labels: torch.Tensor = None,
                schema_dir_mentioned_labels: torch.Tensor = None,
                schema_indir_mentioned_labels: torch.Tensor = None,
                ref_att_map: torch.Tensor = None,
                align_tags: torch.Tensor = None,
                oracle_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # audio_feats: (batch, seq_len, audio_len, audio_dim)
        # audio_mask: (batch, seq_len, audio_len)
        # text_mask: (batch, seq_len)

        # Not sure if this is the right way
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
        rewriter_tag_embeddings = basic_encoding_output['rewriter_tag_embeddings']
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
            encoder_output = self.encoder(word_embeddings, audio_feats_encoded, tag_embeddings, mask)
        else:
            # encoder version should be 'v1', i.e. non-gated fusion

            token_feats = word_embeddings
            if self.concat_audio and self.use_audio:
                token_feats = torch.cat([token_feats, audio_feats_encoded], dim=-1)
            if tag_embeddings is not None:
                token_feats = torch.cat([token_feats, tag_embeddings], dim=-1)

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

        # Use custom-encoder 
        encoder_vec_out = encoder_output['vec_repr']
        # encoder_vec_out: (batch_size, enc_dim)
        # encoder_seq_repr: (batch_size, seq_len, enc_dim)

        score_preds = torch.sigmoid(self.ff2(F.leaky_relu(self.ff1(encoder_vec_out), negative_slope=0.01))).squeeze(1)
        # score_preds: (batch_size,)
    
        output = {"score_preds": score_preds}
        if oracle_label is not None:
            xent = F.binary_cross_entropy(score_preds, oracle_label.float())

            self.oracle_label_NLL(xent.item())

            output["loss"] = xent

            # output["loss"] = F.mse_loss(score_preds, oracle_label.squeeze(1))
            # Why squeeze()??

        if self.training:
            ## Add aux losses
            ## A problem: when a sample is used many times (e.g. 1 correct vs. 5 incorrect, the correct one is used 5 times), then
            ## the aux losses for the sample are also computed many times. Might / might not be a problem
            ## For now, not using any aux losses in config

            if self.using_ref_att_loss:
                assert encoder_output['att_map_logits'] is not None
                assert ref_att_map is not None
                log_att_map = masked_log_softmax(encoder_output['att_map_logits'], mask)
                att_kl_loss = F.kl_div(log_att_map, ref_att_map, reduction='batchmean')

                # NEW: scheduling 
                _coef = self.ref_att_loss_coef
                if self.ref_att_loss_coef_schedule is not None:
                    _decay_rate = self.ref_att_loss_coef_schedule["decay_rate"]
                    _decay_value = self.ref_att_loss_coef_schedule["decay_value"]
                    _min_value = self.ref_att_loss_coef_schedule["min_value"]
                    
                    for _ in range(self.epoch): # Should be set by the callback
                        _coef = max(_min_value, _coef * _decay_rate - _decay_value)

                self.att_KL(att_kl_loss.item())
                output['aux_loss:att_kl_loss'] = _coef * att_kl_loss
                # output['att_kl_loss_coef'] = _coef    # float in output will cause predictor error
                output['loss'] += _coef * att_kl_loss

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
                output['aux_loss:ph_loss'] = self.phoneme_loss_coef * phoneme_loss
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
                output['aux_loss:ph_multi_loss'] = self.phoneme_multilabel_loss_coef * phoneme_multilabel_loss
                output['loss'] += self.phoneme_multilabel_loss_coef * phoneme_multilabel_loss
                output['ph_multi_preds'] = ph_pred_logits.ge(0).detach().cpu().numpy()

            if self.utter_mention_schema_head is not None:
                # (batch, seq_len, 1)
                utter_mention_schema_logits = self.utter_mention_schema_head(encoder_seq_repr).squeeze(-1)
                xent_tensor = F.binary_cross_entropy_with_logits(
                    utter_mention_schema_logits,
                    utter_mention_schema_labels,
                    reduction='none')
                utter_mention_schema_loss = torch.mean(xent_tensor * text_mask)

                self.u_ms_NLL(utter_mention_schema_loss.item())
                output['aux_loss:u_ms_loss'] = self.utter_mention_schema_coef * utter_mention_schema_loss
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
                output['aux_loss:s_dm_loss'] = self.schema_dir_mentioned_coef * schema_dir_mentioned_loss
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
                output['aux_loss:s_im_loss'] = self.schema_indir_mentioned_coef * schema_indir_mentioned_loss
                output['loss'] += self.schema_indir_mentioned_coef * schema_indir_mentioned_loss
                output['s_im_preds'] = schema_indir_mentioned_logits.ge(0).detach().cpu().numpy()

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # return {"accuracy-MAE": self.accuracy.get_metric(reset)}
        metrics_dict = {
            "oracle_label_NLL": self.oracle_label_NLL.get_metric(reset)
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

    


