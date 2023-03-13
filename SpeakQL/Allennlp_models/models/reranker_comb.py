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

from table_bert import TableBertModel

from modules.encoder import SpeakQLEncoder, SpeakQLEncoderV1
from modules.tabert_embedder import TaBERTEmbedder


@Model.register('spider_ASR_reranker_v2_comb')
class SpiderASRRerankerV2_Combined(Model):
    '''
    V2: Audio features are passed as sequences, not avg/max pooled vectors.
    '''
    def __init__(self,
                 src_text_embedder: TextFieldEmbedder = None,
                 use_tabert: bool = False,
                 tabert_model_path: str = None,
                 finetune_tabert: bool = False,
                 audio_seq2vec_encoder: Seq2VecEncoder = None,
                 encoder: SpeakQLEncoder = None,
                 ff_dimension: int = 64,
                 concat_audio: bool = True,
                 align_tag_embedder: TokenEmbedder = None,
                 vocab: Vocabulary = None) -> None:
        super().__init__(vocab)
        self.src_text_embedder = src_text_embedder
        self.align_tag_embedder = align_tag_embedder
        
        self.use_tabert = use_tabert
        self.tabert_model_path = tabert_model_path
        self.finetune_tabert = finetune_tabert
        if use_tabert:
            # self.tabert_model = TableBertModel.from_pretrained(tabert_model_path)
            self.tabert_embedder = TaBERTEmbedder(tabert_model_path, self.finetune_tabert)
        
        self.audio_seq2vec_encoder = audio_seq2vec_encoder
        self.encoder = encoder
        self.ff1 = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                   out_features=ff_dimension)
        self.ff2 = torch.nn.Linear(in_features=ff_dimension,
                                   out_features=1)

        self.concat_audio = concat_audio

        self.accuracy = MeanAbsoluteError()     # Not directly reported

        self.save_intermediate = False
        self.intermediates = dict()
    
    def _maybe_save(self, val, name):
        if self.save_intermediate:
            self.intermediates[name] = val

    def set_save_intermediate(self, save_intermediate: bool):
        self.encoder.set_save_intermediate(save_intermediate)
        ## TODO: other modules
        self.save_intermediate = save_intermediate
    
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                text_mask: torch.Tensor,
                schema_mask: torch.Tensor,
                schema_column_ids: torch.Tensor,    # For each schema token, which column it belongs to (starting at 1), 0 if not column
                audio_feats: torch.Tensor,
                audio_mask: torch.Tensor,
                metadata,
                score: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # audio_feats: (batch, seq_len, audio_len, audio_dim)
        # audio_mask: (batch, seq_len, audio_len)
        # text_mask: (batch, seq_len)

        # Not sure if this is the right way
        cuda_device_id = get_device_of(audio_feats)
        cuda_device = 'cpu' if cuda_device_id < 0 else cuda_device_id

        batch_size, seq_len, audio_len, audio_dim = audio_feats.size()
        assert batch_size == len(metadata)
        
        text_lens = [_meta['text_len'] for _meta in metadata]
        schema_lens = [_meta['schema_len'] for _meta in metadata]
        concat_lens = torch.LongTensor([text_lens[i] + 1 + schema_lens[i] for i in range(batch_size)])
        concat_mask = get_mask_from_sequence_lengths(concat_lens, max_length=seq_len).to(device=cuda_device)
        
        # Get text (sentence) mask
        mask = get_text_field_mask(sentence)
        # mask: (batch, seq_len)
        assert tensors_equal(concat_mask, mask), '{}\n{}'.format(concat_mask, mask)

        input_embeddings = []

        ## (Optional) using TaBERT embeddings
        if self.use_tabert:
            ## TaBERT encoding
            tabert_dim = self.tabert_embedder.tabert_model.output_size

            schema_len_padded = schema_column_ids.size(1)
            assert schema_len_padded == max(schema_lens), f'{schema_len_padded} | {schema_lens}'

            # tabert_sentence_embedding: (batch, seq_len, tabert_dim)
            tabert_sentence_embedding = self.tabert_embedder(schema_column_ids, metadata)

            assert tuple(tabert_sentence_embedding.size()) == (batch_size, seq_len, tabert_dim), \
                f'{tuple(tabert_sentence_embedding.size())} | {(batch_size, seq_len, tabert_dim)}'

            input_embeddings.append(tabert_sentence_embedding)

        ## (Optional) other embeddings than TaBERT
        if self.src_text_embedder is not None:
            # src_text_embedding: (batch, seq_len, emb_dim)
            src_text_embedding = self.src_text_embedder(sentence)
            input_embeddings.append(src_text_embedding)
        
        # word_embeddings: (batch, seq_len, emb_dim)
        word_embeddings = torch.cat(input_embeddings, dim=-1)

        
        # Audio seq2vec encoding
        audio_feats_enc_in = audio_feats.view(batch_size * seq_len, audio_len, audio_dim)
        audio_mask_enc_in = audio_mask.view(batch_size * seq_len, audio_len)
        audio_feats_enc_out = self.audio_seq2vec_encoder(audio_feats_enc_in, audio_mask_enc_in)
        # audio_feats_enc_out: (batch_size * seq_len, audio_enc_out_dim)
        audio_feats_encoded = audio_feats_enc_out.view(batch_size, seq_len, -1)
        # audio_feats_encoded: (batch_size, seq_len, audio_enc_out_dim)
        
        # Audio concatenation
        if self.concat_audio:
            token_feats = torch.cat([word_embeddings, audio_feats_encoded], dim=-1)
        else:
            token_feats = word_embeddings
        # token_feats: (batch_size, seq_len, emb_dim(+audio_dim) )    

        # Use custom-encoder 
        encoder_output = self.encoder(token_feats, audio_feats_encoded, mask)
        encoder_vec_out = encoder_output['vec_repr']
        # encoder_out: (batch_size, enc_dim)
        # encoder_seq_repr: (batch_size, seq_len, enc_dim)

        score_preds = torch.sigmoid(self.ff2(F.leaky_relu(self.ff1(encoder_vec_out), negative_slope=0.01))).squeeze(1)
        # score_preds: (batch_size,)
    
        output = {"score_preds": score_preds}
        if score is not None:
            self.accuracy(score_preds, score.squeeze(1))
            output["loss"] = F.mse_loss(score_preds, score.squeeze(1))
            # Why squeeze()??

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy-MAE": self.accuracy.get_metric(reset)}
    


