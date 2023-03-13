# (2.33.8) Base 2.33.5 (-align_tags); -full DB


local AUDIO_DIM_POOLING = 136;
local AUDIO_DIM_NO_POOLING = 68;

local PLM_MODEL_NAME = "facebook/bart-base";
local TAG_EMB_DIM = 64;
local SRC_EMB_DIM = 768; # XX-base 
# local SRC_EMB_DIM = 1024; # XX-large
local TGT_EMB_DIM = 300;
local AUDIO_ENC_DIM = 128;
local CHAR_EMB_DIM = 128;

local ENCODER_DIM = 256;
local TAGGING_FF_DIM = 64;
local DECODER_DIM = ENCODER_DIM; # It seems that otherwise it can't work 

local PHONEME_VOCAB_NAMESPACE = "phonemes";
local PHONEME_INDEXER_KEY = "phonemes";

# local SPIDER_DIR = "/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider";
# local SPIDER_DIR = "/vault/dataset/yshao/spider";
local SPIDER_DIR = "/vault/spider";

# local TABERT_MODEL_PATH = "/Users/mac/Desktop/syt/Deep-Learning/Repos/TaBERT/pretrained-models/tabert_base_k1/model.bin";
local TABERT_MODEL_PATH = "/vault/TaBERT/pretrained-models/tabert_base_k1/model.bin";

# local CMU_DICT_PATH = "/Users/mac/Desktop/syt/Deep-Learning/Dataset/CMUdict/cmudict-0.7b.txt";
local CMU_DICT_PATH = "/vault/CMUdict/cmudict-0.7b.txt";

local DB_TOK2PHS_PATH = SPIDER_DIR + "/my/db/db_tok2phs.json";


{
  "dataset_reader": {
    "type": "spider_ASR_rewriter_reader_ILM_comb_new",
    "src_token_indexers": {
      "bert": {
        ## Bad naming... but need to keep the "bert" key
        "type": "pretrained_transformer_mismatched",
        "model_name": PLM_MODEL_NAME,
      },
      "char": {
        "type": "characters",
        "namespace": "token_characters",
        "min_padding_length": 5,
      },
    },
    "tgt_token_indexers": {
      "tgt_tokens": {
        "type": "single_id",
        "namespace": "tgt_tokens"
      }
    },
    "ph_token_indexers": {
      [PHONEME_INDEXER_KEY]: {
        "type": "single_id",
        "namespace": PHONEME_VOCAB_NAMESPACE,
      },
    },
    "include_align_tags": false,
    "tables_json_fname": SPIDER_DIR + "/tables.json",
    "dataset_dir": SPIDER_DIR + "/my",
    "databases_dir": SPIDER_DIR + "/database",
    "use_db_input": false,
    # "use_db_cells": "K=5",
    # "db_cells_in_bracket": true,
    "use_tabert": false,
    "tabert_model_path": TABERT_MODEL_PATH,
    "use_phoneme_inputs": false,
    "use_phoneme_labels": true,   # multi-label
    "pronun_dict_path": CMU_DICT_PATH,
    "db_tok2phs_dict_path": DB_TOK2PHS_PATH,
    "specify_full_path": false,
    "max_sequence_len": 300,
    "include_gold_rewrite_seq": true,
    "use_tagger_prediction": false,
  },
  "train_data_path": "train",
  "validation_data_path": "dev",
  "model": {
    "type": "spider_ASR_rewriter_ILM_comb_new",
    # "src_text_embedder": null,
    "src_text_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_mismatched_patched",
          "model_name": PLM_MODEL_NAME,
          "sub_module": "encoder",
          "train_parameters": false,
          "last_layer_only": false,
          "set_output_hidden_states": true,
          # return idx: For transformers 3.0.2, BART is 1 (or -2); T5 is -1 (default, can omit)
          "hidden_states_return_idx": 1,
        },
        "char": {
          "type": "character_encoding",
          # TokenCharactersEncoder(subclass of TokenEmbedder)
          "embedding": {
            "embedding_dim": CHAR_EMB_DIM,
            "vocab_namespace": "token_characters",
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": CHAR_EMB_DIM,
            "num_filters": 4,
            "ngram_filter_sizes": [2, 3, 4, 5],
            "output_dim": CHAR_EMB_DIM
          },
          "dropout": 0.0,
        },
      },
    },
    ## Major modality ablation
    "use_db_input": false,
    "rewriter_tag_embedder": {
      "type": "embedding",
      "embedding_dim": TAG_EMB_DIM,
      "vocab_namespace": "rewriter_tags",
    },
    "align_tag_embedder": null,
    # "align_tag_embedder": {
    #   "type": "embedding",
    #   "embedding_dim": TAG_EMB_DIM,
    #   "vocab_namespace": "align_tags",
    # },
    "phoneme_tag_embedder": {
      "token_embedders": {
        [PHONEME_INDEXER_KEY]: {
          "type": "embedding",
          "embedding_dim": TAG_EMB_DIM,
          "vocab_namespace": PHONEME_VOCAB_NAMESPACE,
        }
      }
    },
    "using_copy_mechanism": true,
    "use_tabert": false,
    "tabert_model_path": TABERT_MODEL_PATH,
    "finetune_tabert": false,
    "audio_seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": AUDIO_DIM_NO_POOLING,
      "num_filters": 4,
      "ngram_filter_sizes": [2, 3, 4, 5],
      "output_dim": AUDIO_ENC_DIM
    },
    "phoneme_tag_seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": TAG_EMB_DIM,
      "num_filters": TAG_EMB_DIM / 4,
      "ngram_filter_sizes": [2, 2, 3, 4],
      "output_dim": TAG_EMB_DIM
    },
    ## NEW: phonemes
    "using_phoneme_input": false,
    # "ph2tok_audio_seq2vec_encoder": {
    #   "type": "cnn",
    #   "embedding_dim": AUDIO_ENC_DIM,
    #   "num_filters": 4,
    #   "ngram_filter_sizes": [2, 3],
    #   "output_dim": AUDIO_ENC_DIM
    # },
    "ph2tok_audio_seq2vec_encoder": null,
    "using_phoneme_labels": false,
    "phoneme_loss_coef": 0.0,
    "using_phoneme_multilabels": true,
    "phoneme_multilabel_loss_coef": 0.1,
    ## ENDNEW
    "encoder": {
      "type": "v1",
      # "audio_attention_layer": {
      #   "type": "cosine"
      # },
      # "audio_attention_residual": "+",
      "attention_type": "phoneme",
      "attention_layer": {
        "type": "cosine"
      },
      "attention_residual": "+",
      "seq2seq_encoders": [
        {
          "type": "lstm",
          "input_size": (SRC_EMB_DIM + AUDIO_ENC_DIM + TAG_EMB_DIM * 2 + CHAR_EMB_DIM),
          "hidden_size": ENCODER_DIM / 2, # bidirectional
          "num_layers": 1,
          "dropout": 0.0,
          "bidirectional": true
        }
      ],
      "seq2vec_encoder": null
    },
    "using_gated_fusion": false,
    "using_ref_att_loss": false,
    "ref_att_loss_coef": 0.0,
    "using_modal_proj": false,
    "rewrite_decoder": {
      "type": "speakql_copynet_seq_decoder",
      "encoder_output_dim": ENCODER_DIM,
      "is_bidirectional_input": true,
      "attention": {
        "type": "bilinear",
        "vector_dim": DECODER_DIM,
        "matrix_dim": ENCODER_DIM
      },
      "beam_size": 4,
      "max_decoding_steps": 50,
      "target_embedder": {
        "embedding_dim": TGT_EMB_DIM,
        "vocab_namespace": "tgt_tokens",
      },
      "target_namespace": "tgt_tokens"
    },
    # "rewrite_decoder": {
    #   "type": "auto_regressive_seq_decoder",
    #   "decoder_net": {
    #     "type": "lstm_cell",
    #     "decoding_dim": DECODER_DIM,
    #     "target_embedding_dim": TGT_EMB_DIM,
    #     "attention": {
    #       "type": "bilinear",
    #       "vector_dim": DECODER_DIM,
    #       "matrix_dim": ENCODER_DIM
    #     },
    #     "bidirectional_input": true
    #   },
    #   "max_decoding_steps": 100,
    #   "target_embedder": {
    #     # "type": "embedding",  # The argument type is already Embedding
    #     "embedding_dim": TGT_EMB_DIM,
    #     "vocab_namespace": "tgt_tokens",
    #   },
    #   "target_namespace": "tgt_tokens",
    #   "beam_size": 4
    # },
    "concat_audio": true
  },
  "data_loader": {
    "type": "pytorch_dataloader",
    "batch_size": 8,
    "shuffle": true
  },
  "vocabulary": {
    "type": "from_instances",
    "min_count": {
      "tgt_tokens": 5,
    }
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    # "learning_rate_scheduler": {
    #   "type": "polynomial_decay",
    #   # "num_epochs": 20,
    #   # "num_steps_per_epoch": 5139,
    #   "power": 1.0,
    #   "warmup_steps": 20000,
    #   "end_learning_rate": 0.0,
    # },
    "grad_norm": 0.1,
    "num_epochs": 300,
    "patience": 20,
    "validation_metric": "-rewrite_seq_WER",
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "cuda_device": 0
  }
}

