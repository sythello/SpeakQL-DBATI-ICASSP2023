# (Base 1.15) - all proposed modules ("clean" setting: -all tags, -char_emb, -DB tokens, -phonemes (-aux_ph_loss, -ph_attention), -audio_feats)


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
local FF_DIM = 64;

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
    "type": "spider_ASR_reranker_reader_v2_comb_new",
    "src_token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "model_name": PLM_MODEL_NAME,
      },
      # "char": {
      #   "type": "characters",
      #   "namespace": "token_characters",
      #   "min_padding_length": 5,
      # },
    },
    # "tgt_token_indexers": {
    #   "tgt_tokens": {
    #     "type": "single_id",
    #     "namespace": "tgt_tokens"
    #   }
    # },
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
    "use_tabert": false,
    "tabert_model_path": TABERT_MODEL_PATH,
    "use_phoneme_inputs": false,
    "use_phoneme_labels": false,   # multi-label
    "default_phoneme_slices": 4,
    "pronun_dict_path": CMU_DICT_PATH,
    "db_tok2phs_dict_path": DB_TOK2PHS_PATH,
    "specify_full_path": false,
    "max_sequence_len": 300,
  },
  "train_data_path": "train",
  "validation_data_path": "dev",
  "model": {
    "type": "spider_ASR_reranker_v2_comb_new",
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
        # "char": {
        #   "type": "character_encoding",
        #   # TokenCharactersEncoder(subclass of TokenEmbedder)
        #   "embedding": {
        #     "embedding_dim": CHAR_EMB_DIM,
        #     "vocab_namespace": "token_characters",
        #   },
        #   "encoder": {
        #     "type": "cnn",
        #     "embedding_dim": CHAR_EMB_DIM,
        #     "num_filters": 4,
        #     "ngram_filter_sizes": [2, 3, 4, 5],
        #     "output_dim": CHAR_EMB_DIM
        #   },
        #   "dropout": 0.0,
        # },
      },
    },
    "use_audio": false,
    "use_db_input": false,
    # "align_tag_embedder": {
    #   "type": "embedding",
    #   "embedding_dim": TAG_EMB_DIM,
    #   "vocab_namespace": "align_tags",
    # },
    # "phoneme_tag_embedder": {
    #   "token_embedders": {
    #     [PHONEME_INDEXER_KEY]: {
    #       "type": "embedding",
    #       "embedding_dim": TAG_EMB_DIM,
    #       "vocab_namespace": PHONEME_VOCAB_NAMESPACE,
    #     }
    #   }
    # },
    "use_tabert": false,
    "tabert_model_path": TABERT_MODEL_PATH,
    "finetune_tabert": false,
    # "audio_seq2vec_encoder": {
    #   "type": "cnn",
    #   "embedding_dim": AUDIO_DIM_NO_POOLING,
    #   "num_filters": 4,
    #   "ngram_filter_sizes": [2, 3, 4, 5],
    #   "output_dim": AUDIO_ENC_DIM
    # },
    # "phoneme_tag_seq2vec_encoder": {
    #   "type": "cnn",
    #   "embedding_dim": TAG_EMB_DIM,
    #   "num_filters": TAG_EMB_DIM / 4,
    #   "ngram_filter_sizes": [2, 2, 3, 4],
    #   "output_dim": TAG_EMB_DIM
    # },
    "using_phoneme_input": false,
    # "ph2tok_audio_seq2vec_encoder": {
    #   "type": "cnn",
    #   "embedding_dim": AUDIO_ENC_DIM,
    #   "num_filters": 4,
    #   "ngram_filter_sizes": [2, 3],
    #   "output_dim": AUDIO_ENC_DIM
    # },
    "ph2tok_audio_seq2vec_encoder": null,
    "using_phoneme_labels": false,  # single-classification label
    "phoneme_loss_coef": 0.0,
    "using_phoneme_multilabels": false,
    "phoneme_multilabel_loss_coef": 0.0,
    "encoder": {
      "type": "v1",
      # "audio_attention_layer": {
      #   "type": "cosine"
      # },
      # "audio_attention_residual": "+",
      "attention_type": null,
      # "attention_layer": {
      #   "type": "cosine"
      # },
      # "attention_residual": "+",
      "seq2seq_encoders": [
        {
          "type": "lstm",
          "input_size": SRC_EMB_DIM,
          "hidden_size": ENCODER_DIM / 2, # bidirectional
          "num_layers": 1,
          "dropout": 0.0,
          "bidirectional": true
        }
      ],
      "seq2vec_encoder": {
        "type": "lstm",
        "input_size": ENCODER_DIM,
        "hidden_size": ENCODER_DIM / 2, # bidirectional
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": true
      }
    },
    "using_gated_fusion": false,
    "using_ref_att_loss": false,
    "ref_att_loss_coef": 0.0,
    "ff_dimension": FF_DIM,
    "concat_audio": true
  },
  "data_loader": {
    "type": "pytorch_dataloader",
    "batch_size": 8,
    "shuffle": true
  },
  # "vocabulary": {
  #   "type": "from_instances",
  #   "min_count": {
  #     "tgt_tokens": 2,
  #   }
  # },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
    },
    # "learning_rate_scheduler": {
    #   "type": "polynomial_decay",
    #   # "num_epochs": 20,
    #   # "num_steps_per_epoch": 5139,
    #   "power": 1.0,
    #   "warmup_steps": 20,
    #   "end_learning_rate": 0.0,
    # },
    "grad_norm": 0.1,
    "num_epochs": 300,
    "patience": 30,
    "validation_metric": "-oracle_label_NLL",
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "cuda_device": 0
  },
  "random_seed": 35980,
  "numpy_seed": 1038,
  "pytorch_seed": 200,
}
