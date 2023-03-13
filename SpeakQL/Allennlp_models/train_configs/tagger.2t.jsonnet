# Adding align tags 

local AUDIO_DIM_POOLING = 136;
local AUDIO_DIM_NO_POOLING = 68;

local TAG_EMB_DIM = 64;
local SRC_EMB_DIM = 768; # BERT 
local TGT_EMB_DIM = 300;
local AUDIO_ENC_DIM = 128;

local ENCODER_DIM = 256;
local TAGGING_FF_DIM = 64;
local DECODER_DIM = ENCODER_DIM; # It seems that otherwise it can't work 

# local SPIDER_DIR = "/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider";
# local SPIDER_DIR = "/vault/dataset/yshao/spider";
local SPIDER_DIR = "/vault/spider";

# local TABERT_MODEL_PATH = "/Users/mac/Desktop/syt/Deep-Learning/Repos/TaBERT/pretrained-models/tabert_base_k1/model.bin";
local TABERT_MODEL_PATH = "/vault/TaBERT/pretrained-models/tabert_base_k1/model.bin";

{
  "dataset_reader": {
    "type": "spider_ASR_rewriter_reader_tagger_comb",
    "src_token_indexers": {
      "bert": {
        "type": "pretrained_transformer_mismatched",
        "model_name": "bert-base-uncased",
      }
    },
    "include_align_tags": true,
    "tables_json_fname": SPIDER_DIR + "/tables.json",
    "dataset_dir": SPIDER_DIR + "/my",
    "databases_dir": SPIDER_DIR + "/database",
    "use_tabert": false,
    "tabert_model_path": TABERT_MODEL_PATH,
    "specify_full_path": false,
    "max_sequence_len": 300,
    "include_gold_tags": true,
    "debug": false
  },
  "train_data_path": "train",
  "validation_data_path": "dev",
  "model": {
    "type": "spider_ASR_rewriter_tagger_comb",
    # "src_text_embedder": null,
    "src_text_embedder": {
      "token_embedders": {
        "bert": {
          "type": "pretrained_transformer_mismatched",
          "model_name": "bert-base-uncased",
          "train_parameters": false,
          "last_layer_only": false
        },
      },
    },
    "align_tag_embedder": {
      "type": "embedding",
      "embedding_dim": TAG_EMB_DIM,
      "vocab_namespace": "align_tags",
    },
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
    "encoder": {
      "type": "v1",
      "audio_attention_layer": {
        "type": "cosine"
      },
      "audio_attention_residual": "+",
      "seq2seq_encoders": [
        {
          "type": "lstm",
          "input_size": (SRC_EMB_DIM + AUDIO_ENC_DIM + TAG_EMB_DIM),
          "hidden_size": ENCODER_DIM / 2, # bidirectional
          "num_layers": 1,
          "dropout": 0.0,
          "bidirectional": true
        }
      ],
      "seq2vec_encoder": null
    },
    "ff_dimension": TAGGING_FF_DIM,
    "concat_audio": true
  },
  "data_loader": {
    "type": "pytorch_dataloader",
    "batch_size": 8,
    "shuffle": true
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "grad_norm": 0.1,
    "num_epochs": 300,
    "patience": 10,
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "cuda_device": 0
  },
  "random_seed": 74751,
  "numpy_seed": 4751,
  "pytorch_seed": 751,
}

