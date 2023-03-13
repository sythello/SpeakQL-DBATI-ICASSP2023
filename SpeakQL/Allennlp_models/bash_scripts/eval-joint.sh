set -e

rat_sql_dir="/users/yshao/rat-sql"
rat_sql_model_dir="/vault/rat-sql/logdir/glove_run"
speakql_outputs_dir="/users/yshao/SpeakQL/SpeakQL/Allennlp_models/outputs"
spider_dir="/vault/spider"

version=$1

mkdir -p $speakql_outputs_dir/ratsql-test-save

python eval/eval_joint.py \
--root_dir $rat_sql_dir  \
--exp_config_path $rat_sql_dir/experiments/spider-glove-run.jsonnet  \
--model_dir $rat_sql_model_dir  \
--checkpoint_step 40000  \
--eval_version $version  \
--pred_dataset_path $speakql_outputs_dir/test-rewriter-${version}.json  \
--test_output_path $speakql_outputs_dir/ratsql-test-save/${version}.json  \
--result_output_path $speakql_outputs_dir/ratsql-test-save/eval-${version}.json
