set -e

rat_sql_dir="/users/yshao/rat-sql"
rat_sql_model_dir="/vault/rat-sql/logdir/glove_run"
speakql_outputs_dir="/users/yshao/SpeakQL/SpeakQL/Allennlp_models/outputs"
eval_out_dir="/users/yshao/SpeakQL/SpeakQL/Allennlp_models/outputs/ratsql-test-save"
spider_dir="/vault/spider"

# rat_sql_dir="/Users/mac/Desktop/syt/Deep-Learning/Repos/rat-sql"
# rat_sql_model_dir="/Users/mac/Desktop/syt/Deep-Learning/Repos/rat-sql/logdir/glove_run/bs=20,lr=7.4e-04,end_lr=0e0,att=0"
# speakql_outputs_dir="/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs"
# eval_out_dir=$speakql_outputs_dir/ratsql-test-save
# spider_dir="/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider"

versions=$@

mkdir -p $eval_out_dir

python eval/eval_joint_batch.py \
--root_dir $rat_sql_dir  \
--exp_config_path $rat_sql_dir/experiments/spider-glove-run.jsonnet  \
--model_dir $rat_sql_model_dir  \
--checkpoint_step 40000  \
--eval_vers $versions  \
--eval_in_dir $speakql_outputs_dir \
--eval_out_dir $eval_out_dir
