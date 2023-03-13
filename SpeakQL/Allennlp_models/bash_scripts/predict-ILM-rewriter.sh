serialize_dir=/vault/SpeakQL/Allennlp_models/runs

version=$1

spider_dir=/vault/spider

mkdir -p outputs

allennlp predict ${serialize_dir}/${version}/model.tar.gz test \
--use-dataset-reader \
--output-file outputs/output-${version}-oracle-tags.json \
--predictor spider_ASR_rewriter_predictor_ILM \
--include-package dataset_readers \
--include-package models \
--include-package predictors \
--silent \
--override \
"{
	\"dataset_reader\": {
	    \"samples_limit\": null,
	}
}" \
${@: 2}


python py_scripts/ILM_output_to_dataset.py \
--test_path ${spider_dir}/my/dev/test_rewriter+phonemes.json \
--ILM_output_path outputs/output-${version}-oracle-tags.json \
--output_path outputs/test-rewriter-${version}-oracle-tags.json

