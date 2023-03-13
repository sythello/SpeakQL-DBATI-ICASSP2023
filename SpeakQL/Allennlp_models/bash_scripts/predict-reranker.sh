serialize_dir=/vault/SpeakQL/Allennlp_models/runs

version=$1

spider_dir=/vault/spider

mkdir -p outputs

## This is for non-siamese

allennlp predict ${serialize_dir}/${version}/model.tar.gz test \
--use-dataset-reader \
--output-file outputs/output-${version}.json \
--predictor spider_ASR_reranker_predictor \
--include-package dataset_readers \
--include-package models \
--silent \
--override \
"{
	\"dataset_reader\": {
	    \"for_training\": false,
	    \"samples_limit\": null,
	}
}" \
${@: 2}

python py_scripts/Reranker_output_to_dataset.py \
--test_path ${spider_dir}/my/dev/test_rewriter+phonemes.json \
--reranker_output_path outputs/output-${version}.json \
--output_path outputs/test-rewriter-${version}.json
