serialize_dir=/vault/SpeakQL/Allennlp_models/runs

version=$1

SPIDER_DIR=/vault/spider

mkdir -p outputs

allennlp predict ${serialize_dir}/${version}/model.tar.gz test \
--use-dataset-reader \
--output-file outputs/output-${version}.json \
--predictor spider_ASR_rewriter_predictor_tagger \
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
