set -e

serialize_dir=/vault/SpeakQL/Allennlp_models/runs

tagger_version=$1
ILM_version=$2

spider_dir=/vault/spider

mkdir -p outputs


# TAG_GIVEN set and > 0 --> skip tagger prediction
# otherwise (unset or =0) --> do tagger prediction

if [[ ! $TAG_GIVEN -gt 0 ]]; then
	allennlp predict ${serialize_dir}/${tagger_version}/model.tar.gz test \
	--use-dataset-reader \
	--output-file outputs/output-${tagger_version}.json \
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
	}"
fi


python py_scripts/tagger_output_to_dataset.py \
--test_path ${spider_dir}/my/dev/test_rewriter+phonemes.json \
--tagger_output_path outputs/output-${tagger_version}.json \
--output_path outputs/test-rewriter-${tagger_version}.json


allennlp predict ${serialize_dir}/${ILM_version}/model.tar.gz "test:outputs/test-rewriter-${tagger_version}.json" \
--use-dataset-reader \
--output-file outputs/output-${tagger_version}-${ILM_version}.json \
--predictor spider_ASR_rewriter_predictor_ILM \
--include-package dataset_readers \
--include-package models \
--include-package predictors \
--silent \
--override \
"{
	\"dataset_reader\": {
	    \"specify_full_path\": true,
	    \"include_gold_rewrite_seq\": false,
	    \"use_tagger_prediction\": true,
	    \"samples_limit\": null,
	}
}"


python py_scripts/ILM_output_to_dataset.py \
--test_path ${spider_dir}/my/dev/test_rewriter+phonemes.json \
--ILM_output_path outputs/output-${tagger_version}-${ILM_version}.json \
--output_path outputs/test-rewriter-${tagger_version}-${ILM_version}.json


