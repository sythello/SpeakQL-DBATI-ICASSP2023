serialize_dir=/vault/SpeakQL/Allennlp_models/runs

version=$1

spider_dir=/vault/spider

mkdir -p outputs

allennlp predict ${serialize_dir}/${version}/model.tar.gz test \
--use-dataset-reader \
--output-file outputs/output-${version}.json \
--predictor predictors.rewriter_s2s_predictor.SpiderASRRewriterPredictor_Seq2seq \
--include-package dataset_readers \
--include-package models \
--silent \
--override \
"{
	\"dataset_reader\": {
	    \"samples_limit\": null,
	}
}" \
${@: 2}


python py_scripts/S2S_output_to_dataset.py \
--test_path ${spider_dir}/my/dev/test_rewriter+phonemes.json \
--S2S_output_path outputs/output-${version}.json \
--output_path outputs/test-rewriter-${version}.json
