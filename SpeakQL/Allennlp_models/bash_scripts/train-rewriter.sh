serialize_dir=/vault/SpeakQL/Allennlp_models/runs

version=$1

regex="^([0-9\.]+)\.([0-9]+)([it]?)$"



if [[ -f train_configs/rewriter_${version}.jsonnet ]]; then
	config_version=${version}
elif [[ $version =~ $regex ]]; then
	config_version="${BASH_REMATCH[1]}"
	run_id="${BASH_REMATCH[2]}"
	spec="${BASH_REMATCH[3]}"

	config_version=${config_version}${spec}
fi

echo config_version=${config_version}
echo run_id=${run_id-(empty)}
echo spec=${spec-(empty)}

allennlp train train_configs/rewriter_${config_version}.jsonnet \
-s ${serialize_dir}/${version} \
--include-package dataset_readers \
--include-package models \
--include-package modules.encoder \
--include-package modules.copynet_decoder \
${@: 2}
