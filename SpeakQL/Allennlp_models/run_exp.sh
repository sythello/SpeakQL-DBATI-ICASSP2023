set -e

for ((r=0;r<5;r++)); do
	bash bash_scripts/train-rewriter.sh tagger.${r}t
	bash bash_scripts/train-rewriter.sh ILM.${r}i
	bash bash_scripts/predict-ILM-on-tagger-rewriter.sh tagger.${r}t ILM.${r}i
done

bash bash_scripts/eval-joint-batch.sh  tagger.0t-ILM.0i  tagger.1t-ILM.1i  tagger.2t-ILM.2i  tagger.3t-ILM.3i  tagger.4t-ILM.4i


