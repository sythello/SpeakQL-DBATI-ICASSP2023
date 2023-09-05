# SpeakQL-DBATI-ICASSP2023

For any questions, please feel free to reach out to yshao@ucsd.edu !


## Spoken Spider Dataset

Link: [Google drive link](https://drive.google.com/file/d/1JcYjYygUM4rqIEhdjhKYkmDXxPSg_bP1/view?usp=drive_link)

For what follows, we assume that the spider dataset is located under directory `Dataset`.


## Data processing

(No need to run these code, processed data are already provided; if you are interested, you can reach out for more info)

Polly synthesizing: 'SpeakQL/spider_processing/question2speech.py'
Amazon ASR transcribing: 'SpeakQL/Amazon_transcribe/AmazonTranscribe.ipynb'

Polly-synthesized spoken questions: "Dataset/spider/my/[train|dev]/speech\_[mp3|wav]"
Polly-synthesized DB schemas: "Dataset/spider/my/db/speech\_[mp3|wav]"
Amazon ASR transcription outputs: "Dataset/spider/my/[train|dev]/spider\_[train|dev]\_batch0"

Getting phonemes (+ timestamps):
	'spider_processing/phoneme_align/phoneme-align.ipynb'
	Based on external tool Prosodylab-Aligner (given token & audio, aligns phonemes to timestamps)

## Experiments

- Training: 'SpeakQL/Allennlp_models/bash_scripts/train-XXX.sh' + version
- Predicting: 'SpeakQL/Allennlp_models/bash_scripts/predict-XXX.sh' + version
- Evaluating: 'SpeakQL/Allennlp_models/bash_scripts/eval-XXX.sh' + version'
	- Need our forked rat-sql: https://github.com/sythello/rat-sql
	- Our trained rat-sql checkpoints: [Google drive link](https://drive.google.com/file/d/12r-rP7Om_AeK3G4MwaXj-VBUrwKG0KWh/view?usp=share_link)

For example (full DBATI):
`cd SpeakQL/Allennlp_models`

`bash bash_scripts/train-rewriter.sh tagger.0t`

`bash bash_scripts/train-rewriter.sh ILM.0i`

`bash bash_scripts/predict-ILM-on-tagger-rewriter.sh tagger.0t ILM.0i`

`bash bash_scripts/eval-joint.sh tagger.0t-ILM.0i`

You can also refer to `run_exp.sh` for an example to run a batch of experiments.


