## Data Processing

### Programs

'SpeakQL/spider_processing/DataProcessing_Reranker.ipynb': Processing of spider (for reranker), including:
	Evaluate: (SQL, SQL) -> score	// partial match
	Build IRNet input: replace "question" and "question_tok" fields with transcriptions, and add audio timestamps
	Build Reranker input: add SQL predictions from IRNet and evaluated scores to samples; put all transcription samples for the same original question (i.e. "original_id") together; split official dev into dev/test based on index files
	Dev/test split: create index files to split official dev set into dev/test set for SpeakQL
'SpeakQL/spider_processing/DataProcessing_Rewriter.ipynb':
	Using FastAlign and custom heuristics to get the alignments between transcription candidates and gold questions.
	Build Rewriter input (inheriting the dev/test split of reranker)
'SpeakQL/spider_processing/DB_tokens_audio.ipynb': Extract audios for DB tokens
'SpeakQL/spider_processing/json2sep.py': (For official train/dev jsons) Extract the questions, queries and db_ids, putting them in separate txt files.


'rat-sql/ratsql-infer.ipynb': Anything that requires ratsql prediction.
	Data preprocessing: load original dataset and transcription files, generate a dataset-like file with rat-sql predicted SQL (but yet no partial match score).
	Evaluation: load original (gold, non-ASR) text, predict SQL and evaluate
	Evaluation: load (rewriter) model output file, postprocess into text questions, predict SQL and evaluate

'SpeakQL/spider_processing/DataProcessing_Reranker_ratsql.ipynb': Processing of spider for reranker, using rat-sql. Including:
	Evaluate: (SQL, SQL) -> score	// partial match
	Build Reranker input: add SQL predictions from ratsql and evaluated scores to samples; put all transcription samples for the same original question (i.e. "original_id") together; split official dev into dev/test based on index files
	Build BRIDGE input: build a "fake" spider directory (spider/my/Bridge-asr) with ASR data, to utilize BRIDGE on ASR


'SpeakQL/SpeakQL/Allennlp_models/py_script/':
	'tagger_output_to_dataset.py': Take as input the test dataset and tagger model prediction file, make a new test dataset file, adding tagger predictor outputs to each candidate on key 'tagger_predicted_rewriter_tags'. Then feed this file to ILM predictor instead of original test dataset file, for full-fledge eval.


### Data Files

'spider/my/dev/dev_asr_amazon.json': Spider-style dataset, for each sample the "question" and "question_tok" are replaced with a transcription candidate. Also added a field "original_id" to track its ID in original dev set.
'spider/my/train/train_asr_amazon.json': (same as above)
'spider/my/[train|dev]/[train|dev]_asr_amazon_RatsqlPredicted.json': on top of [train|dev]_asr_amazon.json, added ratsql predicted SQLs for each transcription candidate.


'IRNet/data/data_asr_amazon/[train|dev].json': IRNet preprocessed '[train|dev]_asr_amazon.json'. Ready to go to IRNet
'IRNet/output/asr_amazon/[output|gold].json': IRNet output of [train|dev].json, one per line. Need realign via 'my-data-check.ipynb'
'IRNet/output/asr_amazon/[output|gold]_full.json': Realigned output file; blank line added for each skipped sample.
'my-data-check.ipynb': Align output and original file ('[train|dev]_asr_amazon.json') by checking the question string match.

'spider/my/dev/dev_reranker(full).json': adding to 'dev_asr_amazon.json' the NLIDB prediction and evaluation score of each sample.
'spider/my/dev/[dev|test]_reranker.json': separated dev_reranker(full).json into dev/test set.

'rat-sql/output/test-predicted-{VERSION}.json': the output of {VERSION} model (rewriter), the rewritten questions, rat-sql predictions and scores of test samples.


## Temp Experiments / Implementation workspace
'SpeakQL/SpeakQL/Allennlp/Reranker|Rewriter.ipynb':
	Initial implementation of dataset reader, model and predictor



## Observation

'SpeakQL/experiments/ASR_result_analysis.ipynb': Check the quality of ASR results and align them (min edit distance) with gold text. Check the quality of audio segmentation.

'SpeakQL/spider_processing/DB_tokens_audio.ipynb':
	Get stats for DB tokens
	Get phonemes for DB tokens
	Call Polly service to generate audios for DB tokens
	Get (pre-compute) the audio features for DB tokens

'SpeakQL/SpeakQL/Allennlp/Results-analysis.ipynb':
	Model aggregation (agreement-based)
	Samples with large improvements
	Plotting results

'SpeakQL/SpeakQL/Allennlp/Rewriter-model-analysis.ipynb':
	Check the internal values (audio feats, audio attention map, gate values, etc.)
	Check if audio attention degrades

'SpeakQL/spider_processing/DB_content_retrieval.ipynb':
	Test audio/phonemes retrieval performance



## Current Working Pipeline

Polly synthesizing: 'SpeakQL/spider_processing/question2speech.py'
Amazon ASR transcribing: 'SpeakQL/Amazon_transcribe/AmazonTranscribe.ipynb' // Contains private keys!!

Polly-synthesized spoken questions: 'Dataset/spider/my/[train|dev]/speech_[mp3|wav]'
Polly-synthesized DB schemas: 'Dataset/spider/my/db/speech_[mp3|wav]'
Amazon ASR transcription outputs: 'Dataset/spider/my/[train|dev]/spider_[train|dev]_batch0'

Getting phonemes (+ timestamps):
	'spider_processing/phoneme_align/phoneme-align.ipynb'
	Based on external tool Prosodylab-Aligner (given token & audio, aligns phonemes to timestamps)

Data preprocessing: 'SpeakQL/spider_processing/DataProcessing_[Reranker|Rewriter]_ratsql.ipynb'
Training: 'SpeakQL/SpeakQL/Allennlp_models/bash_scripts/train-XXX.sh' + version
Predicting: 'SpeakQL/SpeakQL/Allennlp_models/bash_scripts/predict-XXX.sh' + version
Evaluating:
	Rat-sql score: 'rat-sql/ratsql-infer.ipynb'
	BLEU & accuracy@1 (for rerankers): 'SpeakQL/SpeakQL/Allennlp_models/Reranker.ipynb'
	BLEU (for rewriters): 'SpeakQL/SpeakQL/Allennlp_models/Rewriter.ipynb'
Rewriter model interpretation (internal values visualization): 'SpeakQL/SpeakQL/Allennlp_models/Rewriter-model-analysis.ipynb'


## Model version name
(legacy)
-> Combined (combining code for BERT and Tabert)
-> Combined_new (using base class)


