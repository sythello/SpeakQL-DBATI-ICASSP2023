import json
from collections import defaultdict
import argparse
from copy import deepcopy
import os
import re
import spacy

from nltk.tokenize.treebank import TreebankWordDetokenizer

from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
	Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS, \
	Postprocess_rewrite_seq_freeze_POS_v2, Postprocess_rewrite_seq_modify_POS_v2, \
	_detokenize


'''
Make a new test dataset file, adding ILM predictor outputs to each candidate; feed to eval scripts 
'''

def _Postprocess_rewrite_seq_wrapper(cand_dict, pred_dict):
	_tags = pred_dict['rewriter_tags']
	_rewrite_seq = pred_dict['rewrite_seq_prediction']
	_question_toks = cand_dict['question_toks']
	return Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)

class _Postprocess_POS_wrapper:
	def __init__(self, POS, mode, nlp=None):
		self.POS = POS
		assert mode in ('freeze', 'modify'), mode
		self.mode = mode
		self.nlp = nlp
	
	def __call__(self, cand_dict, pred_dict):
		_tags = pred_dict['rewriter_tags']
		_rewrite_seq = pred_dict['rewrite_seq_prediction']
		_question_toks = cand_dict['question_toks']
		
		if self.mode == 'freeze':
			_rewritten_toks = Postprocess_rewrite_seq_freeze_POS_v2(
				_tags,
				_rewrite_seq,
				_question_toks,
				freeze_POS=self.POS,
				nlp=self.nlp)
		else:
			_rewritten_toks = Postprocess_rewrite_seq_modify_POS_v2(
				_tags,
				_rewrite_seq,
				_question_toks,
				modify_POS=self.POS,
				nlp=self.nlp)
		return _rewritten_toks


def output_to_dataset(test_path,
					ILM_output_path,
					output_path,
					ILM_post_processor=None):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# ILM_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	# test_path = args.test_path
	# ILM_output_path = args.ILM_output_path
	# output_path = args.output_path

	ILM_output_jsons = []
	with open(test_path, 'r') as f:
		test_dataset_json = json.load(f)
	with open(ILM_output_path, 'r') as f:
		for l in f:
			ILM_output_jsons.append(json.loads(l))

	if ILM_post_processor is None:
		ILM_post_processor = _Postprocess_rewrite_seq_wrapper


	print(f'len(test_dataset_json) = {len(test_dataset_json)}, len(ILM_output_jsons) = {len(ILM_output_jsons)}')

	orig_test_samples_by_oid = defaultdict(list)
	ILM_output_by_oid = defaultdict(list)

	for d in ILM_output_jsons:
		o_id = d['original_id']
		ILM_output_by_oid[o_id].append(d)

	for d in test_dataset_json:
		if len(d) == 0:
			continue
			
		o_id = d[0]['original_id']
		
		for c in d:
			assert c['original_id'] == o_id 
			orig_test_samples_by_oid[o_id].append(c)

	print(f'len(orig_test_samples_by_oid) = {len(orig_test_samples_by_oid)}, len(ILM_output_by_oid) = {len(ILM_output_by_oid)}')

	ILM_output_test_dataset = []

	for o_id, _outputs in ILM_output_by_oid.items():
		_test_samples = orig_test_samples_by_oid[o_id]
		assert len(_test_samples) == len(_outputs)
		
		d = []
		for c, o in zip(_test_samples, _outputs):
			assert ' '.join(c['question_toks']) == o['question']
			_seq_len = len(c['question_toks'])
			# assert c['rewriter_tags'][:_seq_len] == o['rewriter_tags'][:_seq_len], f"{c['rewriter_tags']}\n{o['rewriter_tags']}"	## Not true, c['rewriter_tags'] is gold, o['rewriter_tags'] is tagger prediction
			assert all([_t == 'O' for _t in o['rewriter_tags'][_seq_len:]]), f"{o['rewriter_tags']}\n{o['rewriter_tags'][_seq_len:]}"
			if 'align_tags' in o:
				assert all([_t == '[O]' for _t in o['align_tags'][_seq_len:]]), f"{o['align_tags']}\n{o['align_tags'][_seq_len:]}"

			_rewritten_question_toks = ILM_post_processor(c, o)
			_rewritten_question = _detokenize(_rewritten_question_toks)
			
			c = deepcopy(c)
			c['predicted_rewriter_tags'] = o['rewriter_tags'][:_seq_len]
			if 'align_tags' in o:
				c['align_tags'] = o['align_tags'][:_seq_len]
			c['asr_question'] = c['question']
			c['asr_question_toks'] = c['question_toks']
			c['question'] = _rewritten_question
			c['question_toks'] = _rewritten_question_toks

			d.append(c)
		
		ILM_output_test_dataset.append(d)

	print(f'len(ILM_output_test_dataset) = {len(ILM_output_test_dataset)}')

	# output_test_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/test-rewriter-2.2tL.json'

	with open(output_path, 'w') as f:
		json.dump(ILM_output_test_dataset, f, indent=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_path', dest='test_path', type=str)
	parser.add_argument('--ILM_output_path', dest='ILM_output_path', type=str)
	parser.add_argument('--output_dir', dest='output_dir', type=str)
	parser.add_argument('--version', dest='version', type=str, default=None)

	args = parser.parse_args()

	if args.version is None:
		# infer the version
		_file_basename = os.path.basename(args.ILM_output_path)
		_re_match = re.match(r'output-(.*)\.json', _file_basename)
		if _re_match is None:
			raise ValueError(_file_basename)
		else:
			args.version = _re_match.group(1)
			print(f'Inferred version: {args.version}')

	# main(args)

	POS_LIST = ["PUNCT", "NUM", "VERB", "PRON", "ADP", "NOUN", "AUX", "DET",
		"SCONJ", "PART", "ADJ", "CCONJ", "ADV", "PROPN"]

	nlp = spacy.load('en_core_web_sm')

	# for mode in ['freeze', 'modify']:
	for mode in ['freeze']:
		for pos in POS_LIST:
			output_path = os.path.join(args.output_dir, f'test-rewriter-{args.version}-{mode}={pos}.json')
			ILM_post_processor = _Postprocess_POS_wrapper(POS=pos, mode=mode, nlp=nlp)

			output_to_dataset(test_path=args.test_path,
				ILM_output_path=args.ILM_output_path,
				output_path=output_path,
				ILM_post_processor=ILM_post_processor)







