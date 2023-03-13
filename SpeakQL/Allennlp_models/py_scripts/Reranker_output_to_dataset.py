import json
from collections import defaultdict
import argparse
from copy import deepcopy

from nltk.tokenize.treebank import TreebankWordDetokenizer

from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
    Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS, \
    _detokenize


'''
Make a new test dataset file, adding reranker predictor outputs to each candidate; feed to eval scripts 
'''

# def _Postprocess_rewrite_seq_wrapper(cand_dict, pred_dict):
#     _tags = pred_dict['rewriter_tags']
#     _rewrite_seq = pred_dict['rewrite_seq_prediction']
#     _question_toks = cand_dict['question_toks']
#     return Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)

def main(args):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# reranker_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	test_path = args.test_path
	reranker_output_path = args.reranker_output_path
	output_path = args.output_path

	reranker_output_jsons = []
	with open(test_path, 'r') as f:
	    test_dataset_json = json.load(f)
	with open(reranker_output_path, 'r') as f:
	    for l in f:
	        reranker_output_jsons.append(json.loads(l))

	print(f'len(test_dataset_json) = {len(test_dataset_json)}, len(reranker_output_jsons) = {len(reranker_output_jsons)}')

	orig_test_samples_by_oid = defaultdict(list)
	reranker_output_by_oid = defaultdict(list)

	for d in reranker_output_jsons:
	    o_id = d['original_id']
	    reranker_output_by_oid[o_id].append(d)

	for d in test_dataset_json:
	    if len(d) == 0:
	        continue
	        
	    o_id = d[0]['original_id']
	    
	    for c in d:
	        assert c['original_id'] == o_id 
	        orig_test_samples_by_oid[o_id].append(c)

	print(f'len(orig_test_samples_by_oid) = {len(orig_test_samples_by_oid)}, len(reranker_output_by_oid) = {len(reranker_output_by_oid)}')

	reranker_output_test_dataset = []

	for o_id, _outputs in reranker_output_by_oid.items():
	    _test_samples = orig_test_samples_by_oid[o_id]
	    assert len(_test_samples) == len(_outputs)
	    
	    d = []
	    for c, o in zip(_test_samples, _outputs):
	        assert ' '.join(c['question_toks']) == o['question']
	        # _seq_len = len(c['question_toks'])
	        # assert c['rewriter_tags'][:_seq_len] == o['rewriter_tags'][:_seq_len], f"{c['rewriter_tags']}\n{o['rewriter_tags']}"	## Not true, c['rewriter_tags'] is gold, o['rewriter_tags'] is tagger prediction
	        # assert all([_t == 'O' for _t in o['rewriter_tags'][_seq_len:]]), f"{o['rewriter_tags']}\n{o['rewriter_tags'][_seq_len:]}"
	        # assert all([_t == '[O]' for _t in o['align_tags'][_seq_len:]]), f"{o['align_tags']}\n{o['align_tags'][_seq_len:]}"

	        # _rewritten_question_toks = o['reranker_prediction']
	        # _rewritten_question = _detokenize(_rewritten_question_toks)
	        
	        c = deepcopy(c)
	        # c['align_tags'] = o['align_tags'][:_seq_len]
	        c['score_preds'] = o['score_preds']

	        d.append(c)

	    # do the "reranking" based on predicted score, high -> low
	    d.sort(key=lambda c: c['score_preds'], reverse=True)
	    
	    reranker_output_test_dataset.append(d)

	print(f'len(reranker_output_test_dataset) = {len(reranker_output_test_dataset)}')

	# output_test_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/test-rewriter-2.2tL.json'

	with open(output_path, 'w') as f:
	    json.dump(reranker_output_test_dataset, f, indent=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_path', dest='test_path', type=str)
	parser.add_argument('--reranker_output_path', dest='reranker_output_path', type=str)
	parser.add_argument('--output_path', dest='output_path', type=str)

	args = parser.parse_args()

	main(args)







