import json
from collections import defaultdict
import argparse
from copy import deepcopy

from nltk.tokenize.treebank import TreebankWordDetokenizer

from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
    Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS, \
    _detokenize


'''
Make a new test dataset file, adding ILM predictor outputs to each candidate; feed to eval scripts 
'''

def Oracle_ILM_rewrite(cand_dict, pred_dict=None, rewrite_seq_toks_save=None):
    ## rewrite_seq_toks_save: Dict, used to save the computed rewrite_seq_toks 
    ## This is oracle ILM, assume using tagger predictions
    ## Since we already had the test dataset with tagger prediction, all info is in cand_dict, no need for pred_dict (use None)
    
    tags = cand_dict['tagger_predicted_rewriter_tags']
    question_toks = cand_dict['question_toks']
    
    gold_toks = cand_dict['gold_question_toks']
    alignment_span_pairs = cand_dict['alignment_span_pairs']  # List[Tuple[List: src_span, List: tgt_span]]
    question_toks_rewritten = []
    rewrite_seq_toks = []

    _last_src_idx = -1
    for src_span, tgt_span in alignment_span_pairs:
        for i in range(_last_src_idx + 1, src_span[0]):
            # ignored src tokens 
            if tags[i].endswith('KEEP'):
                question_toks_rewritten.append(question_toks[i])
            else:
                # del or edit 
                pass
        
        if len(src_span) == len(tgt_span) or len(src_span) == 1:
            # treat src as single tokens 
            if len(src_span) == len(tgt_span):
                _src_spans = [[_idx] for _idx in src_span]
                _tgt_spans = [[_idx] for _idx in tgt_span]
            else:
                _src_spans = [src_span]
                _tgt_spans = [tgt_span]
                
            for _src_span, _tgt_span in zip(_src_spans, _tgt_spans):
                _idx = _src_span[0]
                if tags[_idx].endswith('KEEP'):
                    question_toks_rewritten.append(question_toks[_idx])
                elif tags[_idx].endswith('DEL'):
                    pass
                elif tags[_idx].endswith('EDIT'):
                    # edit to correct, i.e. append corresponding gold tokens 
                    for j in _tgt_span:
                        question_toks_rewritten.append(gold_toks[j])
                        rewrite_seq_toks.append(gold_toks[j])
                        if tags[_idx] in {'U-EDIT', 'L-EDIT'}:
                            rewrite_seq_toks.append('[ANS]')
        else:
            # multi-token            
            if all([tags[i].endswith('DEL') for i in src_span]):
                # if all del, del 
                pass
            elif all([tags[i].endswith('DEL') or tags[i].endswith('EDIT') for i in src_span]):
                # if all edit or del (at least 1 edit), edit 
                for j in tgt_span:
                    question_toks_rewritten.append(gold_toks[j])
            else:
                # otherwise (if has keep), keep; if not all keep, print warning 
                assert any([tags[i].endswith('KEEP') for i in src_span])
                if not all([tags[i].endswith('KEEP') for i in src_span]):
                    print(f'Unaddressed mismatch (treated as KEEP): {cand_dict["original_id"]}')
                    print('Target', [gold_toks[j] for j in tgt_span])
                    print('Source', [(question_toks[i], tags[i]) for i in src_span])

        _last_src_idx = src_span[-1]
                    
    if rewrite_seq_toks_save is not None:
        rewrite_seq_toks_save['rewrite_seq_toks'] = rewrite_seq_toks
            
    return question_toks_rewritten



# def _detokenize(toks):
#     detokenizer = TreebankWordDetokenizer()
#     return detokenizer.detokenize(toks)

def main(args):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# ILM_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	test_path = args.tagger_test_path
	# ILM_output_path = args.ILM_output_path
	output_path = args.output_path

	ILM_output_jsons = []
	with open(test_path, 'r') as f:
	    test_dataset_json = json.load(f)
	# with open(ILM_output_path, 'r') as f:
	#     for l in f:
	#         ILM_output_jsons.append(json.loads(l))

	print(f'len(test_dataset_json) = {len(test_dataset_json)}')

	# orig_test_samples_by_oid = defaultdict(list)
	# ILM_output_by_oid = defaultdict(list)

	# for d in ILM_output_jsons:
	#     o_id = d['original_id']
	#     ILM_output_by_oid[o_id].append(d)

	# for d in test_dataset_json:
	#     if len(d) == 0:
	#         continue
	        
	#     o_id = d[0]['original_id']
	    
	#     for c in d:
	#         assert c['original_id'] == o_id 
	#         orig_test_samples_by_oid[o_id].append(c)

	# print(f'len(orig_test_samples_by_oid) = {len(orig_test_samples_by_oid)}, len(ILM_output_by_oid) = {len(ILM_output_by_oid)}')

	ILM_output_test_dataset = []

	# for o_id, _outputs in ILM_output_by_oid.items():
	#     _test_samples = orig_test_samples_by_oid[o_id]
	#     assert len(_test_samples) == len(_outputs)
	for _test_samples in test_dataset_json:
	    d = []
	    # for c, o in zip(_test_samples, _outputs):
	    for c in _test_samples:
	        _seq_len = len(c['question_toks'])
	        # assert all([_t == 'O' for _t in o['rewriter_tags'][_seq_len:]]), f"{o['rewriter_tags']}\n{o['rewriter_tags'][_seq_len:]}"
	        # assert all([_t == '[O]' for _t in o['align_tags'][_seq_len:]]), f"{o['align_tags']}\n{o['align_tags'][_seq_len:]}"

	        _rewritten_question_toks = Oracle_ILM_rewrite(c, None)
	        _rewritten_question = _detokenize(_rewritten_question_toks)
	        
	        c = deepcopy(c)
	        # c['predicted_rewriter_tags'] = o['rewriter_tags'][:_seq_len]
	        # c['align_tags'] = o['align_tags'][:_seq_len]
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

	parser.add_argument('--tagger_test_path', dest='tagger_test_path', type=str)
	parser.add_argument('--output_path', dest='output_path', type=str)

	args = parser.parse_args()

	main(args)







