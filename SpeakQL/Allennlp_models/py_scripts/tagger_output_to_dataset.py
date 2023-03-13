import json
from collections import defaultdict
import argparse

'''
Make a new test dataset file, adding tagger predictor outputs to each candidate as 'rewrite_tags'; then feed to ILM predictor 
'''

def main(args):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# tagger_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	test_path = args.test_path
	tagger_output_path = args.tagger_output_path
	output_path = args.output_path

	tagger_output_jsons = []
	with open(test_path, 'r') as f:
	    test_dataset_json = json.load(f)
	with open(tagger_output_path, 'r') as f:
	    for l in f:
	        tagger_output_jsons.append(json.loads(l))

	print(f'len(test_dataset_json) = {len(test_dataset_json)}, len(tagger_output_jsons) = {len(tagger_output_jsons)}')

	orig_test_samples_by_oid = defaultdict(list)
	tagger_output_by_oid = defaultdict(list)

	for d in tagger_output_jsons:
	    o_id = d['original_id']
	    tagger_output_by_oid[o_id].append(d)

	for d in test_dataset_json:
	    if len(d) == 0:
	        continue
	        
	    o_id = d[0]['original_id']
	    
	    for c in d:
	        assert c['original_id'] == o_id 
	        orig_test_samples_by_oid[o_id].append(c)

	print(f'len(orig_test_samples_by_oid) = {len(orig_test_samples_by_oid)}, len(tagger_output_by_oid) = {len(tagger_output_by_oid)}')

	tagger_output_test_dataset = []

	for o_id, _outputs in tagger_output_by_oid.items():
	    _test_samples = orig_test_samples_by_oid[o_id]
	    assert len(_test_samples) == len(_outputs)
	    
	    d = []
	    for c, o in zip(_test_samples, _outputs):
	        assert ' '.join(c['question_toks']) == o['question']
	        _seq_len = len(c['question_toks'])
	        assert c['rewriter_tags'][:_seq_len] == o['gold_tags'][:_seq_len]
	        
	        c['tagger_predicted_rewriter_tags'] = o['tags_prediction']
	        d.append(c)
	    
	    tagger_output_test_dataset.append(d)

	print(f'len(tagger_output_test_dataset) = {len(tagger_output_test_dataset)}')

	# output_test_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/test-rewriter-2.2tL.json'

	with open(output_path, 'w') as f:
	    json.dump(tagger_output_test_dataset, f, indent=4)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--test_path', dest='test_path', type=str)
	parser.add_argument('--tagger_output_path', dest='tagger_output_path', type=str)
	parser.add_argument('--output_path', dest='output_path', type=str)

	args = parser.parse_args()

	main(args)







