import json
from collections import defaultdict
import argparse

'''
Take as input json file, assert it's a list; output the flattened nested list
[[1,2,3],[4,5,6]] ==> [1,2,3,4,5,6]
'''

def _nested_list_flatten(l):
	if not isinstance(l, list):
		return [l]

	flattened = []
	for sub_l in l:
		flattened.extend(_nested_list_flatten(sub_l))
	return flattened

def main(args):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# tagger_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	in1_path = args.in1
	out_path = args.out

	with open(in1_path, 'r') as f:
		in1 = json.load(f)

	assert isinstance(in1, list), f'in1 is {type(in1)}'

	out = _nested_list_flatten(in1)
	print('Flattened length:', len(out))

	with open(out_path, 'w') as f:
	    json.dump(out, f, indent=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--in1', dest='in1', type=str)
	parser.add_argument('--out', dest='out', type=str)

	args = parser.parse_args()

	main(args)

	# print(_nested_list_flatten([
	# 	[1,2,3],
	# 	[{'a': 1}, {'b': 2}],
	# 	[[4,5,6], [7,8,9]],
	# ]))






