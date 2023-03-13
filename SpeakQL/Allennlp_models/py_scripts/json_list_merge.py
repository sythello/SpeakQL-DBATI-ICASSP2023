import json
from collections import defaultdict
import argparse

'''
Take as input 2 json files, assert they are lists; output the concatenation of them as a new json file
'''

def main(args):
	# test_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/my/dev/test_rewriter.json'
	# tagger_output_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/Modeling/Allennlp_models/outputs/local-test/output-2.2tL.json'

	in1_path = args.in1
	in2_path = args.in2
	out_path = args.out

	## Wtf is this...? Should "in1/2_path" be "f"?
	with open(in1_path, 'r') as f:
		in1 = json.load(f)
	with open(in2_path, 'r') as f:
		in2 = json.load(f)

	assert isinstance(in1, list), f'in1 is {type(in1)}'
	assert isinstance(in2, list), f'in2 is {type(in1)}'

	out = in1 + in2

	with open(out_path, 'w') as f:
	    json.dump(out, f, indent=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--in1', dest='in1', type=str)
	parser.add_argument('--in2', dest='in2', type=str)
	parser.add_argument('--out', dest='out', type=str)

	args = parser.parse_args()

	main(args)







