import os
import json
from collections import defaultdict
import argparse
import shutil
import random

'''
	Generate dataset files for few-shot testing.
	Output 4 files:
		train (unmodified),
		train_fs_extra: taking K samples from each dev/test DB
		dev: remove samples in train_fs_extra
		test: remove samples in train_fs_extra
'''

def main(args):

	# Train set doesn't need modify
	shutil.copy(args.train_in_path, args.train_out_path)

	# with open(args.train_in_path, 'r') as f:
	#	 train_in_samples = json.load(f)
	with open(args.dev_in_path, 'r') as f:
		dev_in_samples = json.load(f)
	with open(args.test_in_path, 'r') as f:
		test_in_samples = json.load(f)

	print('Input samples len')
	print(f'len(dev_in_samples) = {len(dev_in_samples)}')
	print(f'len(test_in_samples) = {len(test_in_samples)}')
	print()

	dev_db2samples = defaultdict(list)
	test_db2samples = defaultdict(list)
	for s in dev_in_samples:
		dev_db2samples[s[0]['db_id']].append(s)
	for s in test_in_samples:
		test_db2samples[s[0]['db_id']].append(s)
	
	print('DB ids')
	print([(db_id, len(s_list)) for db_id, s_list in dev_db2samples.items()])
	print([(db_id, len(s_list)) for db_id, s_list in test_db2samples.items()])
	print()

	train_extra_samples = []
	dev_out_samples = []
	test_out_samples = []
	random.seed(233)
	for db_id, s_list in dev_db2samples.items():
		random.shuffle(s_list)
		train_extra_samples.extend(s_list[:args.k])
		dev_out_samples.extend(s_list[args.k:])
	for db_id, s_list in test_db2samples.items():
		random.shuffle(s_list)
		train_extra_samples.extend(s_list[:args.k])
		test_out_samples.extend(s_list[args.k:])

	print('Output samples len')
	print(len(train_extra_samples))
	print(len(dev_out_samples))
	print(len(test_out_samples))
	print()

	with open(args.train_extra_out_path, 'w') as f:
		json.dump(train_extra_samples, f, indent=2)
	with open(args.dev_out_path, 'w') as f:
		json.dump(dev_out_samples, f, indent=2)
	with open(args.test_out_path, 'w') as f:
		json.dump(test_out_samples, f, indent=2)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# default_spider_dir = "/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider"
	default_spider_dir = "/vault/spider"

	parser.add_argument('-k', '--k', type=int, required=True)
	parser.add_argument('--spider_dir', type=str,
		default=default_spider_dir)
	parser.add_argument('--train_in_path', type=str,
		default=os.path.join(default_spider_dir, "my/train/train_rewriter+phonemes.json"))
	parser.add_argument('--dev_in_path', type=str,
		default=os.path.join(default_spider_dir, "my/dev/dev_rewriter+phonemes.json"))
	parser.add_argument('--test_in_path', type=str,
		default=os.path.join(default_spider_dir, "my/dev/test_rewriter+phonemes.json"))
	parser.add_argument('--train_out_path', type=str,
		default=None)
	parser.add_argument('--train_extra_out_path', type=str,
		default=None)
	parser.add_argument('--dev_out_path', type=str,
		default=None)
	parser.add_argument('--test_out_path', type=str,
		default=None)

	args = parser.parse_args()

	os.makedirs(os.path.join(default_spider_dir, f"my/few-shot"), exist_ok=True)
	if args.train_out_path is None:
		args.train_out_path = os.path.join(default_spider_dir, f"my/few-shot/train-K={args.k}.json")
	if args.train_extra_out_path is None:
		args.train_extra_out_path = os.path.join(default_spider_dir, f"my/few-shot/train_fs_extra-K={args.k}.json")
	if args.dev_out_path is None:
		args.dev_out_path = os.path.join(default_spider_dir, f"my/few-shot/dev-K={args.k}.json")
	if args.test_out_path is None:
		args.test_out_path = os.path.join(default_spider_dir, f"my/few-shot/test-K={args.k}.json")

	main(args)







