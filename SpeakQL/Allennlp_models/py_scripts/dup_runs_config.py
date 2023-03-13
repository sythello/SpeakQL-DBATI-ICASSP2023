import argparse
import os

random_seed_list = [None, 23333, 74751, 35980, 69544, 54022, 17872, 10977, 81773]
numpy_seed_list = [None, 2333, 4751, 1038, 7176, 3612, 741, 7393, 5554]
pytorch_seed_list = [None, 233, 751, 200, 921, 631, 399, 677, 319]





def main(args):
	mod = args.mod
	type_prefix = f'{args.type}_' if args.type else ''	# "rewriter_", "reranker_"; for e2e, no prefix 

	with open(f'train_configs/{type_prefix}{args.ver}.0{mod}.jsonnet', 'r') as f:
		orig_lines = f.read().strip().split('\n')
	if orig_lines[-2][-1] != ',':
		orig_lines[-2] += ','	# add comma to original last line

	for v in range(1, args.num_runs):
		out_path = f'train_configs/{type_prefix}{args.ver}.{v}{mod}.jsonnet'

		# if os.path.exists(out_path):
		# 	assert False, f'train_configs/{type_prefix}{args.ver}.{v}{mod}.jsonnet exists'

		with open(out_path, 'w') as f:
			out_lines = orig_lines[:-1] + [
					f'  "random_seed": {random_seed_list[v]},',
					f'  "numpy_seed": {numpy_seed_list[v]},',
					f'  "pytorch_seed": {pytorch_seed_list[v]},',
					orig_lines[-1],
				] 
			f.write('\n'.join(out_lines) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-v', '--ver', type=str, required=True)
	parser.add_argument('-m', '--mod', type=str, default='')
	parser.add_argument('-t', '--type', type=str, default='rewriter')
	parser.add_argument('-n', '--num_runs', type=int, default=5)


	args = parser.parse_args()

	main(args)