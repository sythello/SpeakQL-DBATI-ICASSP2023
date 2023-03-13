import json
import os
from tqdm import tqdm

def separate_files(in_json_path, out_path):
	db_id_out_path = os.path.join(out_path, 'db_id.txt')
	question_out_path = os.path.join(out_path, 'question.txt')
	query_out_path = os.path.join(out_path, 'query.txt')

	os.makedirs(out_path, exist_ok=True)

	with open(in_json_path, 'r') as f:
		dataset = json.load(f)

	f_db_id = open(db_id_out_path, 'w')
	f_question = open(question_out_path, 'w')
	f_query = open(query_out_path, 'w')

	for i, datum in tqdm(enumerate(dataset)):
		db_id = datum['db_id']
		question = datum['question']
		query = ' '.join(datum['query_toks'])

		f_db_id.write(db_id + '\n')
		f_question.write(question + '\n')
		f_query.write(query + '\n')

	f_db_id.close()
	f_question.close()
	f_query.close()


if __name__ == '__main__':
	spider_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider'

	train_json_path = os.path.join(spider_path, 'train_spider.json')
	dev_json_path = os.path.join(spider_path, 'dev.json')
	train_out_path = os.path.join(spider_path, 'my', 'train')
	dev_out_path = os.path.join(spider_path, 'my', 'dev')
	separate_files(train_json_path, train_out_path)
	separate_files(dev_json_path, dev_out_path)





