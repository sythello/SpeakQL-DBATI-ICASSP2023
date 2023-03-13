import json
import os, sys
import argparse
from sys import modules
import _jsonnet
from tqdm import tqdm
import spacy
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import random
import importlib
from copy import deepcopy
import editdistance

# sys.path.append(os.path.abspath('/Users/mac/Desktop/syt/Deep-Learning/Repos/rat-sql/third_party/wikisql'))
# sys.path.append(os.path.abspath('/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL'))

from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry

import torch

from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation
from SpeakQL.Allennlp_models.utils import misc_utils
from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
    Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS

# importlib.reload(misc_utils)
# from SpeakQL.Allennlp_models.utils.misc_utils import EvaluateSQL, EvaluateSQL_full, \
#     Postprocess_rewrite_seq, Postprocess_rewrite_seq_freeze_POS, Postprocess_rewrite_seq_modify_POS

nlp = spacy.load('en_core_web_sm')

def Load_Rat_sql(root_dir,
                 exp_config_path,
                 model_dir,
                 checkpoint_step=40000):

    exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
    
    model_config_path = os.path.join(root_dir, exp_config["model_config"])
    model_config_args = exp_config.get("model_config_args")
    
    infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

    inferer = Inferer(infer_config)
    inferer.device = torch.device("cpu")
    model = inferer.load_model(model_dir, checkpoint_step)
    dataset = registry.construct('dataset', inferer.config['data']['val'])

    for _, schema in dataset.schemas.items():
        model.preproc.enc_preproc._preprocess_schema(schema)
    
    _ret_dict = {
        'model': model,
        'dataset': dataset,
        'inferer': inferer,
    }
    
    return _ret_dict
    

def Question(q, db_id, model_dict):
    model = model_dict['model']
    dataset = model_dict['dataset']
    inferer = model_dict['inferer']
    
    spider_schema = dataset.schemas[db_id]
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=spider_schema,
        orig_schema=spider_schema.orig,
        orig={"question": q}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)


# def _Postprocess_rewrite_seq_wrapper(cand_dict, pred_dict):
#     _tags = pred_dict['rewriter_tags']
#     _rewrite_seq = pred_dict['rewrite_seq_prediction']
#     _question_toks = cand_dict['question_toks']
#     return Postprocess_rewrite_seq(_tags, _rewrite_seq, _question_toks)


def Full_evaluate(eval_version,
                  pred_dataset_path,
                  model_dict,
                  pred_key="question",
                  pred_toks_key="question_toks",
                  test_output_path=None,
                  result_output_path=None):
    
    '''
    eval_version: simply for printing results 
    pred_key: the dict key in prediction file dicts for the actual sequence prediction

    Example paths:
    pred_dataset_path = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SpeakQL/SpeakQL/Allennlp_models/outputs/test-reranker|rewriter-{}.json'.format(VERSION)
    
    '''
    
    VERSION = eval_version
    
    with open(pred_dataset_path, 'r') as f:
        # rewriter_preds = [json.loads(l) for l in f.readlines()]
        test_dataset = json.load(f)     ## including predictions 
    # with open(test_dataset_path, 'r') as f:
    #     test_dataset = json.load(f)
    # with open(orig_dev_path, 'r') as f:
    #     orig_dev_dataset = json.load(f)
        
    # Quick evaluation: only using the 1st ASR candidate

    ref_list = []
    hyp_list = []
    wer_numer = 0
    wer_denom = 0
    
    pred_idx = 0

    for d in tqdm(test_dataset):
        if len(d) == 0:
            continue

        c = d[0]

        _db_id = c['db_id']
        _rewritten_question = c[pred_key]

        if _rewritten_question == '':
            print(f'_rewritten_question is empty')
            _pred_sql = ''
            _gold_sql = c['query']
            _exact = _score = _exec = 0
        else:
            _pred_sql = Question(_rewritten_question, _db_id, model_dict=model_dict)[0]['inferred_code']
            _gold_sql = c['query']
            _exact, _score, _exec = EvaluateSQL(_pred_sql, _gold_sql, _db_id)

        c['rewritten_question'] = _rewritten_question
        c['pred_sql'] = _pred_sql
        c['score'] = _score
        c['exact'] = _exact
        c['exec'] = _exec
        
        # For BLEU 
        _rewritten_question_toks = [_t.lower() for _t in c[pred_toks_key]]
        # _question_toks = [_t.lower() for _t in c['question_toks']]
        _gold_question_toks = [_t.lower() for _t in c['gold_question_toks']]

        ref_list.append([_gold_question_toks])
        hyp_list.append(_rewritten_question_toks)
        wer_numer += editdistance.eval(_gold_question_toks, _rewritten_question_toks)
        wer_denom += len(_gold_question_toks)

        pred_idx += len(d)

    # Only using the 1st candidate to rewrite 
    _avg_1st = sum([d[0]['score'] for d in test_dataset]) / len(test_dataset)
    _avg_exact_1st = sum([d[0]['exact'] for d in test_dataset]) / len(test_dataset)
    _avg_exec_1st = sum([d[0]['exec'] for d in test_dataset]) / len(test_dataset)

    ## Std-dev (1st cand only)
    _std_1st = np.std([d[0]['score'] for d in test_dataset])
    
    ## BLEU 
    _bleu = corpus_bleu(list_of_references=ref_list,
                        hypotheses=hyp_list)

    _wer = 1.0 * wer_numer / (wer_denom + 1e-9)

    print('='*20, f'VERSION: {VERSION}', '='*20)
    print('avg_exact = {:.4f}'.format(_avg_exact_1st))
    # print('avg = {:.4f} (std = {:.4f})'.format(_avg_1st, _std_1st))
    print('avg = {:.4f}'.format(_avg_1st))
    print('avg_exec = {:.4f}'.format(_avg_exec_1st))
    print(f'BLEU = {_bleu:.4f}')
    print(f'WER = {_wer:.4f}')
    print('='*55)
    
    if test_output_path is not None:
        with open(test_output_path, 'w') as f:
            json.dump(test_dataset, f, indent=2)

    if result_output_path is not None:
        _res_d = {
            "avg_exact": _avg_exact_1st,
            "avg": _avg_1st,
            "avg_exec": _avg_exec_1st,
            "BLEU": _bleu,
            "WER": _wer,
        }
        with open(result_output_path, 'w') as f:
            json.dump(_res_d, f, indent=2)



def main(args):
    rat_sql_model_dict = Load_Rat_sql(root_dir=args.root_dir,
                                  exp_config_path=args.exp_config_path,
                                  model_dir=args.model_dir,
                                  checkpoint_step=args.checkpoint_step)

    Full_evaluate(eval_version=args.eval_version,
                  pred_dataset_path=args.pred_dataset_path,
                  # test_dataset_path=args.test_dataset_path,
                  # orig_dev_path=args.orig_dev_path,
                  model_dict=rat_sql_model_dict,
                  pred_key=args.pred_key,
                  pred_toks_key=args.pred_toks_key,
                  test_output_path=args.test_output_path,
                  result_output_path=args.result_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', '--root_dir', type=str, required=True)
    parser.add_argument('-exp', '--exp_config_path', type=str, required=True)
    parser.add_argument('-model', '--model_dir', type=str, required=True)
    parser.add_argument('-step', '--checkpoint_step', type=int, default=40000)

    parser.add_argument('-ver', '--eval_version', type=str, required=True)
    parser.add_argument('-pred', '--pred_dataset_path', type=str, required=True)
    # parser.add_argument('-test', '--test_dataset_path', type=str, required=True)
    # parser.add_argument('-odev', '--orig_dev_path', type=str, required=True)
    parser.add_argument('-test_out', '--test_output_path', type=str, required=True)
    parser.add_argument('-res_out', '--result_output_path', type=str, required=True)

    parser.add_argument('-pred_key', '--pred_key', type=str, default="question")
    parser.add_argument('-pred_toks_key', '--pred_toks_key', type=str, default="question_toks")

    args = parser.parse_args()

    main(args)




