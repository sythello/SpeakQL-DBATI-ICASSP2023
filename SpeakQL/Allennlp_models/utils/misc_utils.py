from typing import Iterator, List, Dict, Callable, Optional, cast

import json
import copy
import spacy
from tqdm.notebook import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import os, sys
import json
import sqlite3
import traceback
import argparse
import string
from tqdm.notebook import tqdm
import re
import editdistance

from phonemizer import phonemize
from ipapy.arpabetmapper import ARPABETMapper
from arpabetandipaconvertor.phoneticarphabet2arpabet import PhoneticAlphabet2ARPAbetConvertor

# from spider.process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql
from SpeakQL.Allennlp_models.utils.spider import process_sql, evaluation

import numpy as np
from copy import copy, deepcopy
from collections import defaultdict, Counter

DEFAULT_SPIDER_DIR = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider'
if not os.path.exists(DEFAULT_SPIDER_DIR):
    DEFAULT_SPIDER_DIR = '/vault/spider'
if not os.path.exists(DEFAULT_SPIDER_DIR):
    assert False, "DEFAULT_SPIDER_DIR not exists"

__tables_json = os.path.join(DEFAULT_SPIDER_DIR, 'tables.json')
__kmaps = evaluation.build_foreign_key_map_from_json(__tables_json)

def EvaluateSQL(pred_str,
                gold_str,
                db,
                kmaps=__kmaps,
                db_dir=os.path.join(DEFAULT_SPIDER_DIR, 'database'),
                verbose=True):

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    db_path = os.path.join(db_dir, db, db + ".sqlite")
    schema = process_sql.Schema(process_sql.get_schema(db_path))
    try:
        g_sql = process_sql.get_sql(schema, gold_str)  # Train #3153/18259, in 'assets_maintenance', 'ref_company_types' not found 
        p_sql = process_sql.get_sql(schema, pred_str)
    except:
        vprint('{}\n{}\n{}\nprocess_sql.get_sql() failed'.format(pred_str, gold_str, db))
        return 0, 0, 0
    
    # Rebuilding... copied from official evaluate 
    kmap = __kmaps[db]
    g_valid_col_units = evaluation.build_valid_col_units(g_sql['from']['table_units'], schema)
    g_sql = evaluation.rebuild_sql_val(g_sql)
    g_sql = evaluation.rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = evaluation.build_valid_col_units(p_sql['from']['table_units'], schema)
    p_sql = evaluation.rebuild_sql_val(p_sql)
    p_sql = evaluation.rebuild_sql_col(p_valid_col_units, p_sql, kmap)
    
    try:
        exec_match = evaluation.eval_exec_match(db_path, pred_str, gold_str, p_sql, g_sql)
    except:
        exec_match = 0

    evaluator = evaluation.Evaluator()
    exact_match = evaluator.eval_exact_match(p_sql, g_sql)   # will modify p_sql, g_sql (really?)

    partials = evaluator.partial_scores
    partial_summary_score = sum([partials[tp]['f1'] * max(partials[tp]['label_total'], partials[tp]['pred_total']) for tp in partials]) / sum([max(partials[tp]['label_total'], partials[tp]['pred_total']) for tp in partials])

    return int(exact_match), partial_summary_score, int(exec_match)

def EvaluateSQL_full(glist,
                     plist,
                     db_id_list,
                     kmaps=__kmaps,
                     db_dir=os.path.join(DEFAULT_SPIDER_DIR, 'database'),
                     etype='all'):
    
    # with open(gold) as f:
    #     glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    # with open(predict) as f:
    #     plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    
    # plist = [("select max(Share),min(Share) from performance where Type != 'terminal'", "orchestra")]
    # glist = [("SELECT max(SHARE) ,  min(SHARE) FROM performance WHERE TYPE != 'Live final'", "orchestra")]
    evaluator = evaluation.Evaluator()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0., 'partial_summary': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'f1': 0.,'acc_count':0,'rec_count':0}

    eval_err_num = 0
    for p, g, db in tqdm(zip(plist, glist, db_id_list), total=len(plist)):
        p_str = p
        g_str = g
        db_name = db
        db = os.path.join(db_dir, db, db + ".sqlite")
        schema = process_sql.Schema(process_sql.get_schema(db))
        g_sql = process_sql.get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1

        try:
            p_sql = process_sql.get_sql(schema, p_str)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
            eval_err_num += 1
            print("eval_err_num:{}".format(eval_err_num))

        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        g_valid_col_units = evaluation.build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = evaluation.rebuild_sql_val(g_sql)
        g_sql = evaluation.rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = evaluation.build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = evaluation.rebuild_sql_val(p_sql)
        p_sql = evaluation.rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        if etype in ["all", "exec"]:
            # print(db, p_str, g_str, p_sql, g_sql)
            exec_score = evaluation.eval_exec_match(db, p_str, g_str, p_sql, g_sql)
            if exec_score:
                scores[hardness]['exec'] += 1
                scores['all']['exec'] += 1

        if etype in ["all", "match"]:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            partial_scores = evaluator.partial_scores
            # if exact_score == 0:
            #     print("{} pred: {}".format(hardness,p_str))
            #     print("{} gold: {}".format(hardness,g_str))
            #     print("")
            scores[hardness]['exact'] += exact_score
            scores['all']['exact'] += exact_score
            for type_ in partial_types:
                if partial_scores[type_]['pred_total'] > 0:
                    scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores[hardness]['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores[hardness]['partial'][type_]['rec_count'] += 1
                scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                if partial_scores[type_]['pred_total'] > 0:
                    scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores['all']['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores['all']['partial'][type_]['rec_count'] += 1
                scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

            # Custom
            partial_summary_score = sum([partial_scores[tp]['f1'] * max(partial_scores[tp]['label_total'], partial_scores[tp]['pred_total']) for tp in partial_types]) / sum([max(partial_scores[tp]['label_total'], partial_scores[tp]['pred_total']) for tp in partial_types])
            scores[hardness]['partial_summary'] += partial_summary_score
            scores['all']['partial_summary'] += partial_summary_score
                
            entries.append({
                'predictSQL': p_str,
                'goldSQL': g_str,
                'hardness': hardness,
                'exact': exact_score,
                'partial': partial_scores
            })

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            scores[level]['partial_summary'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                        scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    evaluation.print_scores(scores, etype)
    print('================   PARTIAL MATCHING SUMMARY SCORE    ================')
    this_scores = [scores[level]['partial_summary'] for level in levels]
    print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("partial_summary", *this_scores))
    print()
    print("Exact:", '\t'.join([str(scores[level]['exact']) for level in levels]))
    print("Exec:", '\t'.join([str(scores[level]['exec']) for level in levels]))
    print("Partial:", '\t'.join([str(scores[level]['partial_summary']) for level in levels]))
    # return scores



# Edit distance function, (distance between transcription and gold text) can be used as supervision signal 
def EditDistance(S, T, matching_type=False, return_pairs_idx=False):
    # [i, j]: min edit distance of S[:i] vs. T[:j] 
    min_edit_dist = np.zeros((len(S) + 1, len(T) + 1), dtype=int)

    # for state [i, j], 0 = match, 1 = ignore S, 2 = ignore T, 3 = ignore both  
    backtrack = np.zeros((len(S) + 1, len(T) + 1), dtype=int)

    min_edit_dist[:, 0] = np.arange(len(S) + 1, dtype=int)
    min_edit_dist[0, :] = np.arange(len(T) + 1, dtype=int)
    backtrack[1:, 0] = 1
    backtrack[0, 1:] = 2

    for i in range(1, len(S) + 1):
        for j in range(1, len(T) + 1):
            if S[i-1].lower() == T[j-1].lower():
                min_edit_dist[i, j] = min_edit_dist[i-1, j-1]
                backtrack[i, j] = 0
            else:
                choices = [min_edit_dist[i-1, j], min_edit_dist[i, j-1]]
                if ((S[i-1] in string.punctuation) == (T[j-1] in string.punctuation)) or (not matching_type):
                    # both are punct, or both are words; can substitute 
                    # if not matching_type, everything can substitute 
                    choices.append(min_edit_dist[i-1, j-1])
                min_edit_dist[i, j] = min(*choices) + 1
                backtrack[i, j] = np.argmin(choices) + 1
                
    curr_state = (len(S), len(T))
    pairs = []
    pairs_idx = []
    while curr_state != (0, 0):
        i, j = curr_state
        action = backtrack[i, j]
        if action == 0:
            pairs.append((S[i-1], T[j-1]))
            pairs_idx.append((i-1, j-1))
            curr_state = (i-1, j-1)
        elif action == 1:
            pairs.append((S[i-1], ''))
            pairs_idx.append((i-1, -1))
            curr_state = (i-1, j)
        elif action == 2:
            pairs.append(('', T[j-1]))
            pairs_idx.append((-1, j-1))
            curr_state = (i, j-1)
        else:
            pairs.append((S[i-1], T[j-1]))
            pairs_idx.append((i-1, j-1))
            curr_state = (i-1, j-1)
    pairs.reverse()
    pairs_idx.reverse()

    ret_pairs = pairs_idx if return_pairs_idx else pairs
    
    return min_edit_dist[len(S), len(T)], ret_pairs


def alignment_seq_to_dict(seq, src_str, tgt_str):
    # Translating an alignment seq (0-0, 1-2, 2-3, ...) into dicts 
    # Pairs are sorted by src_index 

    align_items = seq.split(' ')
    src_sen = src_str.split(' ')
    tgt_sen = tgt_str.split(' ')
    
    def _span_match(src_span : tuple,
                    tgt_span : tuple):
        src_span_str = ''.join(src_sen[src_span[0] : src_span[-1] + 1])
        tgt_span_str = ''.join(tgt_sen[tgt_span[0] : tgt_span[-1] + 1])
        edist, _ = EditDistance(src_span_str, tgt_span_str)
        match = max(len(src_span_str), len(tgt_span_str)) - edist
        return match
    
    forward_dict = defaultdict(list)
    backward_dict = defaultdict(list)
    for align_item in align_items:
        i_str, j_str = align_item.split('-')
        i = int(i_str)
        j = int(j_str)
        forward_dict[i].append(j)
        backward_dict[j].append(i)
    
    # Continuous: selecting spans 
    forward_span_dict = defaultdict(tuple)  # Reconstructed every iteration 
    backward_span_dict = defaultdict(tuple) # Reconstructed every iteration
    forward_dict_2 = copy(forward_dict)     # Maintained over iterations 
    backward_dict_2 = copy(backward_dict)   # Maintained over iterations 

    # print('-init-')
    # print(forward_dict_2)
    # print(backward_dict_2)
    # print(forward_span_dict)
    # print(backward_span_dict)
    
    modified = True
    
    while modified:
        modified = False
        forward_span_dict.clear()
        backward_span_dict.clear()
        
        for i in range(len(src_sen)):
            tgt_ids = forward_dict_2[i]
            if len(tgt_ids) == 0:
                continue

            spans = []
            cur_span = []
            for j in tgt_ids:
                if len(cur_span) == 0:
                    cur_span.append(j)
                elif j == cur_span[-1] + 1:
                    cur_span.append(j)
                else:
                    spans.append(tuple(cur_span))
                    cur_span.clear()
                    cur_span.append(j)
            if len(cur_span) > 0:
                spans.append(tuple(cur_span))
                cur_span.clear()
                
            # Selecting best span 
            if len(spans) == 1:
                best_span = spans[0]
            else:
                best_span = tuple()
                max_match = -np.inf
                for span in spans:
                    match = _span_match((i,), span)
                    if match > max_match:
                        best_span = span
                        max_match = match

            forward_span_dict[i] = best_span
            forward_dict_2[i] = list(best_span)
            for j in tgt_ids:
                if j not in best_span:
                    modified = True
                    backward_dict_2[j].remove(i)

        # print('-1-')
        # print(forward_dict_2)
        # print(backward_dict_2)
        # print(forward_span_dict)
        # print(backward_span_dict)

        # For backward, use the backward_dict_2 just updated  
        for j in range(len(tgt_sen)):
            src_ids = backward_dict_2[j]
            if len(src_ids) == 0:
                continue

            spans = []
            cur_span = []
            for i in src_ids:
                if len(cur_span) == 0:
                    cur_span.append(i)
                elif i == cur_span[-1] + 1:
                    cur_span.append(i)
                else:
                    spans.append(tuple(cur_span))
                    cur_span.clear()
                    cur_span.append(i)
            if len(cur_span) > 0:
                spans.append(tuple(cur_span))
                cur_span.clear()

            # Selecting best span 
            if len(spans) == 1:
                best_span = spans[0]
            else:
                best_span = tuple()
                max_match = -np.inf
                for span in spans:
                    match = _span_match(span, (j,))
                    if match > max_match:
                        best_span = span
                        max_match = match

            backward_span_dict[j] = best_span
            backward_dict_2[j] = list(best_span)
            for i in src_ids:
                if i not in best_span:
                    modified = True
                    forward_dict_2[i].remove(j)

        # print('-2-')
        # print(forward_dict_2)
        # print(backward_dict_2)
        # print(forward_span_dict)
        # print(backward_span_dict)
    
    # Merging overlapping spans 
    src_spans = [(i,) for i in range(len(src_sen))]   # [i]: span containing i 
    tgt_spans = [(j,) for j in range(len(tgt_sen))]   # [j]: span containing j 
    
    modified = True
    
    while modified:
        modified = False
        for i in range(len(src_sen)):
            span = src_spans[i]
            span_set = set(span)
            for j in forward_dict_2[i]:
                for _i in backward_dict_2[j]:
                    # merge _i into span with i 
                    if _i == i or src_spans[_i] == span:
                        continue
                    span_set.update(src_spans[_i])
                    modified = True
            
            new_span = tuple(sorted(span_set))
            if new_span != span:
                for _i in new_span:
                    src_spans[_i] = new_span
                    
        for j in range(len(tgt_sen)):
            span = tgt_spans[j]
            span_set = set(span)
            for i in backward_dict_2[j]:
                for _j in forward_dict_2[i]:
                    # merge _j into span with j 
                    if _j == j or tgt_spans[_j] == span:
                        continue
                    span_set.update(tgt_spans[_j])
                    modified = True
            
            new_span = tuple(sorted(span_set))
            if new_span != span:
                for _j in new_span:
                    tgt_spans[_j] = new_span
    
    # Reconstructing span dict (span to span)
    span_dict = defaultdict(tuple)
    
    # for _, src_span in backward_span_dict.items():
    #     if len(src_span) == 0:
    #         continue
        
    #     tgt_span = forward_span_dict[src_span[0]]
    #     assert len(tgt_span) > 0
    #     for i in src_span:
    #         assert forward_span_dict[i] == tgt_span, '{}\n{}'.format(forward_span_dict, backward_span_dict)
        
    #     span_dict[src_span] = tgt_span

    for i in range(len(src_sen)):
        if len(forward_dict_2[i]) == 0:
            continue
            
        src_span = src_spans[i]
        
        if src_spans[i] in span_dict:
            continue
            
        tgt_span = tgt_spans[forward_dict_2[i][0]]
        span_dict[src_span] = tgt_span
    
    # Monotonic 
    last_src_span = None
    for src_span in sorted(span_dict.keys()):
        tgt_span = span_dict[src_span]
        
        if last_src_span is None:
            last_src_span = src_span
            continue
        
        last_tgt_span = span_dict[last_src_span]
        
        if tgt_span[0] > last_tgt_span[-1]:
            # no conflict 
            last_src_span = src_span
            continue
        
        # conflict 
        match = _span_match(src_span, tgt_span)
        last_match = _span_match(last_src_span, last_tgt_span)
        
        if match > last_match:
            del span_dict[last_src_span]
            last_src_span = src_span
        else:
            del span_dict[src_span]
    
    # N-gram pairs 
    pairs = []
    for src_span, tgt_span in sorted(span_dict.items()):
        src_str = ' '.join(src_sen[src_span[0] : src_span[-1] + 1])
        tgt_str = ' '.join(tgt_sen[tgt_span[0] : tgt_span[-1] + 1])
        pairs.append((src_str, tgt_str))
    
    return span_dict, pairs
    
    # TODO?: Forcing monotonic and continuous more wisely 


def Postprocess_rewrite_seq(tags, rewrite_seq, question_toks):
    _question_toks_placeholders = []

    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP'):
            _question_toks_placeholders.append(tok)
        elif (tags[i] == 'U-EDIT') or (tags[i] == 'B-EDIT'):
            _question_toks_placeholders.append('[EDIT]')
        elif (tags[i] == 'I-EDIT') or (tags[i] == 'L-EDIT') or tags[i].endswith('DEL'):
            pass
        else:
            print('Unknown tag: {}'.format(tags[i]))

    _edits = []
    _curr_edit = []
    for tok in rewrite_seq:
        if tok == '[ANS]':
            _edits.append(_curr_edit)
            _curr_edit = []
        elif tok == '@end@':  # Allennlp END_SYMBOL 
            break
        else:
            _curr_edit.append(tok)
    
    _question_toks_rewritten = []
    _edit_idx = 0
    for tok in _question_toks_placeholders:
        if tok == '[EDIT]':
            if _edit_idx >= len(_edits):
                print('--- Not enough edits ---')
                print('Tags:', tags)
                print('Edits:', _edits)
            else:
                _question_toks_rewritten.extend(_edits[_edit_idx])
            _edit_idx += 1
        else:
            _question_toks_rewritten.append(tok)

    return _question_toks_rewritten



def Postprocess_rewrite_seq_freeze_POS(tags, rewrite_seq, question_toks, freeze_POS, nlp=None):
    
    '''
    Freeze certain POS during rewriting (for multi-token DEL, separate; for multi-token EDIT,
        freeze it as long as having 1 frozen POS inside)
    '''
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
        
    _question_str = ' '.join(question_toks)
    _question_doc = nlp(_question_str)
    # _question_pos = [_t.pos_ for i, _t in enumerate(_question_doc) if not str(_t).startswith("'")]
    _question_pos = []
    _di = 0     # index to _question_doc
    ## Patches for Spacy and original tokenization mismatches
    for i, tok in enumerate(question_toks):
        if tok == "id's":
            # will become "i", "d", "'s"
            assert str(_question_doc[_di:_di + 3]) == tok, \
                (' '.join([str(t) for t in _question_doc]), _question_str)
            _question_pos.append(_question_doc[_di + 1].pos_)   # [_di+1] is "d", is NOUN, the closest
            _di += 3
        elif tok == "id":
            # will become "i", "d"
            assert str(_question_doc[_di:_di + 2]) == tok, \
                (' '.join([str(t) for t in _question_doc]), _question_str)
            _question_pos.append(_question_doc[_di + 1].pos_)   # [_di+1] is "d", is NOUN, the closest
            _di += 2
        elif "'" in tok[1:-1]:
            # "'" in middle, *sometimes* will be splitted into 2
            if str(_question_doc[_di:_di + 2]) == tok:
                # splitted
                _question_pos.append(_question_doc[_di].pos_)       # use the 1st one
                _di += 2
            elif str(_question_doc[_di]) == tok:
                # not splitted
                _question_pos.append(_question_doc[_di].pos_)
                _di += 1
        else:
            # no mismatch
            _question_pos.append(_question_doc[_di].pos_)
            _di += 1

    assert len(_question_pos) == len(question_toks), \
        (' '.join([str(t) for t in _question_doc]), _question_str)
        
    # Tags: K  E  E  K  E  D  K  E
    # List: -1 0  0  -1 1  -1 -1 2 
    tok_id2edit_id = []
    _edit_idx = -1
    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP') or tags[i].endswith('DEL'):
            tok_id2edit_id.append(-1)
        elif (tags[i] == 'U-EDIT') or (tags[i] == 'B-EDIT'):
            _edit_idx += 1
            tok_id2edit_id.append(_edit_idx)
        elif (tags[i] == 'I-EDIT') or (tags[i] == 'L-EDIT'):
            tok_id2edit_id.append(tok_id2edit_id[-1])
        else:
            print('Unknown tag: {}'.format(tags[i]))
    
    edit_id2tok_ids = defaultdict(list)
    edit_frozen = defaultdict(bool)
    for _tid, _eid in enumerate(tok_id2edit_id):
        if _eid >= 0:
            edit_id2tok_ids[_eid].append(_tid)
            if _question_pos[_tid] == freeze_POS:
                # freeze this edit
                edit_frozen[_eid] = True

    _edits = []
    _curr_edit = []
    for tok in rewrite_seq:
        if tok == '[ANS]':
            _edits.append(_curr_edit)
            _curr_edit = []
        elif tok == '@end@':  # Allennlp END_SYMBOL 
            break
        else:
            _curr_edit.append(tok)
    
    _question_toks_rewritten = []
    for i, tok in enumerate(question_toks):
        if _question_pos[i] == freeze_POS:
            # Keep regardless of tags 
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('KEEP'):
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('DEL'):
            pass
        elif tags[i].endswith('EDIT'):
            _e_idx = tok_id2edit_id[i]
            assert _e_idx >= 0, (tags, tok_id2edit_id)
            if edit_frozen[_e_idx]:
                # this edit is frozen
                _question_toks_rewritten.append(tok)
                continue
            # here: not frozen, edit as normal 
            if _e_idx >= len(_edits):
                # edit out of range, will be deleted
                continue
            if tags[i] in {'I-EDIT', 'L-EDIT'}:
                pass
            elif tags[i] in {'U-EDIT', 'B-EDIT'}:
                _question_toks_rewritten.extend(_edits[_e_idx])

    return _question_toks_rewritten

def Postprocess_rewrite_seq_modify_POS(tags, rewrite_seq, question_toks, modify_POS, nlp=None):
    
    '''
    Only modify certain POS during rewriting (for multi-token DEL, separate; for multi-token EDIT,
        modify it as long as having 1 modifying POS inside)
    '''
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
        
    _question_str = ' '.join(question_toks)
    _question_doc = nlp(_question_str)
    # _question_pos = [_t.pos_ for i, _t in enumerate(_question_doc) if not str(_t).startswith("'")]
    _question_pos = []
    _di = 0     # index to _question_doc
    ## Patches for Spacy and original tokenization mismatches
    for i, tok in enumerate(question_toks):
        if tok == "id's":
            # will become "i", "d", "'s"
            assert str(_question_doc[_di:_di + 3]) == tok, \
                (' '.join([str(t) for t in _question_doc]), _question_str)
            _question_pos.append(_question_doc[_di + 1].pos_)   # [_di+1] is "d", is NOUN, the closest
            _di += 3
        elif tok == "id":
            # will become "i", "d"
            assert str(_question_doc[_di:_di + 2]) == tok, \
                (' '.join([str(t) for t in _question_doc]), _question_str)
            _question_pos.append(_question_doc[_di + 1].pos_)   # [_di+1] is "d", is NOUN, the closest
            _di += 2
        elif "'" in tok[1:-1]:
            # "'" in middle, *sometimes* will be splitted into 2
            if str(_question_doc[_di:_di + 2]) == tok:
                # splitted
                _question_pos.append(_question_doc[_di].pos_)       # use the 1st one
                _di += 2
            elif str(_question_doc[_di]) == tok:
                # not splitted
                _question_pos.append(_question_doc[_di].pos_)
                _di += 1
        else:
            # no mismatch
            _question_pos.append(_question_doc[_di].pos_)
            _di += 1

    assert len(_question_pos) == len(question_toks), \
        (' '.join([str(t) for t in _question_doc]), _question_str)
        
    # Tags: K  E  E  K  E  D  K  E
    # List: -1 0  0  -1 1  -1 -1 2 
    tok_id2edit_id = []
    _edit_idx = -1
    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP') or tags[i].endswith('DEL'):
            tok_id2edit_id.append(-1)
        elif (tags[i] == 'U-EDIT') or (tags[i] == 'B-EDIT'):
            _edit_idx += 1
            tok_id2edit_id.append(_edit_idx)
        elif (tags[i] == 'I-EDIT') or (tags[i] == 'L-EDIT'):
            tok_id2edit_id.append(tok_id2edit_id[-1])
        else:
            print('Unknown tag: {}'.format(tags[i]))
    
    edit_id2tok_ids = defaultdict(list)
    edit_modify = defaultdict(bool)
    for _tid, _eid in enumerate(tok_id2edit_id):
        if _eid >= 0:
            edit_id2tok_ids[_eid].append(_tid)
            if _question_pos[_tid] == modify_POS:
                # freeze this edit
                edit_modify[_eid] = True

    _edits = []
    _curr_edit = []
    for tok in rewrite_seq:
        if tok == '[ANS]':
            _edits.append(_curr_edit)
            _curr_edit = []
        elif tok == '@end@':  # Allennlp END_SYMBOL 
            break
        else:
            _curr_edit.append(tok)
    
    _question_toks_rewritten = []
    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP'):
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('DEL'):
            if _question_pos[i] == modify_POS:
                # modify: delete
                pass
            else:
                # not modify: keep
                _question_toks_rewritten.append(tok)
        elif tags[i].endswith('EDIT'):
            _e_idx = tok_id2edit_id[i]
            assert _e_idx >= 0, (tags, tok_id2edit_id)
            if not edit_modify[_e_idx]:
                # this edit shouldn't be modified
                _question_toks_rewritten.append(tok)
                continue
            # here: this edit should be modified, edit as normal 
            if _e_idx >= len(_edits):
                # edit out of range, will be deleted
                continue
            if tags[i] in {'I-EDIT', 'L-EDIT'}:
                pass
            elif tags[i] in {'U-EDIT', 'B-EDIT'}:
                _question_toks_rewritten.extend(_edits[_e_idx])

    return _question_toks_rewritten



def _question_toks_POS_tagging(question_toks, nlp=None):
    '''
    Given question_toks (a list of tokens), tag their POS
    Notice: this are some hacking just for Spacy
    '''

    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
        
    _question_str = ' '.join(question_toks)
    _question_doc = nlp(_question_str)
    _question_doc_tokens = [str(t).lower() for t in _question_doc]
    # _question_pos = [_t.pos_ for i, _t in enumerate(_question_doc) if not str(_t).startswith("'")]
    _question_pos = []
    _di = 0     # index to _question_doc
    ## Patches for Spacy and original tokenization mismatches
    for i, tok in enumerate(question_toks):
        if tok.lower() == "id's":
            # will become ["i", "d", "'s"] or ["id", "'s"]
            if str(_question_doc[_di:_di + 3]) == tok:
                assert _question_doc_tokens[_di:_di + 3] == ["i", "d", "'s"]
                _question_pos.append("PROPN")
                _di += 3
            elif str(_question_doc[_di:_di + 2]) == tok:
                assert _question_doc_tokens[_di:_di + 2] == ["id", "'s"]
                _question_pos.append("PROPN")
                _di += 2
            else:
                raise ValueError(_question_doc_tokens)
                
        elif tok.lower() == "id":
            # can become ["i", "d"] or no change ("id")
            if str(_question_doc[_di:_di + 2]) == tok:
                assert _question_doc_tokens[_di:_di + 2] == ["i", "d"]
                _question_pos.append("PROPN")
                _di += 2
            elif _question_doc_tokens[_di] == "id":
                # no change
                _question_pos.append(_question_doc[_di].pos_)
                _di += 1
            else:
                raise ValueError(_question_doc_tokens)

        elif "'" in tok[1:-1]:
            # "'" in middle, *sometimes* will be splitted into 2
            if str(_question_doc[_di:_di + 2]) == tok:
                # splitted
                _question_pos.append(_question_doc[_di].pos_)       # use the 1st one
                _di += 2
            elif str(_question_doc[_di]) == tok:
                # not splitted
                _question_pos.append(_question_doc[_di].pos_)
                _di += 1
            else:
                raise ValueError(_question_doc_tokens)
        else:
            # normal, no change 
            _question_pos.append(_question_doc[_di].pos_)
            _di += 1

    # make sure we handled Spacy special tokenizations 
    assert len(_question_pos) == len(question_toks), \
        (' '.join([str(t) for t in _question_doc]), _question_str)
        
    return _question_pos


def _get_tok_id2edit_id(tags, question_toks):
    # Tags: K  E  E  K  E  D  K  E
    # List: -1 0  0  -1 1  -1 -1 2 

    tok_id2edit_id = []
    _edit_idx = -1
    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP') or tags[i].endswith('DEL'):
            tok_id2edit_id.append(-1)
        elif (tags[i] == 'U-EDIT') or (tags[i] == 'B-EDIT'):
            _edit_idx += 1
            tok_id2edit_id.append(_edit_idx)
        elif (tags[i] == 'I-EDIT') or (tags[i] == 'L-EDIT'):
            tok_id2edit_id.append(tok_id2edit_id[-1])
        else:
            print('Unknown tag: {}'.format(tags[i]))
    
    return tok_id2edit_id


def Postprocess_rewrite_seq_freeze_POS_v2(tags, rewrite_seq, question_toks, freeze_POS, nlp=None):
    
    '''
    Freeze certain POS during rewriting (for multi-token DEL, separate; for multi-token EDIT,
        freeze it as long as having 1 frozen POS inside)
    v2: re-implemented, extracting code into common functions
    '''
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
        
    _question_pos = _question_toks_POS_tagging(question_toks, nlp=nlp)


    # Tags: K  E  E  K  E  D  K  E
    # List: -1 0  0  -1 1  -1 -1 2 
    tok_id2edit_id = _get_tok_id2edit_id(tags, question_toks)

    
    edit_id2tok_ids = defaultdict(list)
    edit_frozen = defaultdict(bool)
    for _tid, _eid in enumerate(tok_id2edit_id):
        if _eid >= 0:
            edit_id2tok_ids[_eid].append(_tid)
            if _question_pos[_tid] == freeze_POS:
                # freeze this edit
                edit_frozen[_eid] = True

    _edits = []
    _curr_edit = []
    for tok in rewrite_seq:
        if tok == '[ANS]':
            _edits.append(_curr_edit)
            _curr_edit = []
        elif tok == '@end@':  # Allennlp END_SYMBOL 
            break
        else:
            _curr_edit.append(tok)
    
    _question_toks_rewritten = []
    for i, tok in enumerate(question_toks):
        if _question_pos[i] == freeze_POS:
            # Keep regardless of tags 
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('KEEP'):
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('DEL'):
            pass
        elif tags[i].endswith('EDIT'):
            _e_idx = tok_id2edit_id[i]
            assert _e_idx >= 0, (tags, tok_id2edit_id)
            if edit_frozen[_e_idx]:
                # this edit is frozen
                _question_toks_rewritten.append(tok)
                continue
            # here: not frozen, edit as normal 
            if _e_idx >= len(_edits):
                # edit out of range, will be deleted
                continue
            if tags[i] in {'I-EDIT', 'L-EDIT'}:
                pass
            elif tags[i] in {'U-EDIT', 'B-EDIT'}:
                _question_toks_rewritten.extend(_edits[_e_idx])

    return _question_toks_rewritten

def Postprocess_rewrite_seq_modify_POS_v2(tags, rewrite_seq, question_toks, modify_POS, nlp=None):
    
    '''
    Only modify certain POS during rewriting (for multi-token DEL, separate; for multi-token EDIT,
        modify it as long as having 1 modifying POS inside)
    v2: re-implemented, extracting code into common functions
    '''
    
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
        
    _question_pos = _question_toks_POS_tagging(question_toks, nlp=nlp)


    # Tags: K  E  E  K  E  D  K  E
    # List: -1 0  0  -1 1  -1 -1 2 
    tok_id2edit_id = _get_tok_id2edit_id(tags, question_toks)


    edit_id2tok_ids = defaultdict(list)
    edit_modify = defaultdict(bool)
    for _tid, _eid in enumerate(tok_id2edit_id):
        if _eid >= 0:
            edit_id2tok_ids[_eid].append(_tid)
            if _question_pos[_tid] == modify_POS:
                # freeze this edit
                edit_modify[_eid] = True

    _edits = []
    _curr_edit = []
    for tok in rewrite_seq:
        if tok == '[ANS]':
            _edits.append(_curr_edit)
            _curr_edit = []
        elif tok == '@end@':  # Allennlp END_SYMBOL 
            break
        else:
            _curr_edit.append(tok)
    
    _question_toks_rewritten = []
    for i, tok in enumerate(question_toks):
        if tags[i].endswith('KEEP'):
            _question_toks_rewritten.append(tok)
        elif tags[i].endswith('DEL'):
            if _question_pos[i] == modify_POS:
                # modify: delete
                pass
            else:
                # not modify: keep
                _question_toks_rewritten.append(tok)
        elif tags[i].endswith('EDIT'):
            _e_idx = tok_id2edit_id[i]
            assert _e_idx >= 0, (tags, tok_id2edit_id)
            if not edit_modify[_e_idx]:
                # this edit shouldn't be modified
                _question_toks_rewritten.append(tok)
                continue
            # here: this edit should be modified, edit as normal 
            if _e_idx >= len(_edits):
                # edit out of range, will be deleted
                continue
            if tags[i] in {'I-EDIT', 'L-EDIT'}:
                pass
            elif tags[i] in {'U-EDIT', 'B-EDIT'}:
                _question_toks_rewritten.extend(_edits[_e_idx])

    return _question_toks_rewritten








def strip_stress(phone: str) -> str:
    _m = re.match(r'(.*)\d+$', phone)
    if _m is not None:
        phone = _m.group(1)
    return phone

def Load_CMU_Dict(cmudict_path: str):
    ''' Load cmudict for pronunciations. Update: now using lowercase words as keys '''

    entry_lines = []

    with open(cmudict_path, 'r', encoding='latin-1') as f:
        for l in f:
            if len(l.strip()) > 0 and (not l.startswith(';;;')):
                entry_lines.append(l.strip())

    word2pron = defaultdict(set) # all possible prons, Dict[str, Set[Tuple[str]]]
    pron2word = defaultdict(set) # all possible words, Dict[Tuple[str], Set[str]]

    for l in entry_lines:
        _word, _pron = l.split('  ')
        
        if _word.endswith(')'):
            # variant
            _m = re.match(r'(.*)\((.*)\)$', _word)
            _word = _m.group(1)
            _variant = _m.group(2)
        else:
            # no variant
            _variant = None
        
        _word = _word.lower()
        _phones = tuple([strip_stress(_phone) for _phone in _pron.split(' ')])
        
        word2pron[_word].add(_phones)
        pron2word[_phones].add(_word)

    print(f'Loading CMU Dict done: {len(entry_lines)} entries, {len(word2pron)} words, {len(pron2word)} prons')

    return word2pron, pron2word

def Load_DB_Tok2phs_Dict(dict_path: str):
    ''' Load db_tok2phs.json for pronunciations '''

    word2pron = defaultdict(set) # all possible prons, Dict[str, Set[Tuple[str]]]
    pron2word = defaultdict(set) # all possible words, Dict[Tuple[str], Set[str]]

    with open(dict_path, 'r') as f:
        db_tok2phs_dict = json.load(f)

    for tok, phs in db_tok2phs_dict.items():
        _phones = tuple([strip_stress(_phone) for _phone in phs])
        
        word2pron[tok].add(_phones)
        pron2word[_phones].add(tok)

    print(f'Loading db_tok2phs Dict done: {len(db_tok2phs_dict)} entries, {len(word2pron)} words, {len(pron2word)} prons')

    return word2pron, pron2word




def WordPronDist(word2pron: dict, w1: str, w2: str):
    w1 = w1.lower()
    w2 = w2.lower()
    if w1 not in word2pron or w2 not in word2pron:
        return max(len(w1), len(w2))
    
    min_edist = max(len(w1), len(w2))
    for p1 in word2pron[w1]:
        for p2 in word2pron[w2]:
            edist = editdistance.eval(p1, p2)
            # print(p1, p2, edist)
            min_edist = min(min_edist, edist)
    
    return min_edist

def WordPronSimilarity(word2pron: dict, w1: str, w2: str):
    w1 = w1.lower()
    w2 = w2.lower()
    if w1 not in word2pron or w2 not in word2pron:
        return 0
    
    max_sim = 0
    for p1 in word2pron[w1]:
        if len(p1) == 0: continue
        for p2 in word2pron[w2]:
            if len(p2) == 0: continue
            edist = editdistance.eval(p1, p2)
            sim = 1 - float(edist) / max(len(p1), len(p2))
            # print(p1, p2, sim)
            max_sim = max(max_sim, sim)
    
    return max_sim

def ConstructWordSimMatrix(
        sen: List[str],
        word2pron: Dict[str, set],
        sim_func: Callable,
        default_val: float = 0.0,
        skip_punct: bool = True):

    # non_empty_ids = [i for i in range(len(_tokens)) if str(_tokens[i])[0] not in string.punctuation]
    # non_empty_tokens = [_tokens[i] for i in non_empty_ids]
    # non_empty_tokens

    matrix = np.ones((len(sen), len(sen))) * default_val
    for i in range(len(sen)):
        if (skip_punct and sen[i][0] in string.punctuation) or (sen[i].lower() not in word2pron):
            continue

        for j in range(len(sen)):
            if (skip_punct and sen[j][0] in string.punctuation) or (sen[j].lower() not in word2pron):
                continue

            matrix[i, j] = sim_func(word2pron, sen[i], sen[j])

    return matrix


def tokens_to_phonemes(tokens, arpa_convertor=None, a_mapper=None, do_strip_stress=True):
    ## First try arpa_converter; if not supported, fall back to a_mapper

    if arpa_convertor is None:
        arpa_convertor = PhoneticAlphabet2ARPAbetConvertor()
    if a_mapper is None:
        a_mapper = ARPABETMapper()

    # List[str:ipa_phonemes]
    ipa_phonemes_list = phonemize(
        tokens,
        language='en-us',
        backend='espeak',
        # separator=Separator(phone=None, word=' ', syllable=''),
        strip=True,
        # with_stress=True,
        preserve_punctuation=False,
        # njobs=4
    )

    arpa_phonemes_list = []
    for phs in ipa_phonemes_list:
        try:
            arpa_phs = arpa_convertor.convert(phs.replace('ː',':')).split(' ')
        except:
            arpa_phs = a_mapper.map_unicode_string(phs, ignore=True, return_as_list=True)

        if do_strip_stress:
            arpa_phs = [strip_stress(_phone) for _phone in arpa_phs]

        arpa_phonemes_list.append(arpa_phs)
    
    return arpa_phonemes_list

def _detokenize(toks):
    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(toks)




