from typing import Iterator, List, Tuple, Dict, Optional

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import numpy as np

from SpeakQL.Allennlp_models.utils.schema_gnn import spider_utils
from SpeakQL.Allennlp_models.utils.schema_gnn.spider_utils import Table, TableColumn, read_dataset_schema
import table_bert

import os
import sqlite3
from collections import OrderedDict, defaultdict
import json
import re
import pandas as pd
import random

import nltk
from nltk.corpus import stopwords

import torch
from fairseq.models.wav2vec import Wav2VecModel

from ipapy.arpabetmapper import ARPABETMapper
from arpabetandipaconvertor.phoneticarphabet2arpabet import PhoneticAlphabet2ARPAbetConvertor


AUDIO_DIM = 136
AUDIO_DIM_NO_POOLING = 68
AUDIO_DIM_WAV2VEC_PROJ = 64

# Some DBs have problems, do not use prototype values, simply use 1st values
ERR_DBS = ['formula_1', 'scholar', 'store_1']


def extractAudioFeatures(audio_fname, span_ranges, window=0.050, step=0.025):
    [freq, speech_audio] = audioBasicIO.read_audio_file(audio_fname)
    
    feats_list = []
    for st_str, ed_str in span_ranges:
        st_secs = float(st_str)
        ed_secs = float(ed_str)
        if st_secs == ed_secs == 0:
            feats_list.append(np.zeros(AUDIO_DIM_POOLING)) ## MFCC(34) * delta(2) * pooling(2)
            continue
            
        st = int(st_secs * freq)
        ed = int(ed_secs * freq)
        audio_span = speech_audio[st : ed]

        w = min(window * freq, ed - st)
        s = step * freq
        feats, f_names = ShortTermFeatures.feature_extraction(audio_span, freq, w, s, deltas=True)
        feats_max = np.max(feats, axis=-1)
        feats_avg = np.mean(feats, axis=-1)
        feats_vec = np.concatenate([feats_max, feats_avg])
        
        feats_list.append(feats_vec)
    
    feats_arr = np.vstack(feats_list)
    return feats_arr

def extractAudioFeatures_NoPooling(audio_fname, span_ranges, window=0.050, step=0.025):
    [freq, speech_audio] = audioBasicIO.read_audio_file(audio_fname)
    
    feats_list = []
    for st_str, ed_str in span_ranges:
        st_secs = float(st_str)
        ed_secs = float(ed_str)
        if st_secs == ed_secs == 0:
            feats_list.append(np.zeros((1, AUDIO_DIM_NO_POOLING))) ## MFCC(34) * delta(2)
            continue
            
        st = int(st_secs * freq)
        ed = int(ed_secs * freq)
        audio_span = speech_audio[st : ed]

        w = min(window * freq, ed - st)
        s = step * freq
        # pyAudioAnaylsis bug workaround
        if s > w*2:
            feats_list.append(np.zeros((1, AUDIO_DIM_NO_POOLING))) ## MFCC(34) * delta(2)
            continue

        feats, f_names = ShortTermFeatures.feature_extraction(audio_span, freq, w, s, deltas=True)

        feats_list.append(feats.T)
    
    return feats_list


def extractAudioFeatures_NoPooling_Wav2vec(audio_fname, span_ranges, w2v_model, out_dim=AUDIO_DIM_WAV2VEC_PROJ):
    [freq, speech_audio] = audioBasicIO.read_audio_file(audio_fname)
    
    feats_list = []
    for st_str, ed_str in span_ranges:
        st_secs = float(st_str)
        ed_secs = float(ed_str)
        if st_secs == ed_secs == 0:
            feats_list.append(np.zeros((1, AUDIO_DIM_WAV2VEC_PROJ)))
            continue
            
        st = int(st_secs * freq)
        ed = int(ed_secs * freq)
        audio_span = speech_audio[st : ed]

        # audio_tensor: (1, raw_audio_len)
        audio_tensor = torch.tensor(audio_span, dtype=torch.float32).view(1, -1)
        # z: (1, w2v_out_dim=512, out_len)
        z = w2v_model.feature_extractor(audio_tensor)
        # c: (1, w2v_out_dim=512, out_len)
        c = w2v_model.feature_aggregator(z)
        # c_array: (out_len, w2v_out_dim=512)
        c_array = c.detach().cpu().squeeze(0).transpose(0, 1).numpy()
        
        # c_array: (out_len, out_dim=64)
        c_array = c_array[:, :out_dim]

        feats_list.append(c_array)
    
    return feats_list


def extractRawAudios(audio_fname, span_ranges):
    [freq, speech_audio] = audioBasicIO.read_audio_file(audio_fname)
    # In raw audio stream, each timestep feature is an int scalar
    
    feats_list = []
    for st_str, ed_str in span_ranges:
        st_secs = float(st_str)
        ed_secs = float(ed_str)
        if st_secs == ed_secs == 0:
            feats_list.append(np.zeros((1, 1), dtype=int))
            continue
            
        st = int(st_secs * freq)
        ed = int(ed_secs * freq)
        audio_span = speech_audio[st : ed].reshape(-1, 1)   # np.array, shape = (audio_len, 1)

        feats_list.append(audio_span)
    
    # print([_a.shape for _a in feats_list])
    return feats_list


def dbToTokens(db_schema: Dict[str, Table]):
    db_tokens = []
    for table_name, table in sorted(db_schema.items()):
        db_tokens.append(table.text)
        db_tokens.append(':')
        for column in table.columns:
            db_tokens.append(column.text)
            db_tokens.append(',')
        db_tokens[-1] = '.'
    return ' '.join(db_tokens).split(' ')

## Legacy
def dbToTokensWithColumnIndexes(db_schema: Dict[str, Table]):
    db_tokens = []
    db_column_ids = []
    column_id = 1
    for table_name, table in sorted(db_schema.items()):
        db_tokens.append(table.text)
        db_tokens.append(':')
        db_column_ids.extend([0] * len(table.text.split(' ')) + [0])
        for column in table.columns:
            db_tokens.append(column.text)
            db_tokens.append(',')
            db_column_ids.extend([column_id] * len(column.text.split(' ')) + [0])
            column_id += 1
        db_tokens[-1] = '.'

    db_tokens_split = ' '.join(db_tokens).split(' ')
    assert len(db_tokens_split) == len(db_column_ids), f'{db_tokens_split}, {db_column_ids}'
    return db_tokens_split, db_column_ids

def dbToTokensWithAddCells(db_schema: Dict[str, Table],
    add_cells: Dict[Tuple, List[str]] = None,
    cells_in_bracket: bool = False):
    '''
    db_schema: the DB schema read by Spider utils. Dict[str:table_name, Table]
    add_cells: the cell values to add. Dict[Tuple:(table, column), List[str:cells]]
    cells_in_bracket:
        If True, do: "TABLE1 : COL1 ( VAL1a , VAL1b ) ; COL2 ; COL3 ( VAL3a ) . TABLE2 : ..."
        If False, do: "TABLE1 : COL1 : VAL1a , VAL1b ; COL2 ; COL3 : VAL3a . TABLE2 : ..."

    Currently not changing case, respect input case
    '''
    db_tokens = []
    db_column_ids = []
    column_id = 1
    for table_name, table in sorted(db_schema.items()):
        table_text_toks = table.text.split(' ')
        # db_tokens.append(table.text)
        db_tokens.extend(table_text_toks)
        db_tokens.append(':')
        db_column_ids.extend([0] * len(table_text_toks) + [0])
        for column in table.columns:
            column_tokens = [column.text]
            if (table_name, column.name) in add_cells:
                _cells_to_add = add_cells[(table_name, column.name)]
                if len(_cells_to_add) == 0:
                    pass
                elif cells_in_bracket:
                    column_tokens.append('(')
                    for cell in add_cells[(table_name, column.name)]:
                        column_tokens.append(cell.strip())
                        column_tokens.append(',')
                    column_tokens[-1] = ')'
                else:
                    column_tokens.append(':')
                    for cell in add_cells[(table_name, column.name)]:
                        column_tokens.append(cell.strip())
                        column_tokens.append(',')
                    column_tokens.pop()  ## remove the trailing ':' or ','
            # column_tokens = [t for t in ' '.join(column_tokens).split(' ') if len(t) > 0]
            column_tokens = [t for t in re.split(r'\s+', ' '.join(column_tokens)) if len(t) > 0]
            db_tokens.extend(column_tokens)
            db_tokens.append(';')
            db_column_ids.extend([column_id] * len(column_tokens) + [0])
            column_id += 1
        db_tokens[-1] = '.'

    db_tokens_split = ' '.join(db_tokens).split(' ')
    assert len(db_tokens_split) == len(db_column_ids), f'{db_tokens_split}, {db_column_ids}'
    return db_tokens_split, db_column_ids


def dbToTokens_new(db_schema: Dict[str, Table]):
    # Return:
    #   db_tokens_split: a list of tokens representing the DB, in the form: Q [SEP] t1 : c11 , c12 ; t2 ...
    #   db_column_ids: a list with the same length as tokens, saving the col_id (starting from 1; 0 is padding) corresponding to each token
    #   db_pointer_spans: a dict with key "column" and "table".
    #       db_pointer_spans["column"]: a list with the same length as columns, saving the pos spans (st, ed) for each column
    #       db_pointer_spans["table"]: (similar)
    #       Note: need to add the length of text in dataset reader

    db_tokens = []
    db_column_ids = []
    db_pointer_spans = {
        "column": [],
        "table": []
    }
    column_id = 1
    for table_name, table in sorted(db_schema.items()):
        table_name_len = len(table.text.split(' '))

        db_column_ids.extend([0] * table_name_len + [0])
        db_pointer_spans["table"].append((len(db_tokens), len(db_tokens) + table_name_len))

        db_tokens.extend(table.text.split(' '))
        db_tokens.append(':')
        
        for column in table.columns:
            column_name_len = len(column.text.split(' '))

            db_column_ids.extend([column_id] * column_name_len + [0])
            db_pointer_spans["column"].append((len(db_tokens), len(db_tokens) + column_name_len))

            db_tokens.extend(column.text.split(' '))
            db_tokens.append(',')
            
            column_id += 1
        db_tokens[-1] = '.'

    db_tokens_split = ' '.join(db_tokens).split(' ')
    assert len(db_tokens_split) == len(db_column_ids), f'{db_tokens_split}, {db_column_ids}'
    return db_tokens_split, db_column_ids, db_pointer_spans


def _get_tabert_c_type_spider(c_type: str):
    if c_type.lower() == 'number':
        tabert_c_type = 'real'
    else:
        # column_type: ['boolean', 'text', 'time', 'others']
        tabert_c_type = 'text'
    return tabert_c_type

def _get_tabert_c_type_sqlite(c_type: str):
    real_type_inits = ['real', 'double', 'float', 'numeric', 'decimal']

    is_real = False
    for _t in real_types_inits:
        if c_type.lower().startswith(_t):
            is_real = True
            break

    if is_real:
        return 'real'
    else:
        return 'text'

def _get_tabert_c_type_sqlite_legacy(c_type: str):
    if c_type.lower() == 'real':
        tabert_c_type = 'real'
    else:
        tabert_c_type = 'text' 
    return tabert_c_type


def read_DB(db_id: str,
            db_dir: str,
            db_schema: spider_utils.Table = None,
            schemas_json_path: str = None,
            prototype_dict: dict = None,
            row_limit: int = 5000,
            column_type_source: str = 'spider') -> Dict[str, table_bert.Table]:

    '''
    This is for tabert.
    Argument explanation:

    column_type_source:
        'spider': using spider schema file (spider/tables.json) to determine column types
            ['number'] -> real
            others: ['boolean', 'text', 'time', 'others'] -> text
        'sqlite': using sqlite_db column types
            startswith: ['real', 'double', 'float', 'numeric', 'decimal'] -> real
            others -> text
        'sqlite_legacy': using sqlite_db column types, as in earlier experiments, just for comparison
            'real' -> real
            others -> text

    '''
    
    assert (db_schema is not None) or (schemas_json_path is not None)

    ## Get schemas
    if db_schema is None:
        schemas = read_dataset_schema(schemas_json_path)
        db_schema = schemas[db_id]      # Dict[table_name_orig, spider_utils.Table]

    columns_name_mapping = dict()       # Dict[table_name_orig, Dict[col_name_orig, col_name_text]]
    columns_type_mapping = dict()       # Dict[table_name_orig, Dict[col_name_orig, col_type]]
    for table_name, _table in db_schema.items():
        # _table is instance of spider_utils.Table
        # It is possible that two columns in a table have the same name, because of (mismatch) errors in Spider!
        # Now avoiding crash by explicitly skipping the db_ids with error
        name_mapping = dict()
        type_mapping = dict()
        for col in _table.columns:
            name_mapping[col.name.lower()] = col.text
            type_mapping[col.name.lower()] = col.column_type

        columns_name_mapping[table_name] = name_mapping
        columns_type_mapping[table_name] = type_mapping

    # print(f'db_schema: {db_schema}')
    # print(json.dumps(columns_name_mapping, indent=4))

    ## Get cell values
    assert isinstance(row_limit, int), f'Invalid row_limit {row_limit}'

    db = os.path.join(db_dir, db_id, db_id + ".sqlite")
    try:
        conn = sqlite3.connect(db)
    except Exception as e:
        raise Exception(f"Can't connect to SQL: {e} in path {db}")
    conn.text_factory = str
    cursor = conn.cursor()

    values = dict()         # Dict[table_name, List[rows: List[cell_values]]]
    for table_name in db_schema:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {row_limit}")
            values[table_name] = cursor.fetchall()
        except:
            conn.text_factory = lambda x: str(x, 'latin1')
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {row_limit}")
            values[table_name] = cursor.fetchall()

    # column_type_determine_func_dict = {
    #     'spider': _get_tabert_c_type_spider,
    #     'sqlite': _get_tabert_c_type_sqlite,
    #     'sqlite_legacy': _get_tabert_c_type_sqlite_legacy
    # }
    # column_type_determine_func = column_type_determine_func_dict[column_type_source]

    ## Build TaBERT Tables
    db_tables = OrderedDict()

    for table_name, _table in sorted(db_schema.items()):
        # _table is instance of spider_utils.Table
        columns_info = list(cursor.execute(f'PRAGMA TABLE_INFO({table_name})'))
        column_name_mapping = columns_name_mapping[table_name]
        column_type_mapping = columns_type_mapping[table_name]
        table_value_rows = [[str(v) for v in row] for row in values[table_name]]
        table_value_columns = list(zip(*table_value_rows))

        tabert_columns = []
        for col_info in columns_info:
            # c_id is interger index; db_id is text ID (name-like)
            c_id, c_name, c_type, _, _, c_pk = col_info

            _spider_c_type = column_type_mapping[c_name.lower()]
            if column_type_source == 'spider':
                tabert_c_type = _get_tabert_c_type_spider(_spider_c_type)
            elif column_type_source == 'sqlite':
                tabert_c_type = _get_tabert_c_type_sqlite(c_type)
            elif column_type_source == 'sqlite_legacy':
                tabert_c_type = _get_tabert_c_type_sqlite_legacy(c_type)
            else:
                raise ValueError(f'Invalid column_type_source: {column_type_source}')

            column_dict = {
                'name': column_name_mapping[c_name.lower()],   # Using text column names instead of original names in db
                'type': tabert_c_type,
                'is_primary_key': bool(c_pk)
            }
            if len(table_value_rows) > 0:
                ## DEBUG
                assert tuple(table_value_columns[c_id]) == tuple([row[c_id] for row in table_value_rows]), \
                    f'Columns: {[column_name_mapping[ci[1].lower()] for ci in columns_info]}\n' + \
                    f'c_id = {c_id}, c_name = {column_dict["name"]}\n' + \
                    f'table_value_rows: {table_value_rows}\ntable_value_columns: {table_value_columns}\n' + \
                    f'table_value_columns[{c_id}] = {tuple(table_value_columns[c_id])}\n' + \
                    f'[row[{c_id}] for row in table_value_rows] = {tuple([row[c_id] for row in table_value_rows])}'

                if (db_id not in ERR_DBS) and (prototype_dict is not None):
                    # Use prototypes
                    column_name = column_dict['name']
                    proto_val = prototype_dict[db_id][table_name][column_name]
                    assert len(table_value_columns[c_id]) == row_limit or (proto_val in table_value_columns[c_id]), \
                        f'c_id = {c_id}, len(table_value_columns[c_id]) = {len(table_value_columns[c_id])}\n' + \
                        f'{db_id}.{table_name}.{column_name}: {proto_val} not in {table_value_columns[c_id]}'
                    column_dict['sample_value'] = proto_val

                    ## DEBUG
                    # print(f'{db_id}.{table_name}.{column_name}: {proto_val}')
                else:
                    column_dict['sample_value'] = str(table_value_rows[0][c_id])
            else:
                column_dict['sample_value'] = 'None'    # on k=1 (Vanilla) sample_value must be str

            tabert_columns.append(table_bert.Column(**column_dict))

        tabert_table = table_bert.Table(
            id=_table.text,                             # Using text table names instead of original names in db
            header=tabert_columns,
            data=table_value_rows
        )

        db_tables[table_name] = tabert_table

    return db_tables


def Get_align_tags(d: List[Dict], max_advantage: int = 4):
    '''
    Arguments:
        d: an example in the dataset json file, a list of cands; will be modified and returned 
        max_advantage: the max value of n in [DIFF-n]. Too large value will cause OOV error in prediction

    Return:
        Modified d, with 'align_tags' added to each cand in d (as well as 'token_to_tspans' and
        'tspan_to_token', which are intermediate results and not intended for use)
    '''
    
    timestamps = set()

    for c in d:
        for _st, _ed in c['span_ranges']:
            if _st == _ed == 0:
                continue
            timestamps.add(_st)
            timestamps.add(_ed)

    timestamps_idx2time = sorted(list(timestamps), key=lambda t : float(t))
    timestamps_time2idx = {t : idx for idx, t in enumerate(timestamps_idx2time)}
    
    ## For each c, get mapping from token_idx to tspan_ids(list), from tspan_id to token_idx 
    for c in d:
        token_idx_to_tspan_ids = []
        tspan_idx_to_token_idx = [-1] * (len(timestamps_idx2time) - 1)

        last_tstamp_idx = 0
        for _token_idx, (_st, _ed) in enumerate(c['span_ranges']):
            if _st == _ed == 0:
                # Punctuation 
                token_idx_to_tspan_ids.append([])
                continue

            while timestamps_idx2time[last_tstamp_idx] < _st:
                last_tstamp_idx += 1

            _tspan_ids = []

            while timestamps_idx2time[last_tstamp_idx] < _ed:
                _tspan_ids.append(last_tstamp_idx)
                last_tstamp_idx += 1

            token_idx_to_tspan_ids.append(_tspan_ids)
            for _tspan_idx in _tspan_ids:
                tspan_idx_to_token_idx[_tspan_idx] = _token_idx

        c['token_to_tspans'] = token_idx_to_tspan_ids
        c['tspan_to_token'] = tspan_idx_to_token_idx
    
    ## Count matches and generate align tags 
    for _i1, c1 in enumerate(d):
    
        _align_tags = [] # Align tags for each token 

        for _token1, _tspans in zip(c1['question_toks'], c1['token_to_tspans']):
            # For a token, find how many other cands have the same token 

            if len(_tspans) == 0:
                # Punctuation 
                _align_tags.append('[PUNCT]')
                continue

            _matches = 0
            for _i2, c2 in enumerate(d):
                if _i2 == _i1: continue

                for _span_i1 in _tspans:
                    _token_i2 = c2['tspan_to_token'][_span_i1]
                    if c2['question_toks'][_token_i2].lower() == _token1.lower():
                        # Match found in cand for token1, go to next cand 
                        _matches += 1
                        break

                # No match in cand for token1, do nothing 

            _mismatches = len(d) - 1 - _matches
            _matches_advantage = _matches - _mismatches
            _matches_advantage = min(max_advantage, max(-max_advantage, _matches_advantage))

            _align_tag = '[SAME]' if _mismatches == 0 else f'[DIFF{_matches_advantage:+d}]'
            _align_tags.append(_align_tag)

        c1['align_tags'] = _align_tags
    
    return d


def load_DB_content(db_id, db_dir):
    db_path = os.path.join(db_dir, f'{db_id}/{db_id}.sqlite')
    # print(db_path)

    tables_dict = dict()  # Dict[str:table_name, DataFrame:db_content]
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        p_res = cursor.fetchall()
    except:
        print(f'Loading table names failed: {db_id}')
        return tables_dict
    table_names = [t[0] for t in p_res]
    
    for table_name in table_names:
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            p_res = cursor.fetchall()
        except:
            print(f'Loading column names failed: {db_id}::{table_name}')
            continue
            
        columns = []
        for _, col_name, col_type, _, _, _ in p_res:
            col_type = col_type.lower()
            # print(col_name, col_type)
            if col_type.startswith('char') or col_type.startswith('varchar') or col_type.startswith('text'):
                columns.append(col_name)
        
        if len(columns) == 0:
            ## No text columns 
            continue
        
        try:
            _columns_str = ','.join([f'"{c}"' for c in columns])
            cursor.execute(f"SELECT {_columns_str} FROM {table_name}")
            p_res = cursor.fetchall()
        except Exception as e:
            print(f'Loading literals failed: {db_id}::{table_name}')
            print(columns)
            print(e)
            continue

        # df = pd.DataFrame(p_res, columns=columns)
        col_cells_list = zip(*p_res)
        col_cells_dict = dict()
        for col_name, col_values in zip(columns, col_cells_list):
            col_cells_dict[col_name] = list(OrderedDict.fromkeys([v for v in col_values if v is not None])) # preserve order 
        tables_dict[table_name] = col_cells_dict
    
    conn.close()
    
    ## Return: Dict[table_name, col_cells_dict:Dict[col_name, col_values:List[cells]]]
    return tables_dict

def text_cell_to_toks(val):
    _val = re.sub(r'["\']', '', val.lower())
    toks = [t for t in re.split(r'[\s\d/_\(\)\[\]\.\-\*\+:\\,&@%!?#;=\^]+', _val) if len(t) > 0]
    return toks

def collect_DB_toks_dict(tables_dict):
    '''
    tables_dict: Dict[table_name, Dict[col_name, List[cell]]]. The output of load_DB_content()

    Return:
    db_content_toks_dict = Dict[tok, List[Tuple(table_name, column_name, cell)]]  (these XX_names are IDs, not textual)
    '''

    db_content_toks_dict = defaultdict(list)
    for table_name, col_cells_dict in tables_dict.items():
        for col_name, cells in col_cells_dict.items():
            for cell in cells:
                if cell is None:
                    continue
                # db_content_cells.add(cell.lower())
                toks = text_cell_to_toks(cell)
                for t in toks:
                    db_content_toks_dict[t].append((table_name, col_name, cell))

    return db_content_toks_dict




if __name__ == '__main__':
    db_dir = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/database'
    db_id = 'flight_2'
    db_tables_dict = load_DB_content(db_id=db_id, db_dir=db_dir)
    print(db_tables_dict)
    db_content_toks_dict = collect_DB_toks_dict(db_tables_dict)
    # print(db_content_toks_dict)

    tables_json_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider/tables.json'
    databases = read_dataset_schema(tables_json_path)
    db_schema = databases[db_id]
    print('schema:')
    for table_name, table in db_schema.items():
        print(table_name, '::', [col.name for col in table.columns])

    sel_toks = random.sample([tok for tok in db_content_toks_dict.keys() if tok not in stopwords.words("english")], k=10)
    print('sel_toks:')
    print(sel_toks)
    add_cells = defaultdict(list)
    for tok in sel_toks:
        for table_name, col_name, cell in db_content_toks_dict[tok]:
            add_cells[(table_name, col_name.lower())].append(cell.lower())
    print('add_cells:')
    print(add_cells)
    db_tokens = dbToTokensWithAddCells(db_schema=db_schema, add_cells=add_cells)
    print('db_tokens:')
    print(db_tokens)










