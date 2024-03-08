from typing import Union
import re
import numpy as np
import pandas as pd

def format_input(input: Union[str, list[str], pd.DataFrame]):
    if isinstance(input, pd.DataFrame):
        df = input.copy(deep=False)
    elif isinstance(input, list):
        df = pd.DataFrame({ 'sequence': input })
    else:
        df = pd.DataFrame({ 'sequence': [input] })
    
    if not 'sequence_id' in df:
        df['sequence_id'] = [f's{i}' for i in range(len(df))]
    
    if not 'id_min' in df or not 'id_max' in df:
        L = df['sequence'].apply(len)
        df['id_max'] = L.expanding(1).sum().sub(1).astype(int)
        df['id_min'] = df['id_max'].sub(L.sub(1)).astype(int)

    return df

def read_fasta(file):
    df = pd.DataFrame(columns=['sequence', 'sequence_id'])
    if isinstance(file, str):
        f = open(file, 'r')
    else:
        f = file
    
    for line in f:
        if line.startswith(('>', ';', '#')):
            if len(df) == 0 or df.loc[len(df) -1, 'sequence'] != '':
                df.loc[len(df)] = {
                    'sequence_id': re.sub(r'^(>|;|#)', '', line).strip().split(' ')[0],
                    'sequence': ''
                }
        elif line.strip() != '':
            df.loc[len(df) - 1, 'sequence'] += line.strip()
    
    if isinstance(file, str):
        f.close()

    return df

def format_output(input: pd.DataFrame, inference: pd.DataFrame):
    for _, seq in input.iterrows():
        inference.loc[seq.id_min:seq.id_max + 1, 'sequence_id'] = seq['sequence_id']
    inference['index_in_sequence'] = inference.groupby('sequence_id')['id'].rank().astype(int)
    # Pandas requires float32 instead of float16 to be able to perform some operations
    inference['reactivity_DMS_MaP'] = inference['reactivity_DMS_MaP'].astype(np.float32)
    inference['reactivity_2A3_MaP'] = inference['reactivity_2A3_MaP'].astype(np.float32)
    # Reorder
    inference.insert(1, 'sequence_id', inference.pop('sequence_id'))
    inference.insert(2, 'index_in_sequence', inference.pop('index_in_sequence'))
