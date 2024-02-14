import pandas as pd

def format_input(input: str | list[str] | pd.DataFrame):
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
