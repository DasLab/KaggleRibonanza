import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def batch_to_device(batch, device):   
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

class TestDataset(Dataset):
    def __init__(self, df):
        self.seq_map = {'A':1,'C':2,'G':3,'U':4}
        df['L'] = df.sequence.apply(len)
        self.Lmax = 206
        self.df = df
        self.padding_idx = 0   
                
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, raw_seq = self.df.loc[idx, ['id_min','id_max','sequence']]
        ids = np.arange(id_min,id_max+1)

        Lmax = len(raw_seq)
        tokens = torch.tensor([self.seq_map[s] for s in raw_seq])        
        tokens = torch.nn.functional.pad(tokens, (0, Lmax - len(tokens)), 'constant', value=self.padding_idx)
        
        ids = np.pad(ids,(0,Lmax-len(tokens)), constant_values=-1)
       
        mask_tokens = torch.ne(tokens, self.padding_idx).int()
        mask_pair_1d = torch.ne(tokens, self.padding_idx).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)


        return {
            'tokens':tokens,
            'mask_tokens': mask_tokens,
            'mask_pair_tokens':mask_pair_2d,
            }, {'ids':ids}
        
from .twintower_lib.twintower import TwinTower
    
class Net(nn.Module):
    def __init__(self,cfg):
        super(Net, self).__init__()
        self.twintower = TwinTower(cfg)        
             
    def forward(self, batch):        
        x = self.twintower(tokens=batch['tokens'].unsqueeze(1),
                            pair_tokens=batch['tokens'],
                            mask_tokens = batch['mask_tokens'].unsqueeze(1),
                            mask_pair_tokens = batch['mask_pair_tokens']
                            )     
                 
        
        output_dict = x['chem']
        #output_dict['conf'] = compute_plddt_score(x['plddt']) / 100

        
        return output_dict 

@torch.jit.ignore
def wrapper(batch):
    tokens=batch['tokens'].unsqueeze(1)
    pair_tokens=batch['pair_tokens']
    mask_tokens = batch['mask_tokens'].unsqueeze(1)
    mask_pair_tokens = batch['mask_pair_tokens']
    
    return tokens, pair_tokens, mask_tokens, mask_pair_tokens

def collate_fn(batch):
    input = dict()
    ids = dict()
    tokens_list, mask_tokens_list, mask_pair_list, ids_list = [], [], [], []
    x, y = zip(*batch)
    batch_max_len = max([len(x['tokens']) for x in x])
    
    for data in x:
        tokens = torch.nn.functional.pad(data['tokens'], (0, batch_max_len - len(data['tokens'])), 'constant', value=0)

        mask_tokens_list.append(torch.ne(tokens, 0).int())
        tokens_list.append(tokens)
        
        mask_pair_1d = torch.ne(tokens, 0).int().unsqueeze(0)
        mask_pair_2d = mask_pair_1d * mask_pair_1d.permute(1,0)
        mask_pair_list.append(mask_pair_2d)
  
    input['tokens'] = torch.stack(tokens_list)
    input['mask_tokens'] = torch.stack(mask_tokens_list)
    input['mask_pair_tokens'] = torch.stack(mask_pair_list)
    
    for id in y:
        id_tensor = np.pad(id['ids'],(0, batch_max_len-len(id['ids'])), constant_values=-1)
        ids_list.append(torch.tensor(id_tensor))
        
    ids['ids'] = torch.stack(ids_list)
    
    return input, ids
