import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def get_keys_to_submodule(model):
    keys_to_submodule = {}
    # iterate all submodules
    for submodule_name, submodule in model.named_modules():
        # iterate all paramters in each submobule
        for param_name, param in submodule.named_parameters():
            # param_name is organized as <name>.<subname>.<subsubname> ...
            # the more we go deep in the model, the less "subname"s we have
            splitted_param_name = param_name.split('.')
            # if we have only one subname, then it means that we reach a "leaf" submodule,
            # we cannot go inside it anymore. This is the actual parameter
            is_leaf_param = len(splitted_param_name) == 1
            if is_leaf_param:
                # we recreate the correct key
                key = f"{submodule_name}.{param_name}"
                # we associate this key with this submodule
                keys_to_submodule[key] = submodule

    return keys_to_submodule

def load_state_dict_with_low_memory(model, state_dict, orig_device):
    # free up memory by placing the model in the `meta` device
    logger.info('load state dict with low memory')
    keys_to_submodule = get_keys_to_submodule(model)
    for key, submodule in keys_to_submodule.items():
        val = state_dict.pop(key)
        # get the valye from the state_dict
        # we need to substitute the parameter inside submodule,
        # remember key is composed of <name>.<subname>.<subsubname>
        # the actual submodule's parameter is stored inside the
        # last subname. If key is `in_proj.weight`, the correct field if `weight`
        param_name = key.split('.')[-1]
        orig_para = getattr(submodule, param_name)
        param_dtype = orig_para.dtype
        val = val.to(param_dtype)
        # create a new parameter
        orig_para.data[:] = val.data[:]
        #new_val = torch.nn.Parameter(val, requires_grad=orig_para.requires_grad)
        #setattr(submodule, param_name, new_val)

    logger.info('load state dict with low memory done')


def all_gather(t):
    result = [torch.zeros_like(t) for i in range(dist.get_world_size())]
    dist.all_gather(result, t)
    return result

def evenly_divisible_all_gather(data: torch.Tensor):
    """
    Utility function for distributed data parallel to pad tensor to make it evenly divisible for all_gather.
    Args:
        data: source tensor to pad and execute all_gather in distributed data parallel.

    """
    if dist.get_world_size() <= 1:
        return data
    # make sure the data is evenly-divisible on multi-GPUs
    length = data.shape[0]
    t_length = torch.tensor(length).to(data.device)
    all_lens = all_gather(t_length)
    max_len = max(all_lens).item()
    if length < max_len:
        size = [max_len - length] + list(data.shape[1:])
        data = torch.cat([data, data.new_full(size, 0)], dim=0)
    # all gather across all processes
    data = all_gather(data)
    data = torch.cat([data[i][:l, ...] for i, l in enumerate(all_lens)], axis=0)
    return data

def tensor2device(tensor, device):
    if isinstance(tensor, list):
        return [tensor2device(t, device) for t in tensor]
    elif torch.is_tensor(tensor):
        return tensor.to(device)
    else:
        return tensor


def batch2device(batch, device):
    for k, v in batch.items():
        batch[k] = tensor2device(v, device)


def sequence_mask(lengths, maxlen=None, dtype=torch.bool, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    if device is None:
        device = lengths.device
    row_vector = torch.arange(0, maxlen, 1, device=device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    return mask.to(dtype)


def get_linear_decay_schedule_with_warmup(optimizer, n_warmup_step, n_decay_step, decay_ratio=0.1, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < n_warmup_step:
            ratio = float(current_step) / float(max(1, n_warmup_step))
        elif current_step<n_decay_step:
            progress = float(min(current_step, n_decay_step) - n_warmup_step) / (n_decay_step - n_warmup_step)

            ratio = 1-(1-decay_ratio)*progress
        else:
            ratio = decay_ratio
        return ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mesh_reduce_losses(losses):
    reduce_losses = {}
    for k in losses[0]:
        reduce_losses[k] = np.mean([loss[k] for loss in losses])
    return reduce_losses


def mesh_reduce_preds(preds):
    new_preds = {}
    for k in preds[0]:
        if isinstance(preds[0][k], list):
            new_preds[k] = [v for pred in preds for v in pred[k]]
        else:
            new_preds[k] = np.concat([pred[k] for pred in preds], 0)
    return new_preds


def mesh_reduce_outputs(outputs):
    new_outputs = {}
    for k in outputs[0]:
        if isinstance(outputs[0][k], list):
            new_outputs[k] = [v for output in outputs for v in output[k]]
        else:
            new_outputs[k] = torch.cat([output[k] for output in outputs], 0)
    return new_outputs
