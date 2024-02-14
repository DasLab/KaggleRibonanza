import os, logging

import numpy as np
import logging
from torch import nn

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb, \
    LlamaAttention, _make_causal_mask, _expand_mask, LlamaForSequenceClassification
from transformers.models.mistral.modeling_mistral import MistralForSequenceClassification, MistralModel
from transformers.modeling_outputs import TokenClassifierOutput
from .scheduled_dropout import ScheduledDropout

logger = logging.getLogger(__name__)

def _make_sliding_window_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = torch.full(
        (tgt_len, tgt_len),
        fill_value=1,
        device=device,
    )
    #mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    mask1 = torch.triu(tensor, diagonal=-sliding_window)
    mask2 = torch.tril(tensor, diagonal=sliding_window)
    mask = mask1*mask2
    mask = torch.log(mask).to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _prepare_decoder_slid_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
#    if input_shape[-1] > 1:
#        combined_attention_mask = _make_causal_mask(
#            input_shape,
#            inputs_embeds.dtype,
#            device=inputs_embeds.device,
#            past_key_values_length=past_key_values_length,
#        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

class TokenClassificationMix():
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        if isinstance(config, LlamaConfig):
            self.model = LlamaModel(config)
            self.model._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        else:
            self.model = MistralModel(config)
            self.model._prepare_decoder_attention_mask = _prepare_decoder_slid_attention_mask
        classifier_dropout = 0
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForTokenClassification(TokenClassificationMix, LlamaForSequenceClassification):
    pass

class MistralForTokenClassification(TokenClassificationMix, MistralForSequenceClassification):
    pass

def requires_grad(module, requires_grad):
    for layer in module.children():
        for param in layer.parameters():
            param.requires_grad = requires_grad


def cross_entropy(logits, labels, mask=None, weight=None, smooth=0):
    if smooth>0:
        loss = label_smooth_cross_entropy(logits, labels, smooth)
    else:
        dim = logits.dim()
        if dim>2:
            logits = logits.permute(0, -1, *range(1, dim - 1))
            #logits = torch.transpose(logits, 1, dim - 1)
        loss = F.cross_entropy(logits, labels, reduction='none')
    if weight is not None:
        loss = loss*weight
    if mask is not None:
        loss = torch.sum(loss*mask)/(torch.sum(mask)+1e-4)
    else:
        loss = torch.mean(loss)
    return loss

def label_smooth_cross_entropy(logits, labels, smooth, mask=None):
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(-1))
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (1-smooth) * nll_loss + smooth * smooth_loss
    return loss


def binary_cross_entropy(logits, labels, mask=None, weight=None):
    losses = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction='none')
    if weight is not None:
        losses = losses*weight
    if mask is None:
        loss = torch.mean(losses)
    else:
        loss = torch.sum(losses*mask)/(torch.sum(mask)+1e-4)
    return loss


def kl_divergence(p, q, mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    # pad_mask is for seq-level tasks
    # You can choose whether to use function "sum" and "mean" depending on your task
    loss = (p_loss + q_loss) / 2
    if mask is not None:
        loss = torch.mean(loss[mask.to(bool)])
    else:
        loss = torch.mean(loss)
    return loss


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        return getattr(nn, activation)()


def create_transformer_model(name, use_pretrain, config=None, data_dir=None, cls=AutoModel, debug=False, use_gc=False, torchscript=False, **kwargs):
    # use manually downloaded pretrained
    fdir = name
    if data_dir is not None:
        fdir = os.path.join(data_dir, name)
    if debug:
        # use auto downloaded pretrained
        fdir = name
        use_pretrain = False
    if use_pretrain:
        model = cls.from_pretrained(fdir, torchscript=torchscript, **kwargs)
    else:
        if config is None:
            config = AutoConfig.from_pretrained(fdir, torchscript=torchscript, **kwargs)
        if cls == LlamaForTokenClassification or cls == MistralForTokenClassification:
            model = cls(config=config)
        else:
            model = cls.from_config(config)
    if use_gc:
        model.gradient_checkpointing_enable()
    logger.info('%s name_or_path:%s', name, model.name_or_path)
    return model


def resize_roberta_position_embedding(old_emb, config, max_len):
    new_config = deepcopy(config)
    new_config.max_position_embeddings = max_len+2
    new_emb = RobertaEmbeddings(new_config)
    new_emb.position_embeddings.weight.data[:config.max_position_embeddings] = old_emb.position_embeddings.weight.data
    new_emb.word_embeddings = old_emb.word_embeddings
    return new_emb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, maxlen=500, hidden_dim=192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_emb = nn.Parameter(self.positional_encoding(maxlen-1, hidden_dim), requires_grad=False)
        self.div = np.sqrt(self.hidden_dim)

    def forward(self, x):
        maxlen = x.shape[1]
        return x + self.pos_emb[None, :maxlen, :]

    def positional_encoding(self, maxlen, hidden_dim):
        depth = hidden_dim/2
        positions = torch.arange(maxlen, dtype=torch.float32)[..., None]
        depths = torch.arange(depth, dtype=torch.float32)[None, :]/depth
        angle_rates = 1/torch.pow(10000.0, depths)
        angle_rads = torch.matmul(positions, angle_rates)
        pos_encoding = torch.cat( [torch.sin(angle_rads), torch.cos(angle_rads)], axis=-1)
        return pos_encoding

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act="gelu"):
        super().__init__()
        if act=='swiglu':
            dim1 = int(hidden_dim*4/3)
            dim1 = 2*(dim1//2)
            dim2 = dim1//2
        else:
            dim1, dim2 = hidden_dim, hidden_dim

        self.net = nn.Sequential(
            nn.Linear(dim, dim1),
            _get_activation_layer(act),
            nn.Dropout(dropout),
            nn.Linear(dim2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, ffd_dim, post_norm=False, act='gelu', use_inst=False, res_factor=1, dropout=0, dp_start=0, dp_end=100000000, use_rope=False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransformerEncoderLayer(d_model, n_head, ffd_dim, post_norm, act=act, use_inst=use_inst, res_factor=res_factor,
                                                  dropout=dropout, dp_start=dp_start, dp_end=dp_end, use_rope=use_rope))
        self.enc_layers = nn.ModuleList(layers)
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_emb= LlamaRotaryEmbedding(
                d_model//n_head,
                max_position_embeddings=512,
                base=10000.0,
            )
        else:
            self.pos_emb = SinusoidalPosEmb(hidden_dim=d_model)

    def forward(self, x, mask, position_ids=None):
        if not self.use_rope:
            x = self.pos_emb(x)
        elif position_ids is None:
            l = x.shape[1]
            position_ids = torch.arange(0, l, dtype=torch.long, device=x.device)[None, ...]
        for enc_layer in self.enc_layers:
            x = enc_layer(x, mask, self.pos_emb, position_ids)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_head, ffd_dim, post_norm=False, act='gelu', res_factor=1, use_inst=False, dropout=0, dp_start=0, dp_end=1000000000, use_rope=False):
        super().__init__()
        self.attn = MultiHeadAttention( embed_dim=embed_dim, num_head=num_head, dropout=0, dp_start=dp_start, dp_end=dp_end, use_rope=use_rope)
        #self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_head, batch_first=True)
        self.ffn = FeedForward(embed_dim, ffd_dim, act=act)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.post_norm = post_norm
        self.res_factor = res_factor
        self.dropout1 = ScheduledDropout(dp_start, dp_end, dropout, use_inst=use_inst)
        self.dropout2 = ScheduledDropout(dp_start, dp_end, dropout, use_inst=use_inst)

    def forward(self, x, mask=None, pos_emb=None, position_ids=None):
        if self.post_norm:
            x = self.norm1(x+self.dropout1(self.attn(x, mask, pos_emb, position_ids))*self.res_factor)
            x = self.norm2(x+self.dropout2(self.ffn(x))*self.res_factor)
        else:
            norm_x = self.norm1(x)
            x = x + self.dropout1(self.attn(norm_x, mask, pos_emb, position_ids))*self.res_factor
            x = x + self.dropout2(self.ffn((self.norm2(x))))*self.res_factor
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 dropout=0,
                 dp_start=0,
                 dp_end=1000000000,
                 use_rope=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.use_rope = use_rope
        self.qk_dim = embed_dim//num_head
        self.v_dim = embed_dim//num_head

        self.q = nn.Linear(embed_dim, self.qk_dim * num_head)
        self.k = nn.Linear(embed_dim, self.qk_dim * num_head)
        self.v = nn.Linear(embed_dim, self.v_dim * num_head)

        self.out = nn.Linear(self.v_dim * num_head, embed_dim)
        self.scale = 1 / (self.qk_dim ** 0.5)

    def forward(self, x, mask=None, pos_emb=None, position_ids=None):
        B, L, dim = x.shape
        # out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        num_head = self.num_head
        qk_dim = self.qk_dim
        v_dim = self.v_dim

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(B, L, num_head, qk_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, L, num_head, qk_dim).permute(0, 2, 3, 1).contiguous()
        v = v.reshape(B, L, num_head, v_dim).permute(0, 2, 1, 3).contiguous()
        if self.use_rope:
            cos, sin = pos_emb(v, seq_len=L)
            k = k.permute(0, 1, 3, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            k = k.permute(0, 1, 3, 2).contiguous()

        dot = torch.matmul(q, k) * self.scale  # H L L
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L).expand(-1, num_head, L, -1)
            dot.masked_fill_(mask, -1e4)
        attn = F.softmax(dot, -1)  # L L

        v = torch.matmul(attn, v)  # L H dim
        v = v.permute(0, 2, 1, 3).reshape(B, L, v_dim * num_head).contiguous()
        out = self.out(v)
        return out

class GNNConv(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, kernel=3):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=(kernel-1)//2)
        self.act = nn.LeakyReLU()
        self.dropout = ScheduledDropout(self.cfg.dp_start, self.cfg.dp_end, self.cfg.dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.norm(x + self.dropout(self.act(self.conv(x.transpose(1,2)).transpose(1, 2))))
        return x

class ATTGNNLayer(nn.Module):
    def __init__(self, cfg, d_model):
        super().__init__()
        self.cfg = cfg
        self.n_adj = self.cfg.n_adj
        self.d_model = d_model
        self.dropout = ScheduledDropout(self.cfg.dp_start, self.cfg.dp_end, self.cfg.dropout)
        self.norm = nn.LayerNorm(d_model)
        self.conv_adj = nn.ModuleList([
            GNNConv(cfg, d_model, d_model) for i in range(self.cfg.n_adj)

        ])
        self.conv = nn.Sequential(
            nn.Conv1d(d_model*self.cfg.n_adj, d_model, kernel_size=5, padding=2),
            nn.LeakyReLU()
        )

    def forward(self, x, adj_all):
        x_as = []
        for i in range(self.cfg.n_adj):
            x_a = self.conv_adj[i](x)
            x_a = torch.matmul(adj_all[:, :, :, i], x_a)
            x_as.append(x_a)
        x_as = torch.cat(x_as, axis=-1)
        x = self.norm(x+ self.dropout(self.conv(x_as.transpose(1,2)).transpose(1,2)))
        return x


class ATTGNN(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        self.cfg = cfg
        if self.cfg.not_use_mfe:
            self.proj = nn.Linear(hidden_size, self.cfg.gnn_dim)
        else:
            self.proj = nn.Linear(hidden_size + self.cfg.node_dim, self.cfg.gnn_dim)
        self.adj_proj = nn.Sequential(
            nn.Linear(self.cfg.n_adj-1, 1),
            nn.ReLU(),
        )
        self.gnn_layers = nn.ModuleList([ATTGNNLayer(self.cfg, self.cfg.gnn_dim) for i in range(self.cfg.gnn_layer)])
        self.node_convs = nn.ModuleList(
            [
                GNNConv(cfg, self.cfg.gnn_dim, self.cfg.gnn_dim, kernel=5) for i in range(self.cfg.gnn_layer)
            ]
        )

    def forward(self, x, node_feature, adj, mask):
        if self.cfg.use_learn:
            adj = torch.cat([adj, self.adj_proj(adj)], axis=-1)
        if self.cfg.not_use_mfe:
            x = self.proj(x)
        else:
            x = self.proj(torch.cat([x, node_feature], axis=-1))
        mask = mask[:, :, None].to(x.dtype)
        outputs = []
        for i, gnn in enumerate(self.gnn_layers):
            x = x * mask
            x = gnn(x, adj)
            outputs.append(x)

        x = torch.cat(outputs, axis=-1)
        return x




class GNNRES(nn.Module):
    def __init__(self, dim, kernel_size=3, dropout=0):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.norm = nn.LayerNorm(dim)
        self.act =  nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + self.dropout(self.act(x)))
        return x


class GNNConvV2(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, dropout=0):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()
        self.res = GNNRES(out_dim)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(self.dropout(self.norm(x)))
        x = self.res(x)
        return x


class GNNLayer(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList([
            GNNConvV2(in_dim, out_dim) for i in range(self.cfg.n_adj)
        ])
        self.in_conv = GNNConvV2(in_dim, out_dim, kernel_size=31)
        self.out_conv = GNNConvV2(out_dim*(self.cfg.n_adj+1), out_dim)
        config = LlamaConfig()
        config.hidden_size = out_dim
        config.num_attention_heads = self.cfg.n_head
        config.num_key_value_heads = self.cfg.n_head
        config.max_position_embeddings = 512
        config.rope_theta = 10000.0
        self.att = LlamaAttention(config)

    def forward(self, x, adj_all, att_mask, position_ids):
        xs = [self.in_conv(x)]
        for i in range(self.cfg.n_adj):
            x_a = self.convs[i](x)
            x_a = torch.matmul(adj_all[:, :, :, i], x_a)
            xs.append(x_a)
        xs = torch.cat(xs, axis=-1)
        x = self.out_conv(xs)
        x = self.att(x, att_mask, position_ids)[0]
        return x


class GNNModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = self.cfg.node_dim
        out_dim = self.cfg.gnn_dim
        out_dims = []
        node_convs = []
        ks = 3
        for i in range(self.cfg.gnn_node_layer):
            out_dims.append(out_dim)
            node_convs.append(GNNConvV2(in_dim, out_dim, kernel_size=ks, dropout=self.cfg.dropout))
            ks = 2*ks+1
            in_dim = out_dim
            out_dim = out_dim//2
        self.out_dim = sum(out_dims)
        self.node_convs = nn.ModuleList(node_convs)
        gnn_layers = []
        in_dim = self.out_dim
        out_dim = self.cfg.gnn_dim
        for i in range(self.cfg.gnn_layer):
            gnn_layers.append(GNNLayer(cfg, in_dim=in_dim, out_dim=out_dim))
            in_dim = out_dim
        self.gnn_layers = nn.ModuleList(gnn_layers)

    def forward(self, x, adj, mask):
        device = x.device
        position_ids = torch.arange(x.shape[1], dtype=torch.long, device=device )
        position_ids = position_ids.unsqueeze(0)
        xs = []
        for conv in self.node_convs:
            x = conv(x)
            xs.append(x)
        x = torch.cat(xs, axis=-1)
        xs = []
        combined_attention_mask = _make_causal_mask(mask.shape, x.dtype, device=x.device, past_key_values_length=0)
        expanded_attn_mask = _expand_mask(mask, x.dtype, tgt_len=x.shape[1]).to(x.device)
        combined_attention_mask = expanded_attn_mask + combined_attention_mask
        mask = mask[:, :, None].to(x.dtype)
        for gnn_layer in self.gnn_layers:
            x = x * mask
            x = gnn_layer(x, adj, combined_attention_mask, position_ids)
            xs.append(x)
        return torch.cat(xs, axis=-1)



class RNA_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(len(self.cfg.oids), self.cfg.d_model)
        self.transformer = TransformerEncoder(num_layers=self.cfg.n_layer, d_model=self.cfg.d_model, n_head=self.cfg.n_head,
                                              ffd_dim=self.cfg.d_model*4, post_norm=self.cfg.post_norm, act=self.cfg.activation,
                                              dropout=self.cfg.dropout, use_inst=self.cfg.use_inst, use_rope=self.cfg.use_rope)
        self.dropout = ScheduledDropout(self.cfg.dp_start, self.cfg.dp_end, self.cfg.output_dp, use_inst=False)
        self.proj_out = nn.Linear(self.cfg.d_model, self.cfg.n_label)

    def get_emb(self, input_ids, mask):
        x = self.emb(input_ids)
        return x

    def forward(self, x, mask):
        x = self.transformer(x, ~mask.to(bool))
        x = self.proj_out(self.dropout(x))
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, dims, activations):
        super().__init__()
        layers = []
        for dim, activation in zip(dims, activations):
            layers.append(nn.Linear(input_dim, dim))
            if activation is not None:
                layers.append(_get_activation_layer(activation))
            input_dim = dim
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_num_of_paras(m):
    num1, num2 = 0, 0
    for p in m.parameters():
        if p.requires_grad:
            num1 += p.numel()
        else:
            num2 += p.numel()
    return num1/1000/1000, num2/1000/1000

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

def get_peft(model, cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
                             inference_mode=False,
                             r=cfg.lora_rank,
                             lora_alpha=cfg.lora_alpha,
                             lora_dropout=0.1,
                             target_modules=[
                                 name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                             ])

    model = get_peft_model(model, peft_config)
    return model


class Module(nn.Module):
    def __init__(self, cfg, config=None, **kwargs):
        super(Module, self).__init__(**kwargs)
        self.config = config
        self.cfg = cfg
        self.create_layers()
        self.last_batchs = []
        self.last_outputs = []
        self.training_step = 0

    def create_layers(self):
        self.output_dropout = ScheduledDropout(self.cfg.dp_start, self.cfg.dp_end, self.cfg.output_dp, use_inst=False)
        self.create_backbone()
        if self.cfg.use_mfe:
            self.proj_mfe = nn.Linear(self.hidden_size, 3)
        if self.cfg.use_lt:
            self.proj_lt = nn.Linear(self.hidden_size, 7)

    def get_encoder(self, model, backbone):
        if 'roberta' in backbone:
            return model.roberta
        elif 'roformer' in backbone:
            return model.roformer
        elif 'deberta' in backbone:
            return model.deberta
        elif 'longformer' in backbone:
            return model.longformer
        elif "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" == backbone:
            return model.roberta
        elif 'sentence-transformers' in backbone:
            return model.bert
        elif 'llama' in backbone.lower():
            return model.model
        elif 'Mistral' in backbone:
            return model.model
        else:
            return model.encoder

    def create_pooler(self):
        if self.cfg.avg_pool:
            pooler = MeanPooling()
        else:
            raise NotImplementedError(self.cfg.avg_pool)
        return pooler

    def resize_token_embeddings(self, model, new_num_tokens, token_ids):
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        assert new_num_tokens==len(token_ids)
        model_embeds = self._resize_token_embeddings(model, new_num_tokens, token_ids)

        # Update base model and current model config
        model.config.vocab_size = new_num_tokens
        model.vocab_size = new_num_tokens

        # Tie weights again if needed
        model.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, model, new_num_tokens, token_ids):
        old_embeddings = model.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(model, old_embeddings, new_num_tokens, token_ids)
        model.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if model.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = model.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(model, old_lm_head, new_num_tokens, token_ids)
            model.set_output_embeddings(new_lm_head)

        return model.get_input_embeddings()

    def _get_resized_embeddings(self, model, old_embeddings: nn.Embedding, new_num_tokens, token_ids) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        model._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[token_ids]
        return new_embeddings

    def _get_resized_lm_head(self, model, old_lm_head, new_num_tokens, token_ids, transposed=False):
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        old_num_tokens, old_lm_head_dim = ( old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size() )

        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None
        new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head = new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)

        # initialize new lm head (in particular added tokens)
        model._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # XXX: put the long block of code in a wrapper
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[token_ids, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, token_ids]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[token_ids]
        return new_lm_head


    def calc_sd_loss(self, inputs, outputs):
        last_batch = self.last_batchs[0]
        last_logits = self.last_outputs[0]
        logits = self.forward(**last_batch)['logits']
        b, l, d = logits.shape
        label_mask, weights = last_batch['label_mask'], last_batch['weight']
        label_mask = label_mask.to(bool)
        weights = weights[:, None, :].repeat(1, l, 1)
        logits, labels, weights = logits[label_mask], last_logits[label_mask], weights[label_mask]
        loss = F.l1_loss(logits, labels, reduction='none')
        loss = torch.mean(loss * weights)

        self.last_batchs = self.last_batchs[1:] + [inputs]
        self.last_outputs = self.last_outputs[1:] + [outputs['logits'].detach()]
        return loss

    def calc_rloss(self, inputs, outputs, losses):
        logits, mask = outputs['logits'], inputs['mask']
        outputs = self.forward(**inputs)
        rlogits = outputs['logits']
        losses2 = self._calc_loss(inputs, outputs)
        rloss = kl_divergence(logits, rlogits, mask)
        losses['loss'] = (losses['loss'] + losses2['loss']) / 2 + self.cfg.rdrop * rloss
        losses['r'] = rloss

    def calc_loss(self, inputs, outputs):
        if self.training:
            self.training_step += 1
        losses = self._calc_loss(inputs, outputs)
        if self.training:
            if self.cfg.adv_lr>0:
                abs = self.cfg.accumulated_batch_size*self.cfg.adv_step
            else:
                abs = self.cfg.accumulated_batch_size
            if self.cfg.sd_weight>0 and len(self.last_batchs)>=abs:
                sd_loss = self.calc_sd_loss(inputs, outputs)
                losses['loss'] = losses['loss'] + self.cfg.sd_weight*sd_loss
                losses['sd'] = sd_loss
            elif self.cfg.sd_weight>0:
                self.last_batchs.append(inputs)
                self.last_outputs.append(outputs['logits'].detach())
        if self.cfg.rdrop>0 and self.training and self.training_step>=self.cfg.dp_start:
            self.calc_rloss(inputs, outputs, losses)
        return losses

    def create_tfm(self):
        self.backbone = RNA_Model(self.cfg)
        self.hidden_size = self.cfg.d_model

    def create_backbone(self, cls=None):
        if self.cfg.use_tfm:
            return self.create_tfm()
        if cls is None:
            cls = AutoModelForTokenClassification
        if 'llama' in self.cfg.backbone.lower():
            cls = LlamaForTokenClassification
        kwargs = dict(num_labels=self.cfg.n_label, trust_remote_code=True)
        if self.cfg.debug:
            kwargs['hidden_size'] = 8
            kwargs['embedding_size'] = 8
            kwargs['intermediate_size'] = 8
            kwargs['num_attention_heads'] = 2
            kwargs['num_hidden_layers'] = 2
            kwargs['num_key_value_heads'] = 2
        elif not self.cfg.use_pretrain:
            kwargs['hidden_size'] = self.cfg.d_model
            kwargs['intermediate_size'] = 4*self.cfg.d_model
            kwargs['num_attention_heads'] = self.cfg.n_head
            kwargs['num_hidden_layers'] = self.cfg.n_layer
            kwargs['num_key_value_heads'] = self.cfg.n_head
        if 'Mistral' in self.cfg.backbone:
            cls = MistralForTokenClassification
            if self.cfg.debug:
                kwargs['sliding_window'] = 2
            else:
                kwargs['sliding_window'] = 128
        model = create_transformer_model(self.cfg.backbone, use_pretrain=self.cfg.use_pretrain and not self.cfg.debug, config=self.config, data_dir=self.cfg.hfd,
                                                 debug=self.cfg.debug, cls=cls,
                                                 use_gc=self.cfg.use_gc, torchscript=self.cfg.torchscript, **kwargs)
        self.backbone = model
        self.backbone.dropout = self.output_dropout
        self.config = model.config
        self.hidden_size = self.config.hidden_size
        if self.config.vocab_size!=len(self.cfg.oids):
            self.resize_token_embeddings(model, new_num_tokens=len(self.cfg.oids), token_ids=self.cfg.oids)
        if self.cfg.use_peft:
            self.backbone = get_peft(self.backbone, self.cfg)


class Pretrain(Module):
    def create_backbone(self, cls=None):
        super().create_backbone(cls=AutoModelForMaskedLM)

    def calc_loss(self, inputs, outputs):
        logits, labels, masked_indices = outputs.pop('logits'), inputs['mlm_label'], inputs['masked_indices']
        logits, labels = logits.flatten(0, 1)[masked_indices], labels.flatten()[masked_indices]
        smooth = self.cfg.label_smooth if self.training else 0
        loss = cross_entropy(logits, labels.to(torch.long), smooth=smooth)
        losses = {'loss': loss}
        return losses

    def forward(self, input_ids, mask, **kwargs):
        logits = self.backbone(input_ids, mask).logits
        return {'logits': logits}


class SRF(Module):
    def _calc_loss(self, inputs, outputs):
        logits, labels, label_mask, weights = outputs['logits'], inputs['label'], inputs['label_mask'], inputs['weight']
        b,l,d = logits.shape
        weights = weights[:, None, :].repeat(1, l, 1)
        label_mask = label_mask.to(bool)
        logits, labels, weights = logits[label_mask], labels[label_mask], weights[label_mask]
        loss = F.l1_loss(logits, labels, reduction='none')
        loss = torch.mean(loss*weights)
        losses = {'loss': loss}
        if self.cfg.use_mfe and self.training:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        return losses

    def get_emb(self, input_ids, mask=None, reads=None, snr=None, **kwargs):
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        if 'llama' in self.cfg.backbone.lower() or 'Mistral' in self.cfg.backbone:
            word_emb = encoder.embed_tokens(input_ids)
        else:
            word_emb = encoder.embeddings.word_embeddings(input_ids)
        if self.cfg.use_ext:
            word_emb = word_emb + self.snf_weight*self.snf_emb(torch.cat([reads, snr], axis=-1))
        if 'llama' in self.cfg.backbone.lower() or 'Mistral' in self.cfg.backbone:
            emb = word_emb
        elif 'roformer' in self.cfg.backbone:
            emb = encoder.embeddings(inputs_embeds=word_emb)
        else:
            emb = encoder.embeddings(inputs_embeds=word_emb, mask=mask)
        return emb

    def forward(self, input_ids, mask, **kwargs):
        if self.cfg.use_tfm:
            emb = self.backbone.get_emb(input_ids, mask)
            logits = self.backbone(emb, mask)
        else:
            embs = self.get_emb(input_ids, mask, **kwargs)
            outputs = self.backbone(attention_mask=mask, inputs_embeds=embs, output_hidden_states=True)
            logits, hidden_states = outputs.logits, outputs.hidden_states[-1]
        if not self.cfg.no_act:
            logits = torch.sigmoid(logits)
        if not self.training:
            logits = torch.clip(logits, 0, 1)
        outputs = dict(logits=logits)
        if self.cfg.use_mfe and self.training:
            bt_logits = self.proj_mfe(hidden_states)
            outputs['bt_logits'] = bt_logits
        return outputs

class SRF1P(SRF):
    def create_layers(self):
        super().create_layers()
        self.exp_emb = nn.Embedding(2, self.hidden_size)
        self.exp_emb.weight.data[:] = 0

    def get_emb(self, input_ids, mask=None, exp_id=None, **kwargs):
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        word_emb = encoder.embeddings.word_embeddings(input_ids)
        word_emb = word_emb + self.exp_emb(exp_id)[:, None, :]
        emb = encoder.embeddings(inputs_embeds=word_emb)
        return emb

class SRFRNN(SRF):
    def create_layers(self):
        super().create_layers()
        if self.cfg.rnn=='lstm':
            self.rnn = nn.LSTM(self.config.hidden_size, self.cfg.rnn_dim, num_layers=self.cfg.rnn_layer, batch_first=True, bidirectional=True)
        else:
            raise NotImplementedError(self.cfg.rnn)
        self.proj = nn.Linear(self.cfg.rnn_dim*2, self.cfg.n_label)

    def forward_rnn(self, hidden_states, input_len):
        hidden_states = nn.utils.rnn.pack_padded_sequence(hidden_states, input_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(hidden_states)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def forward(self, input_ids, mask, input_len, **kwargs):
        embs = self.get_emb(input_ids, mask, **kwargs)
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        hidden_states = encoder(attention_mask=mask, inputs_embeds=embs)[0]
        hidden_states = self.forward_rnn(hidden_states, input_len)

        logits = self.proj(hidden_states)
        logits = torch.sigmoid(logits)
        outputs = dict(logits=logits)
        return outputs


class SRFGNN(SRF):
    def create_backbone(self):
        self.backbone = GNNModule(self.cfg)
        self.hidden_size = self.cfg.gnn_dim
        self.output_layer = GNNConvV2(self.cfg.gnn_dim*self.cfg.gnn_layer, self.cfg.gnn_dim, dropout=self.cfg.output_dp)
        self.proj = nn.Linear(self.cfg.gnn_dim, self.cfg.n_label)
        self.ae_proj = nn.Linear(self.cfg.gnn_dim, self.cfg.node_dim)

    def forward(self, node_feature, adj, mask, **kwargs):
        hidden_states = self.backbone(node_feature, adj, mask)
        hidden_states = self.output_layer(hidden_states)
        logits = self.proj(hidden_states)
        if not self.cfg.no_act:
            logits = torch.sigmoid(logits)
        if not self.training:
            logits = torch.clip(logits, 0, 1)
        outputs = dict(logits=logits)
        return outputs

class PretrainAESRFGNN(SRFGNN):
    def create_layers(self):
        super().create_layers()
        self.ae_proj = nn.Linear(self.hidden_size, self.cfg.node_dim)
        #self.input_dropout = ScheduledDropout(self.cfg.dp_start, self.cfg.dp_end, self.cfg.input_dp, use_spatial=True)

    def mask_node(self, node_feature):
        b, l, d = node_feature.shape
        rd = torch.rand(b, 1, d, device=node_feature.device, dtype=node_feature.dtype)
        mask = (rd<self.cfg.mlm_node).repeat(1, l, 1)
        new_node_feature = torch.zeros_like(node_feature)
        new_node_feature[~mask] = node_feature[~mask]
        return new_node_feature, mask

    def _calc_loss(self, inputs, outputs):
        logits, labels, masks, node_mask = outputs['logits'], inputs['node_feature'], inputs['mask'], outputs['node_mask']
        masks = torch.logical_and(masks.to(bool)[:, :, None], node_mask)
        logits, labels = logits[masks], labels[masks]
        loss = binary_cross_entropy(logits, labels.to(torch.int64))
        losses = {'loss': loss}
        return losses

    def forward(self, node_feature, adj, mask, **kwargs):
        node_feature, node_mask = self.mask_node(node_feature)
        hidden_states = self.backbone(node_feature, adj, mask)
        hidden_states = self.output_layer(hidden_states)
        logits = self.ae_proj(hidden_states)
        outputs = dict(logits=logits, node_mask=node_mask)
        return outputs


class PretrainSRFGNN(SRFGNN):
    def create_layers(self):
        super().create_layers()
        self.proj1 = nn.Linear(self.hidden_size, self.cfg.enc_dim)
        self.proj2 = nn.Linear(self.hidden_size, self.cfg.enc_dim)

    def _calc_loss(self, inputs, outputs):
        logits, labels, masks = outputs['logits'], inputs['bpp'], inputs['mask']
        b, l, _  = logits.shape
        labels, indices = torch.max(labels, axis=-1)
        label_mask = torch.logical_and(labels>self.cfg.bpp_thr, masks)
        logits, labels = logits[label_mask], indices[label_mask]
        loss = cross_entropy(logits, labels)
        losses = {'loss': loss}
        if self.cfg.use_mfe:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        if self.cfg.use_lt:
            logits, labels, mask = outputs['lt_logits'], inputs['lt'], inputs['mask']
            lt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + lt_loss
            losses['lt'] = lt_loss
        return losses

    def forward(self, node_feature, adj, mask, **kwargs):
        hidden_states = self.backbone(node_feature, adj, mask)
        hidden_states = self.output_layer(hidden_states)
        enc1 = self.proj1(hidden_states)
        enc2 = self.proj2(hidden_states)
        logits = torch.einsum("btd,bld->btl", enc1, enc2)
        outputs = dict(logits=logits)
        if self.cfg.use_mfe:
            bt_logits = self.proj_mfe(hidden_states)
            outputs['bt_logits'] = bt_logits
        if self.cfg.use_lt:
            lt_logits = self.proj_lt(hidden_states)
            outputs['lt_logits'] = lt_logits
        return outputs


class SRFBPP(SRF):
    def create_layers(self):
        super().create_layers()
        #self.proj = nn.Linear(2*self.config.hidden_size, self.cfg.n_label)
        self.bpp_emb = nn.Parameter(torch.from_numpy(np.zeros([1, 1, self.hidden_size]).astype(np.float32)))
        self.lt_emb = nn.Embedding(7, self.hidden_size)

    def get_emb(self, input_ids, mask=None, reads=None, snr=None, bpp=None, lt=None, base_type=None, **kwargs):
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        if 'llama' in self.cfg.backbone.lower() or 'Mistral' in self.cfg.backbone:
            word_emb = encoder.embed_tokens(input_ids)
        else:
            word_emb = encoder.embeddings.word_embeddings(input_ids)
        if bpp is not None and not self.cfg.no_emb_bpp:
            bpp = torch.max(bpp, axis=-1)[0]
            word_emb = word_emb + self.bpp_emb*bpp[..., None]
        if lt is not None:
            word_emb = word_emb + self.lt_emb(lt)
        if self.cfg.use_ext:
            word_emb = word_emb + self.snf_weight*self.snf_emb(torch.cat([reads, snr], axis=-1))
        if 'llama' in self.cfg.backbone.lower() or 'Mistral' in self.cfg.backbone:
            emb = word_emb
        elif 'roformer' in self.cfg.backbone:
            emb = encoder.embeddings(inputs_embeds=word_emb)
        else:
            emb = encoder.embeddings(inputs_embeds=word_emb, mask=mask)
        return emb

    def get_bpp_feature(self, hidden_states, bpp, mask):
        weights = torch.softmax(bpp/self.cfg.temp, axis=-1)
        bpp = torch.matmul(weights, hidden_states)
        features = torch.cat([hidden_states, bpp], axis=-1)
        features = self.backbone.dropout(features)
        return features

    def forward(self, input_ids, mask, bpp=None, base_type=None, pos=None, **kwargs):
        if self.cfg.use_tfm:
            emb = self.backbone.get_emb(input_ids, mask)
            bpp = torch.max(bpp, axis=-1)[0]
            emb = emb + self.bpp_emb * bpp[..., None]
            logits = self.backbone(emb, mask)
        else:
            embs = self.get_emb(input_ids, mask, bpp=bpp, base_type=base_type, **kwargs)
            #if pos is not None:
            #    pos = make_log_bucket_position(pos, self.config.position_buckets, self.config.max_position_embeddings)
            #outputs = self.backbone(attention_mask=mask, inputs_embeds=embs, position_ids=pos, output_hidden_states=True)
            outputs = self.backbone(attention_mask=mask, inputs_embeds=embs, output_hidden_states=True)
            logits, hidden_states = outputs.logits, outputs.hidden_states[-1]
        if not self.cfg.no_act:
            logits = torch.sigmoid(logits)
        if not self.training:
            logits = torch.clip(logits, 0, 1)
        outputs = dict(logits=logits)
        if self.cfg.use_mfe and self.training:
            bt_logits = self.proj_mfe(hidden_states)
            outputs['bt_logits'] = bt_logits
        return outputs


class SRFBPPGNN(SRFBPP):
    def create_layers(self):
        super().create_layers()
        self.proj = nn.Linear(self.cfg.gnn_dim*self.cfg.gnn_layer, self.cfg.n_label)
        self.backbone.classifier = nn.Identity()
        self.backbone.dropout = nn.Identity()

        self.gnn = ATTGNN(self.cfg,  self.hidden_size)

    def forward(self, input_ids, mask, adj, node_feature, bpp=None, base_type=None, pos=None, **kwargs):
        if self.cfg.use_tfm:
            emb = self.backbone.get_emb(input_ids, mask)
            bpp = torch.max(bpp, axis=-1)[0]
            emb = emb + self.bpp_emb * bpp[..., None]
            hidden_states = self.backbone(emb, mask)
        else:
            embs = self.get_emb(input_ids, mask, bpp=bpp, base_type=base_type, **kwargs)
            hidden_states = self.backbone(attention_mask=mask, inputs_embeds=embs).logits
        b, l, d = hidden_states.shape
        h_gnn = self.gnn(hidden_states, node_feature, adj, mask)
        logits = self.proj(self.output_dropout(h_gnn))
        if not self.cfg.no_act:
            logits = torch.sigmoid(logits)
        if not self.training:
            logits = torch.clip(logits, 0, 1)
        outputs = dict(logits=logits)
        return outputs

class PretrainSRFBPPGNN(SRFBPPGNN):
    def create_layers(self):
        super().create_layers()
        self.proj1 = nn.Linear(self.hidden_size, self.cfg.enc_dim)
        self.proj2 = nn.Linear(self.hidden_size, self.cfg.enc_dim)

    def forward(self, input_ids, mask, adj, node_feature, bpp, base_type=None, pos=None, **kwargs):
        embs = self.get_emb(input_ids, mask, bpp=None, base_type=base_type, **kwargs)
        if pos is not None:
            pos = make_log_bucket_position(pos, self.config.position_buckets, self.config.max_position_embeddings)
        hidden_states = self.backbone(attention_mask=mask, inputs_embeds=embs, position_ids=pos).logits

        enc1 = self.proj1(hidden_states)
        enc2 = self.proj2(hidden_states)
        logits = torch.einsum("btd,bld->btl", enc1, enc2)
        outputs = dict(logits=logits)
        if self.cfg.use_mfe:
            bt_logits = self.proj_mfe(hidden_states)
            outputs['bt_logits'] = bt_logits
        if self.cfg.use_lt:
            lt_logits = self.proj_lt(hidden_states)
            outputs['lt_logits'] = lt_logits
        return outputs

    def _calc_loss(self, inputs, outputs):
        logits, labels, masks = outputs['logits'], inputs['bpp'], inputs['mask']
        b, l, _  = logits.shape
        labels, indices = torch.max(labels, axis=-1)
        label_mask = torch.logical_and(labels>self.cfg.bpp_thr, masks)
        logits, labels = logits[label_mask], indices[label_mask]
        loss = cross_entropy(logits, labels)
        losses = {'loss': loss}
        if self.cfg.use_mfe:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        if self.cfg.use_lt:
            logits, labels, mask = outputs['lt_logits'], inputs['lt'], inputs['mask']
            lt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + lt_loss
            losses['lt'] = lt_loss
        return losses

class SRFBPP1P(SRFBPP):
    def create_layers(self):
        super(SRFBPP, self).create_layers()
        self.bpp_emb = nn.Linear(self.cfg.bpp_topk, self.hidden_size)

    def get_emb(self, input_ids, mask=None, reads=None, snr=None, bpp=None, **kwargs):
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        word_emb = encoder.embeddings.word_embeddings(input_ids)
        bpp, inds = torch.topk(bpp, self.cfg.bpp_topk, axis=-1)
        word_emb = word_emb + self.bpp_emb(bpp)
        if self.cfg.use_ext:
            word_emb = word_emb + self.snf_weight*self.snf_emb(torch.cat([reads, snr], axis=-1))
        emb = encoder.embeddings(inputs_embeds=word_emb, mask=mask)
        return emb

class PretrainBPP(SRF):
    def create_layers(self):
        super().create_layers()
        self.proj1 = nn.Linear(self.hidden_size, self.cfg.enc_dim)
        self.proj2 = nn.Linear(self.hidden_size, self.cfg.enc_dim)

    def _calc_loss(self, inputs, outputs):
        logits, labels, label_mask, weights = outputs['logits'], inputs['bpp'], inputs['label_mask'], inputs['weight']
        b, l, _  = logits.shape
        label_mask = (torch.max(label_mask, axis=-1)[0])
        #label_mask = (label_mask[:, :, None]*label_mask[:, None, :]*(1-torch.eye(l, l).to(label_mask.device)[None, ...])).to(bool)
        #label_mask = torch.logical_and(label_mask, labels>self.cfg.bpp_thr)
        labels, indices = torch.max(labels, axis=-1)
        label_mask = torch.logical_and(label_mask, labels>self.cfg.bpp_thr)
        logits, labels = logits[label_mask], indices[label_mask]
        loss = cross_entropy(logits, labels)
        losses = dict(loss=loss)
        if self.cfg.use_mfe:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        if self.cfg.use_lt:
            logits, labels, mask = outputs['lt_logits'], inputs['lt'], inputs['mask']
            lt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + lt_loss
            losses['lt'] = lt_loss
        return losses

    def forward(self, input_ids, mask, **kwargs):
        embs = self.get_emb(input_ids, mask, **kwargs)
        encoder = self.get_encoder(self.backbone, self.cfg.backbone)
        hidden_states = encoder(attention_mask=mask, inputs_embeds=embs)[0]
        enc1 = self.proj1(hidden_states)
        enc2 = self.proj2(hidden_states)
        logits = torch.einsum("btd,bld->btl", enc1, enc2)
        outputs = dict(logits=logits)
        if self.cfg.use_mfe:
            bt_logits = self.proj_mfe(hidden_states)
            outputs['bt_logits'] = bt_logits
        if self.cfg.use_lt:
            lt_logits = self.proj_lt(hidden_states)
            outputs['lt_logits'] = lt_logits
        return outputs

class PretrainBPP1P(PretrainBPP):
    def _calc_loss(self, inputs, outputs):
        logits, labels, masks, label_mask = outputs['logits'], inputs['bpp'], inputs['mask'], inputs['label_mask']
        b, l, _  = logits.shape
        masks = torch.logical_and(masks, label_mask)
        labels, indices = torch.max(labels, axis=-1)
        label_mask = torch.logical_and(labels>self.cfg.bpp_thr, masks)
        logits, labels = logits[label_mask], indices[label_mask]
        loss = cross_entropy(logits, labels)
        losses = {'loss': loss}
        if self.cfg.use_mfe:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        if self.cfg.use_lt:
            logits, labels, mask = outputs['lt_logits'], inputs['lt'], inputs['mask']
            lt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + lt_loss
            losses['lt'] = lt_loss
        return losses


class PretrainMFE(PretrainBPP1P):
    def _calc_loss(self, inputs, outputs):
        logits, labels, inds, masks = outputs['logits'], inputs['bp_label'], inputs['bp_ind'], inputs['mask']
        logits = logits.flatten(0, 1)[inds]
        loss = cross_entropy(logits, labels)
        losses = {'loss': loss}
        if self.cfg.use_mfe:
            logits, labels, mask = outputs['bt_logits'], inputs['base_type'], inputs['mask']
            bt_loss = cross_entropy(logits, labels, mask)
            losses['loss'] = losses['loss'] + bt_loss
            losses['bt'] = bt_loss
        return losses
