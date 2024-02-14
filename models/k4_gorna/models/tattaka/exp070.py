import argparse
import math
from typing import Callable, Optional, Union
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from timm.layers import RmsNorm
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "070"
COMMENT = """
    kfold, postnorm, high weight decay, long warmup, low s/n threshold, 
    conv transformer, SHAPE positional encoding, bpps bias, efficient impl, 
    param tuning from exp034, swiGLU, split attention ALiBi and bpps, 
    fixed 0-1 clipping, B2T connection option, low grad clipping, add norm and act for conv1d
    with pseudo_label, RMSNorm
    """

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202
# https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py#L57-L64


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# https://github.com/jaketae/alibi/blob/main/alibi/attention.py#L10-L22
def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    pos = x - y
    return (pos > 0) * -pos + (pos < 0) * pos


# https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742-L752
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


class BiasedConvTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        kernel_size,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        b2t_connection: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.b2t_connection = b2t_connection
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=kernel_size)
        self.conv1d_t = nn.ConvTranspose1d(d_model, d_model, kernel_size=kernel_size)
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.conv2d_t = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=kernel_size),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)  # for swiGLU
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm3 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm4 = RmsNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        src: torch.Tensor,
        bias: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x_att, bias_att = self._sa_block(
                self.norm1(x), bias, src_mask, src_key_padding_mask
            )
            x = x + x_att
            x = x + self._ff_block(self.norm2(x))
            bias = bias + bias_att
        else:
            x_att, bias_att = self._sa_block(x, bias, src_mask, src_key_padding_mask)
            x = self.norm1(x + x_att)
            if self.b2t_connection:
                x = self.norm2(x + self._ff_block(x) + src)
            else:
                x = self.norm2(x + self._ff_block(x))
            bias = bias + bias_att
        return x, bias

    def _sa_block(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.norm3(self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1))
        bias = self.conv2d(bias)
        pos = (
            get_relative_positions(bias.shape[-1])
            .to(dtype=bias.dtype, device=bias.device)
            .repeat(bias.shape[0], self.self_attn.num_heads // 2, 1, 1)
        )
        m = torch.tensor(
            get_slopes(self.self_attn.num_heads // 2),
            dtype=bias.dtype,
            device=bias.device,
        )[None, :, None, None]
        attn_mask, key_padding_mask = self._resize_mask(attn_mask, key_padding_mask, x)
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=torch.cat(
                [bias.repeat(1, self.self_attn.num_heads // 2, 1, 1), pos * m], 1
            ).reshape(-1, bias.shape[-2], bias.shape[-1]),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = F.gelu(self.norm4(self.conv1d_t(x.permute(0, 2, 1)).permute(0, 2, 1)))
        bias = self.conv2d_t(bias)
        return self.dropout1(x), bias

    def _resize_mask(self, src_mask, src_key_padding_mask, x):
        src_key_padding_mask = (
            F.interpolate(
                src_key_padding_mask[:, None].to(dtype=x.dtype),
                x.shape[1],
            )[:, 0]
            > 0.5
        )
        if src_mask is not None:
            src_mask = (
                F.interpolate(
                    src_mask[:, None].to(dtype=x.dtype),
                    (x.shape[1], x.shape[1]),
                )[:, 0]
                > 0.5
            )
        return src_mask, src_key_padding_mask


class RibonanzaModel(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.seq_emb = nn.Embedding(4, dim)
        transformer_encoder = []
        for i in range(depth):
            transformer_encoder.append(
                BiasedConvTransformerEncoderLayer(
                    kernel_size=kernel_size if i < depth - 1 else 1,
                    d_model=dim,
                    nhead=dim // head_size,
                    dim_feedforward=4 * dim,
                    dropout=0.1,
                    activation=SwiGLU(),
                    norm_first=False,
                    b2t_connection=b2t_connection,
                )
            )
        self.transformer_encoder = nn.ModuleList(transformer_encoder)
        self.proj_out = nn.Linear(dim, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x_seq = x0["seq"][:, :Lmax]
        bias_bpps = x0["bp_matrix"][:, None, :Lmax, :Lmax]
        x = self.seq_emb(x_seq)
        for i in range(len(self.transformer_encoder)):
            x, bias_bpps = self.transformer_encoder[i](
                x, bias_bpps, src_key_padding_mask=~mask
            )
        x = self.proj_out(x)
        return x


class RibonanzaLightningModel(pl.LightningModule):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
        lr: float = 1e-3,
        disable_compile: bool = False,
        no_amp: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.no_amp = no_amp
        self.__build_model(
            dim=dim,
            depth=depth,
            head_size=head_size,
            kernel_size=kernel_size,
            b2t_connection=b2t_connection,
        )
        if not disable_compile:
            self.__compile_model()
        self.save_hyperparameters()

    def __build_model(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
    ):
        self.model = RibonanzaModel(
            dim=dim,
            depth=depth,
            head_size=head_size,
            kernel_size=kernel_size,
            b2t_connection=b2t_connection,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.999)
        self.criterions = {"l1": nn.L1Loss(reduction="none")}

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        preds = outputs["preds"]
        targets = labels["targets"]
        p = preds[targets["mask"][:, : preds.shape[1]]]
        y = targets["react"][targets["mask"]]
        l1_loss = self.criterions["l1"](p, y)
        if self.training:
            l1_loss = torch.where(
                torch.logical_or(
                    torch.logical_and(p > 10, y > 10),
                    torch.logical_and(p < -10, y < -10),
                ),
                0,
                l1_loss,
            )
        l1_loss = l1_loss[~torch.isnan(l1_loss)].mean()
        losses["loss"] = l1_loss
        losses["l1_loss"] = l1_loss
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        input, label = batch
        outputs["preds"] = self.model(input)
        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_l1_loss=losses["l1_loss"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        input, label = batch
        outputs["preds"] = self.model_ema.module(input).clip(0, 1)
        loss_target["targets"] = label
        loss_target["targets"]["react"][loss_target["targets"]["mask"]] = loss_target[
            "targets"
        ]["react"][loss_target["targets"]["mask"]].clip(0, 1)
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_l1_loss=losses["l1_loss"],
            )
        )
        return step_output

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.05,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(
            self.get_optimizer_parameters(), eps=1e-6 if not self.no_amp else 1e-8
        )
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RibonanzaLightningModel")
        parser.add_argument(
            "--dim",
            default=192,
            type=int,
            metavar="D",
            dest="dim",
        )
        parser.add_argument(
            "--depth",
            default=12,
            type=int,
            metavar="DPT",
            dest="depth",
        )
        parser.add_argument(
            "--head_size",
            default=32,
            type=int,
            metavar="HS",
            dest="head_size",
        )
        parser.add_argument(
            "--kernel_size",
            default=7,
            type=int,
            metavar="KM",
            dest="kernel_size",
        )
        parser.add_argument(
            "--b2t_connection",
            action="store_true",
            help="b2t_connection option",
            dest="b2t_connection",
        )
        parser.add_argument(
            "--lr",
            default=5e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--disable_compile",
            action="store_true",
            help="disable torch.compile",
            dest="disable_compile",
        )
        return parent_parser
