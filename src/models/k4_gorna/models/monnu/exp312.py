import argparse
import math
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from timm.utils import ModelEmaV2
from torch import nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "312"
COMMENT = """
    kfold, postnorm, high weight decay, long warmup, low s/n threshold, 
    conv transformer, SHAPE positional encoding, bpps bias, efficient impl, 
    param tuning from exp034, swiGLU, split attention ALiBi and bpps, 
    fixed 0-1 clipping, B2T connection option, low grad clipping, add norm and act for conv1d
    with pseudo_label, RMSNorm
    """

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Conv(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))


class ResidualGraphAttention(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h)
    

class SEResidual(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()
        self.se = SELayer(d_model)

    def forward(self, src):
        h = self.conv2(self.conv1(src))
        return self.se(self.relu(src + h))

class RibonanzaModel(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 9,
        dropout=0.1,
        dropout_res=0.1,
        d_model = 256,
        kernel_size_conv=9, 
        kernel_size_gc=9,
        num_layers=12,
        **kwargs,
    ):
        kernel_sizes = [9,9,9,7,7,7,5,5,5,3,3,3]
        super(RibonanzaModel, self).__init__()
        self.dim = dim
        self.seq_emb = nn.Embedding(4, dim)
        self.conv = Conv(dim, d_model, kernel_size=5, dropout=dropout)
    
        self.blocks = nn.ModuleList([SEResidual(d_model, kernel_size=kernel_sizes[i], dropout=dropout_res) for i in range(num_layers)])
        self.attentions = nn.ModuleList([ResidualGraphAttention(d_model, kernel_size=kernel_sizes[i], dropout=dropout_res) for i in range(num_layers)])
        self.lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, num_layers=2, bidirectional=True)
        self.proj_out = nn.Linear(d_model, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x_seq = x0["seq"][:, :Lmax]
        bpps = x0["bp_matrix"][:, :Lmax, :Lmax]
        x = self.seq_emb(x_seq)
        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]
        x = self.conv(x)
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x)
            x = attention(x, bpps)
        x = x.permute([0, 2, 1])  # [batch, seq, features]
        x,_ = self.lstm(x)
        out = self.proj_out(x)
        return out


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
            dim=dim, depth=depth, head_size=head_size, kernel_size=3,
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
            num_training_steps=max_train_steps//2,#fix
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
