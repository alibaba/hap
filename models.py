from __future__ import annotations

import math
import torch
import torch.fx
import torch.nn.functional as F

class TMLP(torch.nn.Module):
    def __init__(self, nhid=2048, nlayers=10, segmentation=True):
        super().__init__()
        modlist = []
        for _ in range(nlayers):
            modlist.append(torch.nn.Linear(nhid, nhid))
            modlist.append(torch.nn.Sigmoid())
        self.layers = torch.nn.ModuleList(modlist)
        self.segmentation = segmentation

    def forward(self, x, y=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.segmentation and i % 2 == 1:
                x = new_segment(x)
        return torch.sum(x)

class TMLP2(torch.nn.Module):
    def __init__(self, nhid=2048, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(nhid, nhid) for _ in range(nlayers)])

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
            x = torch.bmm(x, x)
            x = torch.bmm(x, x)
        return torch.sum(x)

class TTransformer(torch.nn.Module):
    def __init__(self, emsize=2048, nheads=4, nhid=2048, dropout=0.2, nlayers=2, segmentation=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.segmentation = segmentation

    def forward(self, x, y=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.segmentation:
                x = new_segment(x)
        return torch.sum(x)

class TMoE(torch.nn.Module):
    def __init__(self, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers=2): # capacity should be seq_len / n_expert * factor
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class RTransformer(torch.nn.Module):
    def __init__(self, ntokens, seqlen, emsize, nheads, nhid, dropout, nlayers=2, segmentation=True):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(seqlen, emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, emsize)
        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.decoder = torch.nn.Linear(emsize, ntokens)
        self.segmentation = segmentation

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
            if self.segmentation:
                x = new_segment(x)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class RMoE(torch.nn.Module):
    def __init__(self, ntokens, seqlen, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers, segmentation=True):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(seqlen, emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, emsize)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])
        self.decoder = torch.nn.Linear(emsize, ntokens)
        self.segmentation = segmentation

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
            if self.segmentation:
                x = new_segment(x)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class RSwitch(torch.nn.Module):
    def __init__(self, ntokens, seqlen, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(seqlen, emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, emsize)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])
        self.decoder = torch.nn.Linear(emsize, ntokens)

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class VTransformer(torch.nn.Module):
    def __init__(self, nclasses, seqlen, emsize=2048, nheads=4, nhid=2048, dropout=0.2, nlayers=2, segmentation=True):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        assert seqlen == 8 * 8

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, seqlen + 1, emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.decoder = torch.nn.Linear(emsize, nclasses)
        self.segmentation = segmentation

    def forward(self, x, y):
        # x: N, 3, 32, 32
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)
            if self.segmentation:
                x = new_segment(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class VMoE(torch.nn.Module):
    def __init__(self, nclasses, seqlen, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        assert seqlen == 8 * 8

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, seqlen + 1, emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])
        self.decoder = torch.nn.Linear(emsize, nclasses)

    def forward(self, x, y):
        # x: N, 3, 32, 32
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class VSwitch(torch.nn.Module):
    def __init__(self, nclasses, seqlen, emsize, nheads, nhid, dropout, n_expert, capacity, nlayers):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        assert seqlen == 8 * 8

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, seqlen + 1, emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer(emsize, nheads, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])
        self.decoder = torch.nn.Linear(emsize, nclasses)

    def forward(self, x, y):
        # x: N, 3, 32, 32
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class VVGG(torch.nn.Module):
    def __init__(self, nclasses, dropout, segmentation=False):
        super().__init__()

        def make_layers(cfg):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == "M":
                    layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                    in_channels = v
            return torch.nn.Sequential(*layers)

        self.features = make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, nclasses),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.segmentation = segmentation

    def forward(self, x, y):
        for layer in self.features:
            x = layer(x)
            if self.segmentation and isinstance(layer, torch.nn.MaxPool2d):
                x = new_segment(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)


class Top2TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nheads, d_hidden=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, n_expert=4, capacity=None) -> None:
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        self.gate_weight = torch.nn.Parameter(torch.empty((d_model, n_expert)))
        torch.nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        self.w1 = torch.nn.Parameter(torch.empty((n_expert, d_model, d_hidden)))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        self.dropout = torch.nn.Dropout(dropout)

        self.w2 = torch.nn.Parameter(torch.empty((n_expert, d_hidden, d_model)))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.n_expert = n_expert
        self.capacity = capacity
        self.activation = activation

    def forward(self, src: torch.nn.Tensor, src_mask: torch.nn.Tensor | None = None, src_key_padding_mask: torch.nn.Tensor | None = None) -> torch.nn.Tensor:
        """
        gate_input: (batch, seq_len, d_model)
        dispatch_tensor: (batch, seq_len, n_expert, capacity)
        expert_inputs: (batch, n_expert, capacity, d_model)
        expert_outputs: (batch, n_expert, capacity, d_model)
        combine_tensor: (batch, seq_len, n_expert, capacity)
        outputs: (batch, seq_len, d_model)
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._moe_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _moe_block(self, x):
        dispatch_tensor, combine_tensor = top_2_gating(x, self.n_expert, self.capacity, self.gate_weight, train=True) # (batch, seq_len, n_expert, capacity)

        expert_inputs = torch.einsum("bsd,bsec->becd", x, dispatch_tensor) # (batch, n_expert, capacity, d_model)

        h = torch.einsum("edh,becd->bech", self.w1, expert_inputs)

        h = self.activation(h)

        expert_outputs = torch.einsum("ehd,bech->becd", self.w2, h)

        output = torch.einsum("becd,bsec->bsd", expert_outputs, combine_tensor)

        return output

class SwitchTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nheads, d_hidden=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, n_expert=4, capacity=None) -> None:
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)

        self.gate_weight = torch.nn.Parameter(torch.empty((d_model, n_expert)))
        torch.nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        self.w1 = torch.nn.Parameter(torch.empty((n_expert, d_model, d_hidden)))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        self.dropout = torch.nn.Dropout(dropout)

        self.w2 = torch.nn.Parameter(torch.empty((n_expert, d_hidden, d_model)))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.n_expert = n_expert
        self.capacity = capacity
        self.activation = activation

    def forward(self, src: torch.nn.Tensor, src_mask: torch.nn.Tensor | None = None, src_key_padding_mask: torch.nn.Tensor | None = None) -> torch.nn.Tensor:
        """
        gate_input: (batch, seq_len, d_model)
        dispatch_tensor: (batch, seq_len, n_expert, capacity)
        expert_inputs: (batch, n_expert, capacity, d_model)
        expert_outputs: (batch, n_expert, capacity, d_model)
        combine_tensor: (batch, seq_len, n_expert, capacity)
        outputs: (batch, seq_len, d_model)
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._moe_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _moe_block(self, x):
        dispatch_tensor, combine_tensor = switch_gating(x, self.n_expert, self.capacity, self.gate_weight, train=True) # (batch, seq_len, n_expert, capacity)

        expert_inputs = torch.einsum("bsd,bsec->becd", x, dispatch_tensor) # (batch, n_expert, capacity, d_model)

        h = torch.einsum("edh,becd->bech", self.w1, expert_inputs)

        h = self.activation(h)

        expert_outputs = torch.einsum("ehd,bech->becd", self.w2, h)

        output = torch.einsum("becd,bsec->bsd", expert_outputs, combine_tensor)

        return output

@torch.fx.wrap
def switch_gating(gate_input, n_expert, capacity, gate_weight, train: bool = True):
    return _switch_gating(gate_input, n_expert, capacity, gate_weight, train)

@torch.fx.wrap
def top_2_gating(gate_input, n_expert, capacity, gate_weight, train: bool = True):
    return _top_2_gating(gate_input, n_expert, capacity, gate_weight, train)

@torch.fx.wrap
def append_cls_token(x, cls_token):
    return torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

@torch.fx.wrap
def get_cls_token(x):
    return x[:, 0, :]

# @torch.jit.script
def _switch_gating(gate_input, n_expert: int, capacity: int, gate_weight, train: bool = True):
    gate_logits = torch.matmul(gate_input, gate_weight) # (batch, seq_len, n_expert)
    raw_gates = F.softmax(gate_logits, dim=2) # (batch, seq_len, n_expert)

    expert_gate, expert_index = torch.topk(raw_gates, k=1, dim=2, largest=True) # (batch, seq_len, 1)
    expert_gate = torch.squeeze(expert_gate, dim=2) # (batch, seq_len)
    expert_index = torch.squeeze(expert_index, dim=2) # (batch, seq_len)

    expert_mask = F.one_hot(expert_index, n_expert) # (batch, seq_len, n_expert)

    position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask # (batch, seqlen, n_expert)
    expert_mask *= position_in_expert < capacity # (batch, seq_len, n_expert)
    position_in_expert *= position_in_expert < capacity # (batch, seq_len, n_expert)

    expert_mask_flat = torch.sum(expert_mask, dim=2, keepdim=False) # (batch, seq_len)

    combine_tensor = ( # (batch, seq_len, n_expert, capacity)
        torch.unsqueeze(torch.unsqueeze(expert_gate * expert_mask_flat, 2), 3) * # (batch, seq_len, 1, 1)
        torch.unsqueeze(F.one_hot(expert_index, n_expert), 3) * # (batch, seq_len, n_expert, 1) # TODO: why not use expert_mask?
        F.one_hot(position_in_expert, capacity)) # (batch, seq_len, n_expert, capacity)

    dispatch_tensor = (combine_tensor > 0).to(torch.float32)

    return dispatch_tensor, combine_tensor

def _top_2_gating(gate_input, n_expert: int, capacity: int, gate_weight, train: bool = True):
    gate_logits = torch.matmul(gate_input, gate_weight) # (batch, seq_len, n_expert)
    raw_gates = F.softmax(gate_logits, dim=2) # (batch, seq_len, n_expert)

    expert_gate, expert_index = torch.topk(raw_gates, k=2, dim=2, largest=True) # (batch, seq_len, 2)

    expert_mask = F.one_hot(expert_index, n_expert) # (batch, seq_len, 2, n_expert)

    position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask # (batch, seqlen, 2, n_expert)
    expert_1_count = torch.sum(position_in_expert[:, :, 0, :], dim=1, keepdim=True) # (batch, 1, n_expert)
    position_in_expert[:, :, 1, :] += expert_1_count
    position_in_expert = position_in_expert * expert_mask

    expert_mask *= position_in_expert < capacity # (batch, seq_len, 2, n_expert)
    position_in_expert *= position_in_expert < capacity # (batch, seq_len, 2, n_expert)

    expert_mask_flat = torch.sum(expert_mask, dim=3, keepdim=False) # (batch, seq_len, 2)

    combine_tensor = torch.sum(( # (batch, seq_len, n_expert, capacity)
        torch.unsqueeze(torch.unsqueeze(expert_gate * expert_mask_flat, 3), 4) * # (batch, seq_len, 2, 1, 1)
        torch.unsqueeze(F.one_hot(expert_index, n_expert), 4) * # (batch, seq_len, 2, n_expert, 1) # TODO: why not use expert_mask?
        F.one_hot(position_in_expert, capacity)), dim=2, keepdim=False) # (batch, seq_len, 2, n_expert, capacity)

    dispatch_tensor = (combine_tensor > 0).to(torch.float32)

    return dispatch_tensor, combine_tensor

def positional_encoding(seqlen, emsize):
    import numpy as np
    p = np.array([[pos / np.power(10000, 2 * (j // 2) / emsize) for j in range(emsize)] for pos in range(seqlen)])
    p[:, 0::2] = np.sin(p[:, 0::2])
    p[:, 1::2] = np.cos(p[:, 1::2])
    return p


# https://github.com/rwightman/pytorch-image-models/blob/f670d98cb8ec70ed6e03b4be60a18faf4dc913b5/timm/models/layers/patch_embed.py#L15
class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        x = torch.flatten(x, 2).transpose(1, 2)  # BCHW -> BNC
        return x

# TODO: make it accepting multiple arguments?
@torch.fx.wrap
def new_segment(x):
    return x
