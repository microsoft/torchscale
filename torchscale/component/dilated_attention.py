# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from .multihead_attention import MultiheadAttention
from .utils import padding_to_multiple_of, all_gather_func, get_data_parallel_rank, get_data_parallel_world_size


class DilatedAttention(MultiheadAttention):

    def dense_to_sparse(self, x, ratio):
        length = x.size(1)
        padding = padding_to_multiple_of(length, ratio)
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value = 0.)

        x = rearrange(x, 'b (l r1) (r2 h) d -> b l h d r1 r2', r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, 'b l h d r -> b l (r h) d')
        
        if head_padding > 0:
            x = x[:, :, :self.num_heads]

        return x

    def sparse_to_dense(self, out, lse, ratio):
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value = 0.)
            lse = F.pad(lse, (0, 0, 0, head_padding), value = -1e8)

        out = rearrange(out, 'b l (r h) d -> b l h d r', r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, 'b l h d r1 r2 -> b (r2 h) (l r1) d', r1=ratio, r2=ratio)

        lse = rearrange(lse, 'b (r h) l -> b l h r', r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse==0, -1e8)
        lse = rearrange(lse, 'b l h r1 r2 -> b (r2 h) (l r1) 1', r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, :self.num_heads]
            lse = lse[:, :self.num_heads]

        return out, lse

    def gather_kv(self, x, sl, seq_len, is_causal=True):
        bsz = x.size(0)
        assert sl % seq_len == 0
        num_rank_per_segment = sl // seq_len

        x = all_gather_func(x)
        current_rank = get_data_parallel_rank()
        x = rearrange(x, '(w b) l h d -> w b l h d', b=bsz)
        
        if is_causal:
            if current_rank > 0:
                x = x[:current_rank]
            else:
                x = x[:1] * 0
        
        current_segment = current_rank // num_rank_per_segment * num_rank_per_segment
        x = x[current_segment:current_segment+num_rank_per_segment]

        x = rearrange(x, 'w b l h d -> b (w l) h d')
        return x
    
    def gathering(self, x, dr, sl, is_causal=True, offset=0, is_kv=False, seq_parall=True):

        curr_x = x
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.)
        seq_len = curr_x.size(1)
        should_gather_kv = is_kv and (get_data_parallel_world_size() > 1) and (sl > seq_len) and seq_parall
        _sl = sl
        sl = min(sl, seq_len)
        padding = padding_to_multiple_of(seq_len, sl)

        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value = 0.)

        curr_x = rearrange(curr_x, 'b (n g) h d -> (b n) g h d', g=sl)
        curr_x = self.dense_to_sparse(curr_x, dr)

        if should_gather_kv:
            curr_x = self.gather_kv(curr_x, _sl, seq_len, is_causal)

        curr_x = rearrange(curr_x, 'b l h d -> (b h) l d')
        
        return curr_x

    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.args.dilated_ratio) == 0
        all_outs, all_lses = [], []
        drs = self.args.dilated_ratio
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses):
            o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, '(b n) h g d -> (b h) (n g) d', b=bsz)
            lse = rearrange(lse, '(b n) h g 1 -> (b h) (n g) 1', b=bsz)
            o = o[:, offset:offset+seq_len]
            lse = lse[:, offset:offset+seq_len]

            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0)
            max_lse = max_lse.max(0)[0]
            all_lses = [torch.exp(lse-max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = 0
        for o, lse in zip(all_outs, all_lses):
            out += o * lse.type_as(o)
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.num_heads)

        return out

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        assert self.args.flash_attention
        assert rel_pos is None
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> (b h) l d', h=self.num_heads)

        if incremental_state is not None and not is_first_step:
            offset = src_len - 1
        else:
            offset = 0

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None and not is_first_step:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)
        
        q = rearrange(q, '(b h) l d -> b l h d', h=self.num_heads)
        k = rearrange(k, '(b h) l d -> b l h d', h=self.num_heads)
        v = rearrange(v, '(b h) l d -> b l h d', h=self.num_heads)

        outs, lses = [], []
        for sl, dr in zip(self.args.segment_length, self.args.dilated_ratio):
            ki = self.gathering(k, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.args.seq_parallel)
            vi = self.gathering(v, dr, sl, is_causal=is_causal, offset=0, is_kv=True, seq_parall=self.args.seq_parallel)
            qi = self.gathering(q, dr, sl, is_causal=is_causal, offset=offset, is_kv=False, seq_parall=self.args.seq_parallel)

            out, lse = self.attention_ops(qi, ki, vi, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)

            outs.append(out)
            lses.append(lse)

        attn = self.scattering(outs, lses, tgt_len, bsz, offset=offset)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, None
