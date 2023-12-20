# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]


from typing import Any, Optional
import torch

if torch.cuda.is_available():
    try:
        if torch.cuda.get_device_capability()[0] > 7:
            from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func

            def flash_attn_func(q, k, v, dropout=0.0, bias=None, softmax_scale=None, is_causal=False):
                assert bias is None
                attn, lse, _ = _flash_attn_func(q, k, v, dropout_p=dropout, softmax_scale=softmax_scale, causal=is_causal, return_attn_probs=True)
                return attn, lse

        else:
            from xformers.ops.fmha import (
                cutlass,
                Inputs,
                Context,
                _memory_efficient_attention_forward_requires_grad,
                _memory_efficient_attention_backward,
                LowerTriangularMask,
            )

            class FlashAttnFunc(torch.autograd.Function):
                @staticmethod
                # type: ignore
                def forward(ctx, q, k, v, dropout=0.0, bias=None, softmax_scale=None, is_causal=False):
                    if is_causal:
                        assert bias is None
                        attn_bias = LowerTriangularMask()
                    else:
                        attn_bias = bias

                    inp = Inputs(
                        query=q,
                        key=k,
                        value=v,
                        attn_bias=attn_bias,
                        p=dropout,
                        scale=softmax_scale,
                    )
                    op_fw = cutlass.FwOp
                    op_bw = cutlass.BwOp

                    out, op_ctx = _memory_efficient_attention_forward_requires_grad(
                        inp=inp, op=op_fw
                    )

                    # Saving attn_bias is a bit complicated, as the
                    # torch part should go in `save_for_backward`
                    if isinstance(inp.attn_bias, torch.Tensor):
                        attn_bias_tensor = inp.attn_bias
                        attn_bias_ctx = None
                    else:
                        attn_bias_tensor = None
                        attn_bias_ctx = inp.attn_bias

                    ctx.save_for_backward(
                        inp.query,
                        inp.key,
                        inp.value,
                        op_ctx.out,
                        op_ctx.lse,
                    )
                    ctx.rng_state = op_ctx.rng_state
                    ctx.attn_bias_tensor = attn_bias_tensor
                    if op_ctx.op_bw is not None:
                        if op_bw is not None and op_bw is not op_ctx.op_bw:
                            raise ValueError(
                                f"Specified op_bw={op_bw.NAME}, but forward op "
                                f"can only run with op_bw={op_ctx.op_bw.NAME}. Please set op_bw=None."
                            )
                        op_bw = op_ctx.op_bw
                    ctx.op_fw = op_fw
                    ctx.op_bw = op_bw
                    ctx.p = inp.p

                    ctx.scale = inp.scale
                    ctx.attn_bias_ctx = attn_bias_ctx
                    return out, op_ctx.lse

                @staticmethod
                def deserialize_bias(
                    attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]
                ) -> Any:
                    if attn_bias_tensor is None:
                        return attn_bias_ctx
                    return attn_bias_tensor

                @classmethod
                @torch.autograd.function.once_differentiable
                def backward(cls, ctx, grad, dlse):
                    # Re-create context
                    query, key, value, out, lse = ctx.saved_tensors
                    attn_bias_tensor = ctx.attn_bias_tensor
                    rng_state = ctx.rng_state
                    inp = Inputs(
                        query=query,
                        key=key,
                        value=value,
                        attn_bias=cls.deserialize_bias(ctx.attn_bias_ctx, attn_bias_tensor),
                        p=ctx.p,
                        scale=ctx.scale,
                    )
                    op_ctx = Context(
                        lse=lse,
                        out=out,
                        rng_state=rng_state,
                    )
                    grads = _memory_efficient_attention_backward(
                        ctx=op_ctx, inp=inp, grad=grad, op=ctx.op_bw
                    )
                    return grads.dq, grads.dk, grads.dv, None, grads.db, None, None
            
            flash_attn_func = FlashAttnFunc.apply
    except ModuleNotFoundError:
        flash_attn_func = None
else:
    flash_attn_func = None
