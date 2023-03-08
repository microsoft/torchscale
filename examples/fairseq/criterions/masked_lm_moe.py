# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import MoECriterion, register_criterion, MoECriterionConfig


@register_criterion("masked_lm_moe_cross_entropy", dataclass=MoECriterionConfig)
class MaskedLMMoECrossEntropyCriterion(MoECriterion):

    def compute_inner_loss(self, model, sample, reduce=True):
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        masked_tokens = torch.where(
            masked_tokens.any(),
            masked_tokens,
            masked_tokens.new([True]),
        )

        net_output = model(**sample["net_input"], masked_tokens=masked_tokens)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)

        if masked_tokens is not None:
            target = target[masked_tokens]
            
        nll_loss = F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        logging_output = {
            "inner_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return net_output, nll_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        MaskedLMMoECrossEntropyCriterion.reduce_moe_metrics(logging_outputs)

        loss_sum = sum(log.get("inner_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "inner_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["inner_loss"].avg)
            )