# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_kl_div')
class LabelSmoothedCrossEntropyCriterionWithKLDivergence(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, KL_lambda):
        super().__init__(task, sentence_avg, label_smoothing)
        self.KL_lambda = KL_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--KL-lambda', default=1.0, type=float, metavar='D',
                            help='weight for the KL-divergence loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        KL_div = self.compute_KL_divergence(net_output[1], net_output[2])
        if KL_div is not None:
            logging_output["kl_div"] = utils.item(KL_div.data)
            loss += self.KL_lambda * KL_div

        return loss, sample_size, logging_output

    def compute_KL_divergence(self, piror_out, pos_approx_out):
        KLD_list = []

        for prior, pos_approx in zip(prior_out, pos_approx_out):
            mu_a, R_a = prior
            mu_b, R_b = pos_approx
            C_a = R_a @ R_a.transpose(-1, -2) # batch_size x sequence_len x size x size
            C_b = R_b @ R_b.transpose(-1, -2) # batch_size x sequence_len x size x size
            KLD = 0.5
            KLD_list.append(KLD)



        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        kl_div_loss_sum = utils.item(sum(log.get('kl_div', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('kl_div_loss', kl_div_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True