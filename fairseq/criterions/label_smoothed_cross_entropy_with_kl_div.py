# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_kl_div')
class LabelSmoothedCrossEntropyCriterionWithKLDivergence(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, KL_lambda, alpha):
        super().__init__(task, sentence_avg, label_smoothing)
        self.KL_lambda = KL_lambda
        self.alpha = alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--KL-lambda', default=1.0, type=float, metavar='D',
                            help='weight for the KL-divergence loss')
        parser.add_argument('--alpha', default=1.0, type=float, metavar='D',
                            help='parameter for the weight of GSNN model loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #import pdb
        #pdb.set_trace()
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)
        gsnn_loss, gsnn_nll_loss = None, None
        if net_output[1] is not None:
            gsnn_loss, gsnn_nll_loss = self.compute_loss(model, net_output[1], sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        KL_div = self.compute_kl_divergence(net_output[2], net_output[3])

        if KL_div is not None:
            logging_output["kl_div"] = utils.item(KL_div.data)
            loss += self.KL_lambda * KL_div
            if gsnn_loss is not None:
                logging_output['gsnn_loss'] = utils.item(gsnn_loss.data) if reduce else gsnn_loss.data
                logging_output['gsnn_nll_loss'] = utils.item(gsnn_nll_loss.data) if reduce else gsnn_nll_loss.data
                loss += self.alpha * gsnn_loss

        logging_output['total_loss'] = utils.item(loss.data) if reduce else loss.data
        return loss, sample_size, logging_output

    def compute_kl_divergence(self, pos_approx_out, prior_out):

        if pos_approx_out is not None and prior_out is not None:
            kl_div_list = []
            for pos_approx, prior in zip(pos_approx_out.encoder_states, prior_out.encoder_states):
                pos_mu, pos_logvar = pos_approx
                pri_mu, pri_logvar = prior
                pos_mu, pos_logvar = torch.mean(pos_mu, dim=0), torch.mean(pos_logvar, dim=0)
                pri_mu, pri_logvar = torch.mean(pri_mu, dim=0), torch.mean(pri_logvar, dim=0)
                kl_div = - 0.5 * torch.sum(1 + (pos_logvar - pri_logvar)
                                        - torch.div((pri_mu - pos_mu).pow(2), (pri_logvar.exp()))
                                        - torch.div(pos_logvar.exp(), (pri_logvar.exp())))
                kl_div_list.append(kl_div)

            return torch.stack(kl_div_list, dim=0).sum(dim=0)
        else:
            return None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        total_sum = utils.item(sum(log.get('total_loss', 0) for log in logging_outputs))
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        gsnn_loss_sum = utils.item(sum(log.get('gsnn_loss', 0) for log in logging_outputs))
        gsnn_nll_loss_sum = utils.item(sum(log.get('gsnn_nll_loss', 0) for log in logging_outputs))
        kl_div_loss_sum = utils.item(sum(log.get('kl_div', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('total_loss', total_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('gsnn_loss', gsnn_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('gsnn_nll_loss', gsnn_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
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
