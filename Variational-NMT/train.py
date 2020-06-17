#!/usr/bin/env python

import os
import sys
import glob
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Utils.utils import trace
from Utils.utils import check_save_path
from Utils.args import parse_args
from Utils.config import read_config
from Utils.config import format_config
from Utils.DataLoader import DataBatchIterator
from Utils.DataLoader import PAD_WORD


from NMT import Trainer, GNMTGlobalScorer, BatchTranslator, TranslationBuilder
from NMT import Statistics
from NMT import NMTLoss
from NMT import Optimizer
from NMT import model_factory



def train_model(model, optimizer, loss_func,
                train_data_iter, valid_data_iter, config):

    trainer = Trainer(model, loss_func, optimizer, config)

    best_ppl = float('inf')
    patient_cnt = 0
    writer = SummaryWriter(log_dir=config.save_log)

    for epoch in range(1, config.epochs + 1):
        train_iter = iter(train_data_iter)
        valid_iter = iter(valid_data_iter)
        
        # train
        train_stats = trainer.train(
            train_iter, epoch, train_data_iter.num_batches)

        print('')
        trace('Epoch %d, Train acc: %g, ppl: %g'%
              (epoch, train_stats.accuracy(), train_stats.ppl()))

        # validate

        scorer = GNMTGlobalScorer(config.alpha, config.beta, config.coverage_penalty,
                                  config.length_penalty)

        translator = BatchTranslator(model, config, valid_data_iter.trg_vocab, global_scorer=scorer)

        builder = TranslationBuilder(valid_data_iter.src_vocab, valid_data_iter.trg_vocab, config, train=True)

        valid_stats = trainer.validate(valid_iter, translator, builder)
        trace('Epoch %d, Valid acc: %g, ppl: %g' %
              (epoch, valid_stats.accuracy(), valid_stats.ppl()))
        
        writer.add_scalar('Train/acc', train_stats.accuracy(), epoch)
        writer.add_scalar('Train/ppl', train_stats.ppl(), epoch)
        writer.add_scalar('Train/loss', train_stats.loss, epoch)
        writer.add_scalar('Valid/acc', valid_stats.accuracy(), epoch)
        writer.add_scalar('Valid/ppl', valid_stats.ppl(), epoch)
        writer.add_scalar('Valid/loss', valid_stats.loss, epoch)
        writer.add_scalar('LR', optimizer.lr, epoch)

        # # log
        # train_stats.log("train", config.model_name, optimizer.lr)
        # valid_stats.log("valid", config.model_name, optimizer.lr)

        # update the learning rate
        trainer.lr_step(valid_stats.ppl(), epoch)

        if best_ppl > valid_stats.ppl():
            # Renew best perplexity! 
            # dump a checkpoint if needed.
            best_ppl = valid_stats.ppl()
            trainer.dump_checkpoint(epoch, config, train_stats)
        else: 
            patient_cnt += 1 

        if patient_cnt > config.patient_cnt:
            break
        


def build_optimizer(model, config):
    optimizer = Optimizer(config.optim, config)

    optimizer.set_parameters(model.named_parameters())

    return optimizer


def main():
    # Load checkpoint if we resume from a previous training.

    args, parser = parse_args("train")
    config = read_config(args, parser, args.config)
    trace(format_config(config))
    train_data_iter = DataBatchIterator(
                            config=config,
                            is_train=True,
                            dataset="train",
                            batch_size=config.batch_size,
                            shuffle=True)
    train_data_iter.load()

    src_vocab = train_data_iter.src_vocab
    trg_vocab = train_data_iter.trg_vocab

    check_save_path(config.save_vocab)
    torch.save(src_vocab, config.save_vocab + "." + config.src_lang)
    torch.save(trg_vocab, config.save_vocab + "." + config.trg_lang)
    valid_data_iter = DataBatchIterator(
                            config=config, 
                            is_train=True, 
                            dataset="valid",
                            batch_size=config.valid_batch_size)
    valid_data_iter.set_vocab(src_vocab, trg_vocab)
    valid_data_iter.load()

    # Build model.
    model = model_factory(config, src_vocab, trg_vocab)
    if len(config.gpu_ids) > 1:
        trace('Multi gpu training: ', config.gpu_ids)
        model = nn.DataParallel(model, device_ids=config.gpu_ids, dim=1)

    trace(model)

    # Build optimizer.
    optimizer = build_optimizer(model, config)

    padding_idx = trg_vocab.stoi[PAD_WORD]
    # Build loss functions for training set and validation set.
    loss_func = NMTLoss(config, padding_idx)
    # Do training.
    train_model(model, optimizer, loss_func,
                train_data_iter, valid_data_iter, config)


if __name__ == "__main__":
    main()
