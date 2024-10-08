#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LE_Processor(Processor):
    """
        Processor for Linear Evaluation.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        assert self.arg.model_args['pretrain'] == False
        # assert self.arg.model_args['return_each_feature'] == ''
        self.model.apply(weights_init)
        self.num_grad_layers = 0
        fc_weight_name = []
        fc_bias_name = []
        for i in range(self.arg.ensemble_n):
            fc_weight_name.append('encoder_q.{}.fc.weight'.format(i))
            fc_bias_name.append('encoder_q.{}.fc.bias'.format(i))
        print(fc_weight_name)
        print(fc_bias_name)
        for name, param in self.model.named_parameters():
            print(name)
            # if name not in ['fc.weight', 'fc.bias']:
            #     param.requires_grad = False
            if name in fc_weight_name or name in fc_bias_name:
                param.requires_grad = True
                self.num_grad_layers += 1
            else:
                param.requires_grad = False
        print('num_grad_layers: ', self.num_grad_layers)
        assert self.num_grad_layers == self.arg.ensemble_n * 2
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == self.num_grad_layers
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            self.global_step += 1
            # get data
            data1 = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            if self.arg.ensemble_n == 3:
                im_q = {'0': self.get_stream(data1, 'joint'),
                        '1': self.get_stream(data1, 'motion'),
                        '2': self.get_stream(data1, 'bone')}
            elif self.arg.ensemble_n == 6:
                im_q = {'0': self.get_stream(data1, 'joint'),
                        '1': self.get_stream(data1, 'motion'),
                        '2': self.get_stream(data1, 'bone'),
                        '3': self.get_stream(data1, 'joint'),
                        '4': self.get_stream(data1, 'motion'),
                        '5': self.get_stream(data1, 'bone')
                        }
            else:
                raise NotImplementedError
            # forward
            output = self.model(im_q, view=self.arg.view)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            data1 = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            if self.arg.ensemble_n == 3:
                im_q = {'0': self.get_stream(data1, 'joint'),
                        '1': self.get_stream(data1, 'motion'),
                        '2': self.get_stream(data1, 'bone')}
            elif self.arg.ensemble_n == 6:
                im_q = {'0': self.get_stream(data1, 'joint'),
                        '1': self.get_stream(data1, 'motion'),
                        '2': self.get_stream(data1, 'bone'),
                        '3': self.get_stream(data1, 'joint'),
                        '4': self.get_stream(data1, 'motion'),
                        '5': self.get_stream(data1, 'bone')
                        }
            else:
                raise NotImplementedError
            # inference
            with torch.no_grad():
                output = self.model(im_q, view=self.arg.view)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss'] = np.mean(loss_value)
        self.show_eval_info()

        # show top-k accuracy 
        for k in self.arg.show_topk:
            self.show_topk(k)
        self.show_best(1)

        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')
        parser.add_argument('--part', type=str, default='body', help='the part of input')
        parser.add_argument('--ensemble_n', type=int, default=3)

        # endregion yapf: enable

        return parser

    def get_stream(self, data1, view, part='body'):
        if view == 'joint':
            data1_part = data1
            pass
        elif view == 'motion':
            motion1 = torch.zeros_like(data1)

            motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
            del data1

            data1_part = motion1
        elif view == 'bone':
            if part == 'body':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
            elif part == 'hand':
                Bone = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 5),
                        (8, 1), (9, 8), (10, 9), (11, 10), (12, 11), (13, 11), (14, 1), (15, 1), (16, 15)]
            elif part == 'leg':
                Bone = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4),
                        (5, 1), (6, 5), (7, 6), (8, 7), (9, 1)]
            elif part == 'left':
                Bone = [(1, 1), (2, 1), (3, 2), (4, 1), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 1),
                        (11, 10), (12, 11), (13, 12), (14, 13), (15, 14)]
            elif part == 'right':
                Bone = [(1, 1), (2, 1), (3, 2), (4, 1), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 1),
                        (11, 10), (12, 11), (13, 12), (14, 13), (15, 14)]
            else:
                raise

            bone1 = torch.zeros_like(data1)

            for v1, v2 in Bone:
                bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
            del data1

            data1_part = bone1
        else:
            raise ValueError
        return data1_part