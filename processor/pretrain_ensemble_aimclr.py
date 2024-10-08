import sys
import argparse
import yaml
import math
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class Pretrain_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_joint_value = []
        loss_motion_value = []
        loss_bone_value = []
        ddm_loss = lambda x, y: -torch.mean(
            torch.sum(torch.log(torch.softmax(x, dim=1)) * y, dim=1))  # DDM loss
        mask_loss = lambda x, y: (-(F.log_softmax(x, dim=1) * y).sum(1) / y.sum(1)).mean()
        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            im_e = data1.float().to(self.dev, non_blocking=True)
            im_q = data2.float().to(self.dev, non_blocking=True)
            im_k = data3.float().to(self.dev, non_blocking=True)
            # label = label.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.mining_epoch:
                logits_joint, logits_motion, logits_bone, labels, \
                labels_ddm_joint, labels_ddm_motion, labels_ddm_bone, \
                logits_joint_e, logits_motion_e, logits_bone_e, logits_joint_ed, logits_motion_ed, logits_bone_ed\
                    = self.model(im_q, im_k, im_e)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(logits_joint.size(0))
                else:
                    self.model.update_ptr(logits_joint.size(0))
                loss_joint_1 = self.loss(logits_joint, labels)
                loss_joint_2 = ddm_loss(logits_joint_e, labels_ddm_joint)
                loss_joint_3 = ddm_loss(logits_joint_ed, labels_ddm_joint)
                loss_joint = loss_joint_1 + (loss_joint_2 + loss_joint_3) / 2.

                loss_motion_1 = self.loss(logits_motion, labels)
                loss_motion_2 = ddm_loss(logits_motion_e, labels_ddm_motion)
                loss_motion_3 = ddm_loss(logits_motion_ed, labels_ddm_motion)
                loss_motion = loss_motion_1 + (loss_motion_2 + loss_motion_3) / 2.

                loss_bone_1 = self.loss(logits_bone, labels)
                loss_bone_2 = ddm_loss(logits_bone_e, labels_ddm_bone)
                loss_bone_3 = ddm_loss(logits_bone_ed, labels_ddm_bone)
                loss_bone = loss_bone_1 + (loss_bone_2 + loss_bone_3) / 2.

                loss = loss_joint + loss_motion + loss_bone
            else:
                if self.arg.cross:
                    logits_joint, logits_joint_e, logits_joint_ed, \
                    logits_motion, logits_motion_e, logits_motion_ed, \
                    logits_bone, logits_bone_e, logits_bone_ed, \
                    pos_mask_j, pos_mask_m, pos_mask_b, labels_ddm_joint, labels_ddm_motion, labels_ddm_bone\
                        = self.model(im_q, im_k, im_e, topk=self.arg.topk, cross=True)
                elif self.arg.ensemble:
                    logits_joint, logits_joint_e, logits_joint_ed, \
                    logits_motion, logits_motion_e, logits_motion_ed, \
                    logits_bone, logits_bone_e, logits_bone_ed, \
                    pos_mask_j, pos_mask_m, pos_mask_b, labels_ddm_joint, labels_ddm_motion, labels_ddm_bone \
                        = self.model(im_q, im_k, im_e, topk=self.arg.topk, ensemble=True, context=self.arg.context)
                else:
                    raise 'cross or ensemble must be True'
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(logits_joint.size(0))
                else:
                    self.model.update_ptr(logits_joint.size(0))
                loss_joint_1 = mask_loss(logits_joint, pos_mask_j)
                loss_joint_2 = ddm_loss(logits_joint_e, labels_ddm_joint)
                loss_joint_3 = ddm_loss(logits_joint_ed, labels_ddm_joint)
                loss_joint = loss_joint_1 + (loss_joint_2 + loss_joint_3) / 2.

                loss_motion_1 = mask_loss(logits_motion, pos_mask_m)
                loss_motion_2 = ddm_loss(logits_motion_e, labels_ddm_motion)
                loss_motion_3 = ddm_loss(logits_motion_ed, labels_ddm_motion)
                loss_motion = loss_motion_1 + (loss_motion_2 + loss_motion_3) / 2.

                loss_bone_1 = mask_loss(logits_bone, pos_mask_b)
                loss_bone_2 = ddm_loss(logits_bone_e, labels_ddm_bone)
                loss_bone_3 = ddm_loss(logits_bone_ed, labels_ddm_bone)
                loss_bone = loss_bone_1 + (loss_bone_2 + loss_bone_3) / 2.

                loss = loss_joint + loss_motion + loss_bone

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_joint'] = loss_joint.data.item()
            self.iter_info['loss_motion'] = loss_motion.data.item()
            self.iter_info['loss_bone'] = loss_bone.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            loss_joint_value.append(self.iter_info['loss_joint'])
            loss_motion_value.append(self.iter_info['loss_motion'])
            loss_bone_value.append(self.iter_info['loss_bone'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        self.epoch_info['train_mean_loss_joint'] = np.mean(loss_joint_value)
        self.epoch_info['train_mean_loss_motion'] = np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone'] = np.mean(loss_bone_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_joint', self.epoch_info['train_mean_loss_joint'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')
        # parser.add_argument('--part', type=str, default='body', help='the part of input')
        parser.add_argument('--cross', type=str2bool, default=False, help='use cross or not')
        parser.add_argument('--ensemble', type=str2bool, default=False, help='use ensemble or not')
        parser.add_argument('--context', type=str2bool, default=False, help='use context or not')
        return parser

