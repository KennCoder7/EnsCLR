import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class CLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=num_class,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_k_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_q_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_q_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q_joint.fc.weight.shape[1]
                self.encoder_q_joint.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_q_joint.fc)
                self.encoder_k_joint.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_k_joint.fc)
                self.encoder_q_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q_motion.fc)
                self.encoder_k_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k_motion.fc)
                self.encoder_q_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_bone.fc)
                self.encoder_k_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_bone.fc)

            for param_q, param_k in zip(self.encoder_q_joint.parameters(), self.encoder_k_joint.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue_joint", torch.randn(feature_dim, self.K))
            self.queue_joint = F.normalize(self.queue_joint, dim=0)
            self.register_buffer("queue_ptr_joint", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_motion", torch.randn(feature_dim, self.K))
            self.queue_motion = F.normalize(self.queue_motion, dim=0)
            self.register_buffer("queue_ptr_motion", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_bone", torch.randn(feature_dim, self.K))
            self.queue_bone = F.normalize(self.queue_bone, dim=0)
            self.register_buffer("queue_ptr_bone", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder_joint(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q_joint.parameters(), self.encoder_k_joint.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_motion(self):
        for param_q, param_k in zip(self.encoder_q_motion.parameters(), self.encoder_k_motion.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_bone(self):
        for param_q, param_k in zip(self.encoder_q_bone.parameters(), self.encoder_k_bone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_joint(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_joint)
        gpu_index = keys.device.index
        self.queue_joint[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_motion)
        gpu_index = keys.device.index
        self.queue_motion[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_bone(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_bone)
        gpu_index = keys.device.index
        self.queue_bone[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        # self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_joint[0] = (self.queue_ptr_joint[0] + batch_size) % self.K
        self.queue_ptr_motion[0] = (self.queue_ptr_motion[0] + batch_size) % self.K
        self.queue_ptr_bone[0] = (self.queue_ptr_bone[0] + batch_size) % self.K

    def forward(self, im_q_joint, im_k_joint=None, im_q_joint_extreme=None, view='all', cross=False, topk=1,
                context=True, ensemble=False, single_name=None):
        """
        Input:
            im_q_joint: a batch of query images
            im_k_joint: a batch of key images
        """
        assert cross is False or ensemble is False
        if ensemble:
            return self.ensemble_training(im_q_joint, im_k_joint, im_q_joint_extreme, topk, context)
        if cross:
            return self.cross_training(im_q_joint, im_k_joint, im_q_joint_extreme, topk)

        im_q_motion = torch.zeros_like(im_q_joint)
        im_q_motion[:, :, :-1, :, :] = im_q_joint[:, :, 1:, :, :] - im_q_joint[:, :, :-1, :, :]
        im_q_bone = torch.zeros_like(im_q_joint)
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q_joint[:, :, :, v1 - 1, :] - im_q_joint[:, :, :, v2 - 1, :]

        # if single_name == 'joint':
        #     single_q = im_q_joint
        #     single_q_e = im_q_joint_extreme
        #
        # elif single_name == 'motion':
        #     single_q = im_q_motion
        #     single_q_e = im_q_motion_extreme
        #
        # elif single_name == 'bone':
        #     single_q = im_q_bone
        #
        #     single_q_e = im_q_bone_extreme
        # else:
        #     raise ValueError
        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q_joint(im_q_joint)
            elif view == 'motion':
                return self.encoder_q_motion(im_q_motion)
            elif view == 'bone':
                return self.encoder_q_bone(im_q_bone)
            elif view == 'all':
                return (self.encoder_q_joint(im_q_joint) + self.encoder_q_motion(im_q_motion) + self.encoder_q_bone(
                    im_q_bone)) / 3.
            # elif view == 'single':
            #     return self.encoder_q(single_q)
            else:
                raise ValueError
        im_q_motion_extreme = torch.zeros_like(im_q_joint_extreme)
        im_q_motion_extreme[:, :, :-1, :, :] = im_q_joint_extreme[:, :, 1:, :, :] - im_q_joint_extreme[:, :, :-1, :]
        im_q_bone_extreme = torch.zeros_like(im_q_joint_extreme)
        for v1, v2 in self.Bone:
            im_q_bone_extreme[:, :, :, v1 - 1, :] = im_q_joint_extreme[:, :, :, v1 - 1, :] - im_q_joint_extreme[:,
                                                                                             :, :, v2 - 1, :]
        im_k_motion = torch.zeros_like(im_k_joint)
        im_k_motion[:, :, :-1, :, :] = im_k_joint[:, :, 1:, :, :] - im_k_joint[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k_joint)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k_joint[:, :, :, v1 - 1, :] - im_k_joint[:, :, :, v2 - 1, :]
        # if single_name == 'joint':
        #     single_k = im_k_joint
        # elif single_name == 'motion':
        #     single_k = im_k_motion
        # elif single_name == 'bone':
        #     single_k = im_k_bone
        # else:
        #     raise ValueError
        # compute query features
        q_joint = self.encoder_q_joint(im_q_joint)  # queries: NxC
        q_joint = F.normalize(q_joint, dim=1)
        q_joint_e, q_joint_ed = self.encoder_q_joint(im_q_joint_extreme, drop=True)
        q_joint_e = F.normalize(q_joint_e, dim=1)
        q_joint_ed = F.normalize(q_joint_ed, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)
        q_motion_e, q_motion_ed = self.encoder_q_motion(im_q_motion_extreme, drop=True)
        q_motion_e = F.normalize(q_motion_e, dim=1)
        q_motion_ed = F.normalize(q_motion_ed, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)
        q_bone_e, q_bone_ed = self.encoder_q_bone(im_q_bone_extreme, drop=True)
        q_bone_e = F.normalize(q_bone_e, dim=1)
        q_bone_ed = F.normalize(q_bone_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_joint()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            # self._momentum_update_key_encoder()

            k_joint = self.encoder_k_joint(im_k_joint)  # keys: NxC
            k_joint = F.normalize(k_joint, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_joint = torch.einsum('nc,nc->n', [q_joint, k_joint]).unsqueeze(-1)
        l_neg_joint = torch.einsum('nc,ck->nk', [q_joint, self.queue_joint.clone().detach()])
        l_pos_joint_e = torch.einsum('nc,nc->n', [q_joint_e, k_joint]).unsqueeze(-1)
        l_neg_joint_e = torch.einsum('nc,ck->nk', [q_joint_e, self.queue_joint.clone().detach()])
        l_pos_joint_ed = torch.einsum('nc,nc->n', [q_joint_ed, k_joint]).unsqueeze(-1)
        l_neg_joint_ed = torch.einsum('nc,ck->nk', [q_joint_ed, self.queue_joint.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])
        l_pos_motion_e = torch.einsum('nc,nc->n', [q_motion_e, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_motion_e, self.queue_motion.clone().detach()])
        l_pos_motion_ed = torch.einsum('nc,nc->n', [q_motion_ed, k_motion]).unsqueeze(-1)
        l_neg_motion_ed = torch.einsum('nc,ck->nk', [q_motion_ed, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])
        l_pos_bone_e = torch.einsum('nc,nc->n', [q_bone_e, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_bone_e, self.queue_bone.clone().detach()])
        l_pos_bone_ed = torch.einsum('nc,nc->n', [q_bone_ed, k_bone]).unsqueeze(-1)
        l_neg_bone_ed = torch.einsum('nc,ck->nk', [q_bone_ed, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits_joint = torch.cat([l_pos_joint, l_neg_joint], dim=1)
        logits_joint_e = torch.cat([l_pos_joint_e, l_neg_joint_e], dim=1)
        logits_joint_ed = torch.cat([l_pos_joint_ed, l_neg_joint_ed], dim=1)

        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
        logits_motion_ed = torch.cat([l_pos_motion_ed, l_neg_motion_ed], dim=1)

        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
        logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)
        logits_bone_ed = torch.cat([l_pos_bone_ed, l_neg_bone_ed], dim=1)

        # apply temperature
        logits_joint /= self.T
        logits_joint_e /= self.T
        logits_joint_ed /= self.T

        logits_motion /= self.T
        logits_motion_e /= self.T
        logits_motion_ed /= self.T

        logits_bone /= self.T
        logits_bone_e /= self.T
        logits_bone_ed /= self.T

        labels_ddm_joint = logits_joint.clone().detach()
        labels_ddm_joint = torch.softmax(labels_ddm_joint, dim=1)
        labels_ddm_joint = labels_ddm_joint.detach()

        labels_ddm_motion = logits_motion.clone().detach()
        labels_ddm_motion = torch.softmax(labels_ddm_motion, dim=1)
        labels_ddm_motion = labels_ddm_motion.detach()

        labels_ddm_bone = logits_bone.clone().detach()
        labels_ddm_bone = torch.softmax(labels_ddm_bone, dim=1)
        labels_ddm_bone = labels_ddm_bone.detach()

        # labels: positive key indicators
        labels = torch.zeros(logits_joint.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_joint(k_joint)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits_joint, logits_motion, logits_bone, labels, \
               labels_ddm_joint, labels_ddm_motion, labels_ddm_bone, \
               logits_joint_e, logits_motion_e, logits_bone_e, logits_joint_ed, logits_motion_ed, logits_bone_ed

    def ensemble_training(self, im_q_joint, im_k_joint, im_q_joint_extreme, topk=1, context=True):
        im_q_motion = torch.zeros_like(im_q_joint)
        im_q_motion[:, :, :-1, :, :] = im_q_joint[:, :, 1:, :, :] - im_q_joint[:, :, :-1, :, :]

        im_k_motion = torch.zeros_like(im_k_joint)
        im_k_motion[:, :, :-1, :, :] = im_k_joint[:, :, 1:, :, :] - im_k_joint[:, :, :-1, :, :]

        im_q_bone = torch.zeros_like(im_q_joint)
        im_k_bone = torch.zeros_like(im_k_joint)

        im_q_motion_extreme = torch.zeros_like(im_q_joint_extreme)
        im_q_motion_extreme[:, :, :-1, :, :] = im_q_joint_extreme[:, :, 1:, :, :] - im_q_joint_extreme[:, :, :-1, :,
                                                                                    :]
        im_q_bone_extreme = torch.zeros_like(im_q_joint_extreme)
        for v1, v2 in self.Bone:
            im_q_bone_extreme[:, :, :, v1 - 1, :] = im_q_joint_extreme[:, :, :, v1 - 1, :] - im_q_joint_extreme[:,
                                                                                             :, :, v2 - 1, :]
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q_joint[:, :, :, v1 - 1, :] - im_q_joint[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k_joint[:, :, :, v1 - 1, :] - im_k_joint[:, :, :, v2 - 1, :]

        # if single_name == 'joint':
        #     single_q = im_q_joint
        #     single_k = im_k_joint
        #     # single_q_e = im_q_joint_extreme
        # elif single_name == 'motion':
        #     single_q = im_q_motion
        #     single_k = im_k_motion
        #     # single_q_e = im_q_motion_extreme
        # elif single_name == 'bone':
        #     single_q = im_q_bone
        #     single_k = im_k_bone
        #     # single_q_e = im_q_bone_extreme
        # else:
        #     raise NotImplementedError

        q_joint = self.encoder_q_joint(im_q_joint)  # queries: NxC
        q_joint = F.normalize(q_joint, dim=1)
        q_joint_e, q_joint_ed = self.encoder_q_joint(im_q_joint_extreme, drop=True)
        q_joint_e = F.normalize(q_joint_e, dim=1)
        q_joint_ed = F.normalize(q_joint_ed, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)
        q_motion_e, q_motion_ed = self.encoder_q_motion(im_q_motion_extreme, drop=True)
        q_motion_e = F.normalize(q_motion_e, dim=1)
        q_motion_ed = F.normalize(q_motion_ed, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)
        q_bone_e, q_bone_ed = self.encoder_q_bone(im_q_bone_extreme, drop=True)
        q_bone_e = F.normalize(q_bone_e, dim=1)
        q_bone_ed = F.normalize(q_bone_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_joint()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            # self._momentum_update_key_encoder()

            k_joint = self.encoder_k_joint(im_k_joint)  # keys: NxC
            k_joint = F.normalize(k_joint, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_joint = torch.einsum('nc,nc->n', [q_joint, k_joint]).unsqueeze(-1)
        l_neg_joint = torch.einsum('nc,ck->nk', [q_joint, self.queue_joint.clone().detach()])
        l_pos_joint_e = torch.einsum('nc,nc->n', [q_joint_e, k_joint]).unsqueeze(-1)
        l_neg_joint_e = torch.einsum('nc,ck->nk', [q_joint_e, self.queue_joint.clone().detach()])
        l_pos_joint_ed = torch.einsum('nc,nc->n', [q_joint_ed, k_joint]).unsqueeze(-1)
        l_neg_joint_ed = torch.einsum('nc,ck->nk', [q_joint_ed, self.queue_joint.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])
        l_pos_motion_e = torch.einsum('nc,nc->n', [q_motion_e, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_motion_e, self.queue_motion.clone().detach()])
        l_pos_motion_ed = torch.einsum('nc,nc->n', [q_motion_ed, k_motion]).unsqueeze(-1)
        l_neg_motion_ed = torch.einsum('nc,ck->nk', [q_motion_ed, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])
        l_pos_bone_e = torch.einsum('nc,nc->n', [q_bone_e, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_bone_e, self.queue_bone.clone().detach()])
        l_pos_bone_ed = torch.einsum('nc,nc->n', [q_bone_ed, k_bone]).unsqueeze(-1)
        l_neg_bone_ed = torch.einsum('nc,ck->nk', [q_bone_ed, self.queue_bone.clone().detach()])

        l_ensemble = (l_neg_joint + l_neg_motion + l_neg_bone) / 3.
        l_ensemble_e = (l_neg_joint_e + l_neg_motion_e + l_neg_bone_e) / 3.
        l_ensemble_ed = (l_neg_joint_ed + l_neg_motion_ed + l_neg_bone_ed) / 3.

        if context:
            l_context_j = torch.einsum('nk,nk->nk', [l_neg_joint, l_ensemble])
            l_context_je = torch.einsum('nk,nk->nk', [l_neg_joint_e, l_ensemble_e])
            l_context_jed = torch.einsum('nk,nk->nk', [l_neg_joint_ed, l_ensemble_ed])

            l_context_m = torch.einsum('nk,nk->nk', [l_neg_motion, l_ensemble])
            l_context_me = torch.einsum('nk,nk->nk', [l_neg_motion_e, l_ensemble_e])
            l_context_med = torch.einsum('nk,nk->nk', [l_neg_motion_ed, l_ensemble_ed])

            l_context_b = torch.einsum('nk,nk->nk', [l_neg_bone, l_ensemble])
            l_context_be = torch.einsum('nk,nk->nk', [l_neg_bone_e, l_ensemble_e])
            l_context_bed = torch.einsum('nk,nk->nk', [l_neg_bone_ed, l_ensemble_ed])

            logits_joint = torch.cat([l_pos_joint, l_neg_joint, l_context_j], dim=1)
            logits_joint_e = torch.cat([l_pos_joint_e, l_neg_joint_e, l_context_je], dim=1)
            logits_joint_ed = torch.cat([l_pos_joint_ed, l_neg_joint_ed, l_context_jed], dim=1)

            logits_motion = torch.cat([l_pos_motion, l_neg_motion, l_context_m], dim=1)
            logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e, l_context_me], dim=1)
            logits_motion_ed = torch.cat([l_pos_motion_ed, l_neg_motion_ed, l_context_med], dim=1)

            logits_bone = torch.cat([l_pos_bone, l_neg_bone, l_context_b], dim=1)
            logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e, l_context_be], dim=1)
            logits_bone_ed = torch.cat([l_pos_bone_ed, l_neg_bone_ed, l_context_bed], dim=1)

        else:
            # logits: Nx(1+K)
            logits_joint = torch.cat([l_pos_joint, l_neg_joint], dim=1)
            logits_joint_e = torch.cat([l_pos_joint_e, l_neg_joint_e], dim=1)
            logits_joint_ed = torch.cat([l_pos_joint_ed, l_neg_joint_ed], dim=1)

            logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
            logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
            logits_motion_ed = torch.cat([l_pos_motion_ed, l_neg_motion_ed], dim=1)

            logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
            logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)
            logits_bone_ed = torch.cat([l_pos_bone_ed, l_neg_bone_ed], dim=1)

        # apply temperature
        logits_joint /= self.T
        logits_joint_e /= self.T
        logits_joint_ed /= self.T

        logits_motion /= self.T
        logits_motion_e /= self.T
        logits_motion_ed /= self.T

        logits_bone /= self.T
        logits_bone_e /= self.T
        logits_bone_ed /= self.T

        labels_ddm_joint = logits_joint.clone().detach()
        labels_ddm_joint = torch.softmax(labels_ddm_joint, dim=1)
        labels_ddm_joint = labels_ddm_joint.detach()

        labels_ddm_motion = logits_motion.clone().detach()
        labels_ddm_motion = torch.softmax(labels_ddm_motion, dim=1)
        labels_ddm_motion = labels_ddm_motion.detach()

        labels_ddm_bone = logits_bone.clone().detach()
        labels_ddm_bone = torch.softmax(labels_ddm_bone, dim=1)
        labels_ddm_bone = labels_ddm_bone.detach()

        _, topkdix = torch.topk(l_ensemble, topk, dim=1)
        _, topkdix_e = torch.topk(l_ensemble_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_ensemble_ed, topk, dim=1)

        topk_onehot_ensemble = torch.zeros_like(l_ensemble)
        topk_onehot_ensemble.scatter_(1, topkdix, 1)
        topk_onehot_ensemble.scatter_(1, topkdix_e, 1)
        topk_onehot_ensemble.scatter_(1, topkdix_ed, 1)

        if context:
            pos_mask_j = torch.cat(
                [torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble, topk_onehot_ensemble],
                dim=1)
            pos_mask_m = torch.cat(
                [torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble, topk_onehot_ensemble],
                dim=1)
            pos_mask_b = torch.cat(
                [torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble, topk_onehot_ensemble],
                dim=1)
        else:
            pos_mask_j = torch.cat([torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble], dim=1)
            pos_mask_m = torch.cat([torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble], dim=1)
            pos_mask_b = torch.cat([torch.ones(topk_onehot_ensemble.size(0), 1).cuda(), topk_onehot_ensemble], dim=1)

        self._dequeue_and_enqueue_joint(k_joint)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits_joint, logits_joint_e, logits_joint_ed, \
               logits_motion, logits_motion_e, logits_motion_ed, \
               logits_bone, logits_bone_e, logits_bone_ed, \
               pos_mask_j, pos_mask_m, pos_mask_b, labels_ddm_joint, labels_ddm_motion, labels_ddm_bone

    def cross_training(self, im_q_joint, im_k_joint, im_q_joint_extreme, topk=1):
        im_q_motion = torch.zeros_like(im_q_joint)
        im_q_motion[:, :, :-1, :, :] = im_q_joint[:, :, 1:, :, :] - im_q_joint[:, :, :-1, :, :]

        im_k_motion = torch.zeros_like(im_k_joint)
        im_k_motion[:, :, :-1, :, :] = im_k_joint[:, :, 1:, :, :] - im_k_joint[:, :, :-1, :, :]

        im_q_bone = torch.zeros_like(im_q_joint)
        im_k_bone = torch.zeros_like(im_k_joint)

        im_q_motion_extreme = torch.zeros_like(im_q_joint_extreme)
        im_q_motion_extreme[:, :, :-1, :, :] = im_q_joint_extreme[:, :, 1:, :, :] - im_q_joint_extreme[:, :, :-1, :,
                                                                                    :]
        im_q_bone_extreme = torch.zeros_like(im_q_joint_extreme)
        for v1, v2 in self.Bone:
            im_q_bone_extreme[:, :, :, v1 - 1, :] = im_q_joint_extreme[:, :, :, v1 - 1, :] - im_q_joint_extreme[:,
                                                                                             :, :, v2 - 1, :]
        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q_joint[:, :, :, v1 - 1, :] - im_q_joint[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k_joint[:, :, :, v1 - 1, :] - im_k_joint[:, :, :, v2 - 1, :]

        q_joint = self.encoder_q_joint(im_q_joint)  # queries: NxC
        q_joint = F.normalize(q_joint, dim=1)
        q_joint_e, q_joint_ed = self.encoder_q_joint(im_q_joint_extreme, drop=True)
        q_joint_e = F.normalize(q_joint_e, dim=1)
        q_joint_ed = F.normalize(q_joint_ed, dim=1)

        q_motion = self.encoder_q_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)
        q_motion_e, q_motion_ed = self.encoder_q_motion(im_q_motion_extreme, drop=True)
        q_motion_e = F.normalize(q_motion_e, dim=1)
        q_motion_ed = F.normalize(q_motion_ed, dim=1)

        q_bone = self.encoder_q_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)
        q_bone_e, q_bone_ed = self.encoder_q_bone(im_q_bone_extreme, drop=True)
        q_bone_e = F.normalize(q_bone_e, dim=1)
        q_bone_ed = F.normalize(q_bone_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_joint()  # update the key encoder
            self._momentum_update_key_encoder_motion()
            self._momentum_update_key_encoder_bone()
            # self._momentum_update_key_encoder()

            k_joint = self.encoder_k_joint(im_k_joint)  # keys: NxC
            k_joint = F.normalize(k_joint, dim=1)

            k_motion = self.encoder_k_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.encoder_k_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_joint = torch.einsum('nc,nc->n', [q_joint, k_joint]).unsqueeze(-1)
        l_neg_joint = torch.einsum('nc,ck->nk', [q_joint, self.queue_joint.clone().detach()])
        l_pos_joint_e = torch.einsum('nc,nc->n', [q_joint_e, k_joint]).unsqueeze(-1)
        l_neg_joint_e = torch.einsum('nc,ck->nk', [q_joint_e, self.queue_joint.clone().detach()])
        l_pos_joint_ed = torch.einsum('nc,nc->n', [q_joint_ed, k_joint]).unsqueeze(-1)
        l_neg_joint_ed = torch.einsum('nc,ck->nk', [q_joint_ed, self.queue_joint.clone().detach()])

        l_pos_motion = torch.einsum('nc,nc->n', [q_motion, k_motion]).unsqueeze(-1)
        l_neg_motion = torch.einsum('nc,ck->nk', [q_motion, self.queue_motion.clone().detach()])
        l_pos_motion_e = torch.einsum('nc,nc->n', [q_motion_e, k_motion]).unsqueeze(-1)
        l_neg_motion_e = torch.einsum('nc,ck->nk', [q_motion_e, self.queue_motion.clone().detach()])
        l_pos_motion_ed = torch.einsum('nc,nc->n', [q_motion_ed, k_motion]).unsqueeze(-1)
        l_neg_motion_ed = torch.einsum('nc,ck->nk', [q_motion_ed, self.queue_motion.clone().detach()])

        l_pos_bone = torch.einsum('nc,nc->n', [q_bone, k_bone]).unsqueeze(-1)
        l_neg_bone = torch.einsum('nc,ck->nk', [q_bone, self.queue_bone.clone().detach()])
        l_pos_bone_e = torch.einsum('nc,nc->n', [q_bone_e, k_bone]).unsqueeze(-1)
        l_neg_bone_e = torch.einsum('nc,ck->nk', [q_bone_e, self.queue_bone.clone().detach()])
        l_pos_bone_ed = torch.einsum('nc,nc->n', [q_bone_ed, k_bone]).unsqueeze(-1)
        l_neg_bone_ed = torch.einsum('nc,ck->nk', [q_bone_ed, self.queue_bone.clone().detach()])

        # logits: Nx(1+K)
        logits_joint = torch.cat([l_pos_joint, l_neg_joint], dim=1)
        logits_joint_e = torch.cat([l_pos_joint_e, l_neg_joint_e], dim=1)
        logits_joint_ed = torch.cat([l_pos_joint_ed, l_neg_joint_ed], dim=1)

        logits_motion = torch.cat([l_pos_motion, l_neg_motion], dim=1)
        logits_motion_e = torch.cat([l_pos_motion_e, l_neg_motion_e], dim=1)
        logits_motion_ed = torch.cat([l_pos_motion_ed, l_neg_motion_ed], dim=1)

        logits_bone = torch.cat([l_pos_bone, l_neg_bone], dim=1)
        logits_bone_e = torch.cat([l_pos_bone_e, l_neg_bone_e], dim=1)
        logits_bone_ed = torch.cat([l_pos_bone_ed, l_neg_bone_ed], dim=1)


        # apply temperature
        logits_joint /= self.T
        logits_joint_e /= self.T
        logits_joint_ed /= self.T

        logits_motion /= self.T
        logits_motion_e /= self.T
        logits_motion_ed /= self.T

        logits_bone /= self.T
        logits_bone_e /= self.T
        logits_bone_ed /= self.T

        labels_ddm_joint = logits_joint.clone().detach()
        labels_ddm_joint = torch.softmax(labels_ddm_joint, dim=1)
        labels_ddm_joint = labels_ddm_joint.detach()

        labels_ddm_motion = logits_motion.clone().detach()
        labels_ddm_motion = torch.softmax(labels_ddm_motion, dim=1)
        labels_ddm_motion = labels_ddm_motion.detach()

        labels_ddm_bone = logits_bone.clone().detach()
        labels_ddm_bone = torch.softmax(labels_ddm_bone, dim=1)
        labels_ddm_bone = labels_ddm_bone.detach()

        _, topkdix = torch.topk(l_neg_joint, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_joint_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_joint_ed, topk, dim=1)
        topk_onehot_joint = torch.zeros_like(l_neg_joint)
        topk_onehot_joint.scatter_(1, topkdix, 1)
        topk_onehot_joint.scatter_(1, topkdix_e, 1)
        topk_onehot_joint.scatter_(1, topkdix_ed, 1)

        _, topkdix = torch.topk(l_neg_motion, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_motion_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_motion_ed, topk, dim=1)
        topk_onehot_motion = torch.zeros_like(l_neg_motion)
        topk_onehot_motion.scatter_(1, topkdix, 1)
        topk_onehot_motion.scatter_(1, topkdix_e, 1)
        topk_onehot_motion.scatter_(1, topkdix_ed, 1)

        _, topkdix = torch.topk(l_neg_bone, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_bone_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_bone_ed, topk, dim=1)
        topk_onehot_bone = torch.zeros_like(l_neg_bone)
        topk_onehot_bone.scatter_(1, topkdix, 1)
        topk_onehot_bone.scatter_(1, topkdix_e, 1)
        topk_onehot_bone.scatter_(1, topkdix_ed, 1)

        pos_mask_joint = torch.cat([torch.ones(topk_onehot_joint.size(0), 1).cuda(), topk_onehot_joint], dim=1)
        pos_mask_motion = torch.cat([torch.ones(topk_onehot_motion.size(0), 1).cuda(), topk_onehot_motion], dim=1)
        pos_mask_bone = torch.cat([torch.ones(topk_onehot_bone.size(0), 1).cuda(), topk_onehot_bone], dim=1)

        self._dequeue_and_enqueue_joint(k_joint)
        self._dequeue_and_enqueue_motion(k_motion)
        self._dequeue_and_enqueue_bone(k_bone)

        return logits_joint, logits_joint_e, logits_joint_ed, \
               logits_motion, logits_motion_e, logits_motion_ed, \
               logits_bone, logits_bone_e, logits_bone_ed, \
               pos_mask_joint, pos_mask_motion, pos_mask_bone, \
               labels_ddm_joint, labels_ddm_motion, labels_ddm_bone