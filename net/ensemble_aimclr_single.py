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
        self.gamma = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                      (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                      (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        if not self.pretrain:
            self.encoder_q_alpha = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=num_class,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_q_beta = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_gamma = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=num_class,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q_alpha = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_k_alpha = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_q_beta = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_k_beta = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.encoder_q_gamma = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)
            self.encoder_k_gamma = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                hidden_dim=hidden_dim, num_class=feature_dim,
                                                dropout=dropout, graph_args=graph_args,
                                                edge_importance_weighting=edge_importance_weighting,
                                                **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q_alpha.fc.weight.shape[1]
                self.encoder_q_alpha.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_q_alpha.fc)
                self.encoder_k_alpha.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_k_alpha.fc)
                self.encoder_q_beta.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_q_beta.fc)
                self.encoder_k_beta.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.encoder_k_beta.fc)
                self.encoder_q_gamma.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_q_gamma.fc)
                self.encoder_k_gamma.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                        nn.ReLU(),
                                                        self.encoder_k_gamma.fc)

            for param_q, param_k in zip(self.encoder_q_alpha.parameters(), self.encoder_k_alpha.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_beta.parameters(), self.encoder_k_beta.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            for param_q, param_k in zip(self.encoder_q_gamma.parameters(), self.encoder_k_gamma.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue_alpha", torch.randn(feature_dim, self.K))
            self.queue_alpha = F.normalize(self.queue_alpha, dim=0)
            self.register_buffer("queue_ptr_alpha", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_beta", torch.randn(feature_dim, self.K))
            self.queue_beta = F.normalize(self.queue_beta, dim=0)
            self.register_buffer("queue_ptr_beta", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_gamma", torch.randn(feature_dim, self.K))
            self.queue_gamma = F.normalize(self.queue_gamma, dim=0)
            self.register_buffer("queue_ptr_gamma", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder_alpha(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q_alpha.parameters(), self.encoder_k_alpha.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_beta(self):
        for param_q, param_k in zip(self.encoder_q_beta.parameters(), self.encoder_k_beta.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_gamma(self):
        for param_q, param_k in zip(self.encoder_q_gamma.parameters(), self.encoder_k_gamma.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_alpha(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_alpha)
        gpu_index = keys.device.index
        self.queue_alpha[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_beta(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_beta)
        gpu_index = keys.device.index
        self.queue_beta[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def _dequeue_and_enqueue_gamma(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_gamma)
        gpu_index = keys.device.index
        self.queue_gamma[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        # self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_alpha[0] = (self.queue_ptr_alpha[0] + batch_size) % self.K
        self.queue_ptr_beta[0] = (self.queue_ptr_beta[0] + batch_size) % self.K
        self.queue_ptr_gamma[0] = (self.queue_ptr_gamma[0] + batch_size) % self.K

    def forward(self, im_q_alpha, im_k_alpha=None, im_q_alpha_extreme=None, view='all', cross=False, topk=1,
                context=True, ensemble=False, single_name=None):
        """
        Input:
            im_q_alpha: a batch of query images
            im_k_alpha: a batch of key images
        """
        assert cross is False or ensemble is False
        if ensemble:
            return self.ensemble_training(im_q_alpha, im_k_alpha, im_q_alpha_extreme, topk, context)
        if cross:
            return self.cross_training(im_q_alpha, im_k_alpha, im_q_alpha_extreme, topk)
        if self.pretrain:
            (im_q_alpha, im_q_beta, im_q_gamma) = (im_q_alpha[:, 0:3], im_q_alpha[:, 3:6], im_q_alpha[:, 6:9])
        # print(im_q_alpha.shape, im_q_beta.shape, im_q_gamma.shape)
        # print(im_q_alpha == im_q_beta)
        #     if True:  # for visualization
        #         q_alpha = self.encoder_q_alpha(im_q_alpha)  # queries: NxC
        #         q_alpha = F.normalize(q_alpha, dim=1)
        #
        #         q_beta = self.encoder_q_beta(im_q_beta)
        #         q_beta = F.normalize(q_beta, dim=1)
        #
        #         q_gamma = self.encoder_q_gamma(im_q_gamma)
        #         q_gamma = F.normalize(q_gamma, dim=1)
        #
        #
        #         exit()
        else:
            im_q_alpha = im_q_alpha
            im_q_beta = im_q_alpha
            im_q_gamma = im_q_alpha
            if view == 'alpha':
                return self.encoder_q_alpha(im_q_alpha)
            elif view == 'beta':
                return self.encoder_q_beta(im_q_beta)
            elif view == 'gamma':
                return self.encoder_q_gamma(im_q_gamma)
            elif view == 'all':
                return (self.encoder_q_alpha(im_q_alpha) + self.encoder_q_beta(im_q_beta) + self.encoder_q_gamma(
                    im_q_gamma)) / 3.
            # elif view == 'single':
            #     return self.encoder_q(single_q)
            else:
                raise ValueError
        (im_q_alpha_extreme, im_q_beta_extreme, im_q_gamma_extreme) = (im_q_alpha_extreme[:, 0:3],
                                                                       im_q_alpha_extreme[:, 3:6],
                                                                       im_q_alpha_extreme[:, 6:9])
        (im_k_alpha, im_k_beta, im_k_gamma) = (im_k_alpha[:, 0:3], im_k_alpha[:, 3:6], im_k_alpha[:, 6:9])

        q_alpha = self.encoder_q_alpha(im_q_alpha)  # queries: NxC
        q_alpha = F.normalize(q_alpha, dim=1)
        q_alpha_e, q_alpha_ed = self.encoder_q_alpha(im_q_alpha_extreme, drop=True)
        q_alpha_e = F.normalize(q_alpha_e, dim=1)
        q_alpha_ed = F.normalize(q_alpha_ed, dim=1)

        q_beta = self.encoder_q_beta(im_q_beta)
        q_beta = F.normalize(q_beta, dim=1)
        q_beta_e, q_beta_ed = self.encoder_q_beta(im_q_beta_extreme, drop=True)
        q_beta_e = F.normalize(q_beta_e, dim=1)
        q_beta_ed = F.normalize(q_beta_ed, dim=1)

        q_gamma = self.encoder_q_gamma(im_q_gamma)
        q_gamma = F.normalize(q_gamma, dim=1)
        q_gamma_e, q_gamma_ed = self.encoder_q_gamma(im_q_gamma_extreme, drop=True)
        q_gamma_e = F.normalize(q_gamma_e, dim=1)
        q_gamma_ed = F.normalize(q_gamma_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_alpha()  # update the key encoder
            self._momentum_update_key_encoder_beta()
            self._momentum_update_key_encoder_gamma()
            # self._momentum_update_key_encoder()

            k_alpha = self.encoder_k_alpha(im_k_alpha)  # keys: NxC
            k_alpha = F.normalize(k_alpha, dim=1)

            k_beta = self.encoder_k_beta(im_k_beta)
            k_beta = F.normalize(k_beta, dim=1)

            k_gamma = self.encoder_k_gamma(im_k_gamma)
            k_gamma = F.normalize(k_gamma, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_alpha = torch.einsum('nc,nc->n', [q_alpha, k_alpha]).unsqueeze(-1)
        l_neg_alpha = torch.einsum('nc,ck->nk', [q_alpha, self.queue_alpha.clone().detach()])
        l_pos_alpha_e = torch.einsum('nc,nc->n', [q_alpha_e, k_alpha]).unsqueeze(-1)
        l_neg_alpha_e = torch.einsum('nc,ck->nk', [q_alpha_e, self.queue_alpha.clone().detach()])
        l_pos_alpha_ed = torch.einsum('nc,nc->n', [q_alpha_ed, k_alpha]).unsqueeze(-1)
        l_neg_alpha_ed = torch.einsum('nc,ck->nk', [q_alpha_ed, self.queue_alpha.clone().detach()])

        l_pos_beta = torch.einsum('nc,nc->n', [q_beta, k_beta]).unsqueeze(-1)
        l_neg_beta = torch.einsum('nc,ck->nk', [q_beta, self.queue_beta.clone().detach()])
        l_pos_beta_e = torch.einsum('nc,nc->n', [q_beta_e, k_beta]).unsqueeze(-1)
        l_neg_beta_e = torch.einsum('nc,ck->nk', [q_beta_e, self.queue_beta.clone().detach()])
        l_pos_beta_ed = torch.einsum('nc,nc->n', [q_beta_ed, k_beta]).unsqueeze(-1)
        l_neg_beta_ed = torch.einsum('nc,ck->nk', [q_beta_ed, self.queue_beta.clone().detach()])

        l_pos_gamma = torch.einsum('nc,nc->n', [q_gamma, k_gamma]).unsqueeze(-1)
        l_neg_gamma = torch.einsum('nc,ck->nk', [q_gamma, self.queue_gamma.clone().detach()])
        l_pos_gamma_e = torch.einsum('nc,nc->n', [q_gamma_e, k_gamma]).unsqueeze(-1)
        l_neg_gamma_e = torch.einsum('nc,ck->nk', [q_gamma_e, self.queue_gamma.clone().detach()])
        l_pos_gamma_ed = torch.einsum('nc,nc->n', [q_gamma_ed, k_gamma]).unsqueeze(-1)
        l_neg_gamma_ed = torch.einsum('nc,ck->nk', [q_gamma_ed, self.queue_gamma.clone().detach()])

        l_aplha_beta = torch.einsum('nc,nc->n', [q_alpha, q_beta]).unsqueeze(-1)
        l_aplha_beta_e = torch.einsum('nc,nc->n', [q_alpha_e, q_beta_e]).unsqueeze(-1)
        l_aplha_beta_ed = torch.einsum('nc,nc->n', [q_alpha_ed, q_beta_ed]).unsqueeze(-1)

        l_aplha_gamma = torch.einsum('nc,nc->n', [q_alpha, q_gamma]).unsqueeze(-1)
        l_aplha_gamma_e = torch.einsum('nc,nc->n', [q_alpha_e, q_gamma_e]).unsqueeze(-1)
        l_aplha_gamma_ed = torch.einsum('nc,nc->n', [q_alpha_ed, q_gamma_ed]).unsqueeze(-1)

        l_beta_gamma = torch.einsum('nc,nc->n', [q_beta, q_gamma]).unsqueeze(-1)
        l_beta_gamma_e = torch.einsum('nc,nc->n', [q_beta_e, q_gamma_e]).unsqueeze(-1)
        l_beta_gamma_ed = torch.einsum('nc,nc->n', [q_beta_ed, q_gamma_ed]).unsqueeze(-1)

        # logits: Nx(1+K+1+1)
        logits_alpha = torch.cat([l_pos_alpha, l_neg_alpha, l_aplha_beta, l_aplha_gamma], dim=1)
        logits_alpha_e = torch.cat([l_pos_alpha_e, l_neg_alpha_e, l_aplha_beta_e, l_aplha_gamma_e], dim=1)
        logits_alpha_ed = torch.cat([l_pos_alpha_ed, l_neg_alpha_ed, l_aplha_beta_ed, l_aplha_gamma_ed], dim=1)

        logits_beta = torch.cat([l_pos_beta, l_neg_beta, l_aplha_beta, l_beta_gamma], dim=1)
        logits_beta_e = torch.cat([l_pos_beta_e, l_neg_beta_e, l_aplha_beta_e, l_beta_gamma_e], dim=1)
        logits_beta_ed = torch.cat([l_pos_beta_ed, l_neg_beta_ed, l_aplha_beta_ed, l_beta_gamma_ed], dim=1)

        logits_gamma = torch.cat([l_pos_gamma, l_neg_gamma, l_aplha_gamma, l_beta_gamma], dim=1)
        logits_gamma_e = torch.cat([l_pos_gamma_e, l_neg_gamma_e, l_aplha_gamma_e, l_beta_gamma_e], dim=1)
        logits_gamma_ed = torch.cat([l_pos_gamma_ed, l_neg_gamma_ed, l_aplha_gamma_ed, l_beta_gamma_ed], dim=1)

        # apply temperature
        logits_alpha /= self.T
        logits_alpha_e /= self.T
        logits_alpha_ed /= self.T

        logits_beta /= self.T
        logits_beta_e /= self.T
        logits_beta_ed /= self.T

        logits_gamma /= self.T
        logits_gamma_e /= self.T
        logits_gamma_ed /= self.T

        labels_ddm_alpha = logits_alpha.clone().detach()
        labels_ddm_alpha = torch.softmax(labels_ddm_alpha, dim=1)
        labels_ddm_alpha = labels_ddm_alpha.detach()

        labels_ddm_beta = logits_beta.clone().detach()
        labels_ddm_beta = torch.softmax(labels_ddm_beta, dim=1)
        labels_ddm_beta = labels_ddm_beta.detach()

        labels_ddm_gamma = logits_gamma.clone().detach()
        labels_ddm_gamma = torch.softmax(labels_ddm_gamma, dim=1)
        labels_ddm_gamma = labels_ddm_gamma.detach()

        # labels: positive key indicators
        labels = torch.zeros(logits_alpha.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_alpha(k_alpha)
        self._dequeue_and_enqueue_beta(k_beta)
        self._dequeue_and_enqueue_gamma(k_gamma)

        return logits_alpha, logits_beta, logits_gamma, labels, \
               labels_ddm_alpha, labels_ddm_beta, labels_ddm_gamma, \
               logits_alpha_e, logits_beta_e, logits_gamma_e, logits_alpha_ed, logits_beta_ed, logits_gamma_ed

    def ensemble_training(self, im_q_alpha, im_k_alpha, im_q_alpha_extreme, topk=1, context=True):
        (im_q_alpha, im_q_beta, im_q_gamma) = (im_q_alpha[:, 0:3], im_q_alpha[:, 3:6], im_q_alpha[:, 6:9])
        (im_q_alpha_extreme, im_q_beta_extreme, im_q_gamma_extreme) = (im_q_alpha_extreme[:, 0:3],
                                                                       im_q_alpha_extreme[:, 3:6],
                                                                       im_q_alpha_extreme[:, 6:9])
        (im_k_alpha, im_k_beta, im_k_gamma) = (im_k_alpha[:, 0:3], im_k_alpha[:, 3:6], im_k_alpha[:, 6:9])

        q_alpha = self.encoder_q_alpha(im_q_alpha)  # queries: NxC
        q_alpha = F.normalize(q_alpha, dim=1)
        q_alpha_e, q_alpha_ed = self.encoder_q_alpha(im_q_alpha_extreme, drop=True)
        q_alpha_e = F.normalize(q_alpha_e, dim=1)
        q_alpha_ed = F.normalize(q_alpha_ed, dim=1)

        q_beta = self.encoder_q_beta(im_q_beta)
        q_beta = F.normalize(q_beta, dim=1)
        q_beta_e, q_beta_ed = self.encoder_q_beta(im_q_beta_extreme, drop=True)
        q_beta_e = F.normalize(q_beta_e, dim=1)
        q_beta_ed = F.normalize(q_beta_ed, dim=1)

        q_gamma = self.encoder_q_gamma(im_q_gamma)
        q_gamma = F.normalize(q_gamma, dim=1)
        q_gamma_e, q_gamma_ed = self.encoder_q_gamma(im_q_gamma_extreme, drop=True)
        q_gamma_e = F.normalize(q_gamma_e, dim=1)
        q_gamma_ed = F.normalize(q_gamma_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_alpha()  # update the key encoder
            self._momentum_update_key_encoder_beta()
            self._momentum_update_key_encoder_gamma()
            # self._momentum_update_key_encoder()

            k_alpha = self.encoder_k_alpha(im_k_alpha)  # keys: NxC
            k_alpha = F.normalize(k_alpha, dim=1)

            k_beta = self.encoder_k_beta(im_k_beta)
            k_beta = F.normalize(k_beta, dim=1)

            k_gamma = self.encoder_k_gamma(im_k_gamma)
            k_gamma = F.normalize(k_gamma, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_alpha = torch.einsum('nc,nc->n', [q_alpha, k_alpha]).unsqueeze(-1)
        l_neg_alpha = torch.einsum('nc,ck->nk', [q_alpha, self.queue_alpha.clone().detach()])
        l_pos_alpha_e = torch.einsum('nc,nc->n', [q_alpha_e, k_alpha]).unsqueeze(-1)
        l_neg_alpha_e = torch.einsum('nc,ck->nk', [q_alpha_e, self.queue_alpha.clone().detach()])
        l_pos_alpha_ed = torch.einsum('nc,nc->n', [q_alpha_ed, k_alpha]).unsqueeze(-1)
        l_neg_alpha_ed = torch.einsum('nc,ck->nk', [q_alpha_ed, self.queue_alpha.clone().detach()])

        l_pos_beta = torch.einsum('nc,nc->n', [q_beta, k_beta]).unsqueeze(-1)
        l_neg_beta = torch.einsum('nc,ck->nk', [q_beta, self.queue_beta.clone().detach()])
        l_pos_beta_e = torch.einsum('nc,nc->n', [q_beta_e, k_beta]).unsqueeze(-1)
        l_neg_beta_e = torch.einsum('nc,ck->nk', [q_beta_e, self.queue_beta.clone().detach()])
        l_pos_beta_ed = torch.einsum('nc,nc->n', [q_beta_ed, k_beta]).unsqueeze(-1)
        l_neg_beta_ed = torch.einsum('nc,ck->nk', [q_beta_ed, self.queue_beta.clone().detach()])

        l_pos_gamma = torch.einsum('nc,nc->n', [q_gamma, k_gamma]).unsqueeze(-1)
        l_neg_gamma = torch.einsum('nc,ck->nk', [q_gamma, self.queue_gamma.clone().detach()])
        l_pos_gamma_e = torch.einsum('nc,nc->n', [q_gamma_e, k_gamma]).unsqueeze(-1)
        l_neg_gamma_e = torch.einsum('nc,ck->nk', [q_gamma_e, self.queue_gamma.clone().detach()])
        l_pos_gamma_ed = torch.einsum('nc,nc->n', [q_gamma_ed, k_gamma]).unsqueeze(-1)
        l_neg_gamma_ed = torch.einsum('nc,ck->nk', [q_gamma_ed, self.queue_gamma.clone().detach()])

        l_ensemble = (l_neg_alpha + l_neg_beta + l_neg_gamma) / 3.
        l_ensemble_e = (l_neg_alpha_e + l_neg_beta_e + l_neg_gamma_e) / 3.
        l_ensemble_ed = (l_neg_alpha_ed + l_neg_beta_ed + l_neg_gamma_ed) / 3.

        if context:
            l_context_j = torch.einsum('nk,nk->nk', [l_neg_alpha, l_ensemble])
            l_context_je = torch.einsum('nk,nk->nk', [l_neg_alpha_e, l_ensemble_e])
            l_context_jed = torch.einsum('nk,nk->nk', [l_neg_alpha_ed, l_ensemble_ed])

            l_context_m = torch.einsum('nk,nk->nk', [l_neg_beta, l_ensemble])
            l_context_me = torch.einsum('nk,nk->nk', [l_neg_beta_e, l_ensemble_e])
            l_context_med = torch.einsum('nk,nk->nk', [l_neg_beta_ed, l_ensemble_ed])

            l_context_b = torch.einsum('nk,nk->nk', [l_neg_gamma, l_ensemble])
            l_context_be = torch.einsum('nk,nk->nk', [l_neg_gamma_e, l_ensemble_e])
            l_context_bed = torch.einsum('nk,nk->nk', [l_neg_gamma_ed, l_ensemble_ed])

            logits_alpha = torch.cat([l_pos_alpha, l_neg_alpha, l_context_j], dim=1)
            logits_alpha_e = torch.cat([l_pos_alpha_e, l_neg_alpha_e, l_context_je], dim=1)
            logits_alpha_ed = torch.cat([l_pos_alpha_ed, l_neg_alpha_ed, l_context_jed], dim=1)

            logits_beta = torch.cat([l_pos_beta, l_neg_beta, l_context_m], dim=1)
            logits_beta_e = torch.cat([l_pos_beta_e, l_neg_beta_e, l_context_me], dim=1)
            logits_beta_ed = torch.cat([l_pos_beta_ed, l_neg_beta_ed, l_context_med], dim=1)

            logits_gamma = torch.cat([l_pos_gamma, l_neg_gamma, l_context_b], dim=1)
            logits_gamma_e = torch.cat([l_pos_gamma_e, l_neg_gamma_e, l_context_be], dim=1)
            logits_gamma_ed = torch.cat([l_pos_gamma_ed, l_neg_gamma_ed, l_context_bed], dim=1)

        else:
            # logits: Nx(1+K)
            logits_alpha = torch.cat([l_pos_alpha, l_neg_alpha], dim=1)
            logits_alpha_e = torch.cat([l_pos_alpha_e, l_neg_alpha_e], dim=1)
            logits_alpha_ed = torch.cat([l_pos_alpha_ed, l_neg_alpha_ed], dim=1)

            logits_beta = torch.cat([l_pos_beta, l_neg_beta], dim=1)
            logits_beta_e = torch.cat([l_pos_beta_e, l_neg_beta_e], dim=1)
            logits_beta_ed = torch.cat([l_pos_beta_ed, l_neg_beta_ed], dim=1)

            logits_gamma = torch.cat([l_pos_gamma, l_neg_gamma], dim=1)
            logits_gamma_e = torch.cat([l_pos_gamma_e, l_neg_gamma_e], dim=1)
            logits_gamma_ed = torch.cat([l_pos_gamma_ed, l_neg_gamma_ed], dim=1)

        # apply temperature
        logits_alpha /= self.T
        logits_alpha_e /= self.T
        logits_alpha_ed /= self.T

        logits_beta /= self.T
        logits_beta_e /= self.T
        logits_beta_ed /= self.T

        logits_gamma /= self.T
        logits_gamma_e /= self.T
        logits_gamma_ed /= self.T

        labels_ddm_alpha = logits_alpha.clone().detach()
        labels_ddm_alpha = torch.softmax(labels_ddm_alpha, dim=1)
        labels_ddm_alpha = labels_ddm_alpha.detach()

        labels_ddm_beta = logits_beta.clone().detach()
        labels_ddm_beta = torch.softmax(labels_ddm_beta, dim=1)
        labels_ddm_beta = labels_ddm_beta.detach()

        labels_ddm_gamma = logits_gamma.clone().detach()
        labels_ddm_gamma = torch.softmax(labels_ddm_gamma, dim=1)
        labels_ddm_gamma = labels_ddm_gamma.detach()

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

        self._dequeue_and_enqueue_alpha(k_alpha)
        self._dequeue_and_enqueue_beta(k_beta)
        self._dequeue_and_enqueue_gamma(k_gamma)

        return logits_alpha, logits_alpha_e, logits_alpha_ed, \
               logits_beta, logits_beta_e, logits_beta_ed, \
               logits_gamma, logits_gamma_e, logits_gamma_ed, \
               pos_mask_j, pos_mask_m, pos_mask_b, labels_ddm_alpha, labels_ddm_beta, labels_ddm_gamma

    def cross_training(self, im_q_alpha, im_k_alpha, im_q_alpha_extreme, topk=1):
        im_q_beta = torch.zeros_like(im_q_alpha)
        im_q_beta[:, :, :-1, :, :] = im_q_alpha[:, :, 1:, :, :] - im_q_alpha[:, :, :-1, :, :]

        im_k_beta = torch.zeros_like(im_k_alpha)
        im_k_beta[:, :, :-1, :, :] = im_k_alpha[:, :, 1:, :, :] - im_k_alpha[:, :, :-1, :, :]

        im_q_gamma = torch.zeros_like(im_q_alpha)
        im_k_gamma = torch.zeros_like(im_k_alpha)

        im_q_beta_extreme = torch.zeros_like(im_q_alpha_extreme)
        im_q_beta_extreme[:, :, :-1, :, :] = im_q_alpha_extreme[:, :, 1:, :, :] - im_q_alpha_extreme[:, :, :-1, :,
                                                                                  :]
        im_q_gamma_extreme = torch.zeros_like(im_q_alpha_extreme)
        for v1, v2 in self.gamma:
            im_q_gamma_extreme[:, :, :, v1 - 1, :] = im_q_alpha_extreme[:, :, :, v1 - 1, :] - im_q_alpha_extreme[:,
                                                                                              :, :, v2 - 1, :]
        for v1, v2 in self.gamma:
            im_q_gamma[:, :, :, v1 - 1, :] = im_q_alpha[:, :, :, v1 - 1, :] - im_q_alpha[:, :, :, v2 - 1, :]
            im_k_gamma[:, :, :, v1 - 1, :] = im_k_alpha[:, :, :, v1 - 1, :] - im_k_alpha[:, :, :, v2 - 1, :]

        q_alpha = self.encoder_q_alpha(im_q_alpha)  # queries: NxC
        q_alpha = F.normalize(q_alpha, dim=1)
        q_alpha_e, q_alpha_ed = self.encoder_q_alpha(im_q_alpha_extreme, drop=True)
        q_alpha_e = F.normalize(q_alpha_e, dim=1)
        q_alpha_ed = F.normalize(q_alpha_ed, dim=1)

        q_beta = self.encoder_q_beta(im_q_beta)
        q_beta = F.normalize(q_beta, dim=1)
        q_beta_e, q_beta_ed = self.encoder_q_beta(im_q_beta_extreme, drop=True)
        q_beta_e = F.normalize(q_beta_e, dim=1)
        q_beta_ed = F.normalize(q_beta_ed, dim=1)

        q_gamma = self.encoder_q_gamma(im_q_gamma)
        q_gamma = F.normalize(q_gamma, dim=1)
        q_gamma_e, q_gamma_ed = self.encoder_q_gamma(im_q_gamma_extreme, drop=True)
        q_gamma_e = F.normalize(q_gamma_e, dim=1)
        q_gamma_ed = F.normalize(q_gamma_ed, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_alpha()  # update the key encoder
            self._momentum_update_key_encoder_beta()
            self._momentum_update_key_encoder_gamma()
            # self._momentum_update_key_encoder()

            k_alpha = self.encoder_k_alpha(im_k_alpha)  # keys: NxC
            k_alpha = F.normalize(k_alpha, dim=1)

            k_beta = self.encoder_k_beta(im_k_beta)
            k_beta = F.normalize(k_beta, dim=1)

            k_gamma = self.encoder_k_gamma(im_k_gamma)
            k_gamma = F.normalize(k_gamma, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        l_pos_alpha = torch.einsum('nc,nc->n', [q_alpha, k_alpha]).unsqueeze(-1)
        l_neg_alpha = torch.einsum('nc,ck->nk', [q_alpha, self.queue_alpha.clone().detach()])
        l_pos_alpha_e = torch.einsum('nc,nc->n', [q_alpha_e, k_alpha]).unsqueeze(-1)
        l_neg_alpha_e = torch.einsum('nc,ck->nk', [q_alpha_e, self.queue_alpha.clone().detach()])
        l_pos_alpha_ed = torch.einsum('nc,nc->n', [q_alpha_ed, k_alpha]).unsqueeze(-1)
        l_neg_alpha_ed = torch.einsum('nc,ck->nk', [q_alpha_ed, self.queue_alpha.clone().detach()])

        l_pos_beta = torch.einsum('nc,nc->n', [q_beta, k_beta]).unsqueeze(-1)
        l_neg_beta = torch.einsum('nc,ck->nk', [q_beta, self.queue_beta.clone().detach()])
        l_pos_beta_e = torch.einsum('nc,nc->n', [q_beta_e, k_beta]).unsqueeze(-1)
        l_neg_beta_e = torch.einsum('nc,ck->nk', [q_beta_e, self.queue_beta.clone().detach()])
        l_pos_beta_ed = torch.einsum('nc,nc->n', [q_beta_ed, k_beta]).unsqueeze(-1)
        l_neg_beta_ed = torch.einsum('nc,ck->nk', [q_beta_ed, self.queue_beta.clone().detach()])

        l_pos_gamma = torch.einsum('nc,nc->n', [q_gamma, k_gamma]).unsqueeze(-1)
        l_neg_gamma = torch.einsum('nc,ck->nk', [q_gamma, self.queue_gamma.clone().detach()])
        l_pos_gamma_e = torch.einsum('nc,nc->n', [q_gamma_e, k_gamma]).unsqueeze(-1)
        l_neg_gamma_e = torch.einsum('nc,ck->nk', [q_gamma_e, self.queue_gamma.clone().detach()])
        l_pos_gamma_ed = torch.einsum('nc,nc->n', [q_gamma_ed, k_gamma]).unsqueeze(-1)
        l_neg_gamma_ed = torch.einsum('nc,ck->nk', [q_gamma_ed, self.queue_gamma.clone().detach()])

        # logits: Nx(1+K)
        logits_alpha = torch.cat([l_pos_alpha, l_neg_alpha], dim=1)
        logits_alpha_e = torch.cat([l_pos_alpha_e, l_neg_alpha_e], dim=1)
        logits_alpha_ed = torch.cat([l_pos_alpha_ed, l_neg_alpha_ed], dim=1)

        logits_beta = torch.cat([l_pos_beta, l_neg_beta], dim=1)
        logits_beta_e = torch.cat([l_pos_beta_e, l_neg_beta_e], dim=1)
        logits_beta_ed = torch.cat([l_pos_beta_ed, l_neg_beta_ed], dim=1)

        logits_gamma = torch.cat([l_pos_gamma, l_neg_gamma], dim=1)
        logits_gamma_e = torch.cat([l_pos_gamma_e, l_neg_gamma_e], dim=1)
        logits_gamma_ed = torch.cat([l_pos_gamma_ed, l_neg_gamma_ed], dim=1)

        # apply temperature
        logits_alpha /= self.T
        logits_alpha_e /= self.T
        logits_alpha_ed /= self.T

        logits_beta /= self.T
        logits_beta_e /= self.T
        logits_beta_ed /= self.T

        logits_gamma /= self.T
        logits_gamma_e /= self.T
        logits_gamma_ed /= self.T

        labels_ddm_alpha = logits_alpha.clone().detach()
        labels_ddm_alpha = torch.softmax(labels_ddm_alpha, dim=1)
        labels_ddm_alpha = labels_ddm_alpha.detach()

        labels_ddm_beta = logits_beta.clone().detach()
        labels_ddm_beta = torch.softmax(labels_ddm_beta, dim=1)
        labels_ddm_beta = labels_ddm_beta.detach()

        labels_ddm_gamma = logits_gamma.clone().detach()
        labels_ddm_gamma = torch.softmax(labels_ddm_gamma, dim=1)
        labels_ddm_gamma = labels_ddm_gamma.detach()

        _, topkdix = torch.topk(l_neg_alpha, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_alpha_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_alpha_ed, topk, dim=1)
        topk_onehot_alpha = torch.zeros_like(l_neg_alpha)
        topk_onehot_alpha.scatter_(1, topkdix, 1)
        topk_onehot_alpha.scatter_(1, topkdix_e, 1)
        topk_onehot_alpha.scatter_(1, topkdix_ed, 1)

        _, topkdix = torch.topk(l_neg_beta, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_beta_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_beta_ed, topk, dim=1)
        topk_onehot_beta = torch.zeros_like(l_neg_beta)
        topk_onehot_beta.scatter_(1, topkdix, 1)
        topk_onehot_beta.scatter_(1, topkdix_e, 1)
        topk_onehot_beta.scatter_(1, topkdix_ed, 1)

        _, topkdix = torch.topk(l_neg_gamma, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_gamma_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_gamma_ed, topk, dim=1)
        topk_onehot_gamma = torch.zeros_like(l_neg_gamma)
        topk_onehot_gamma.scatter_(1, topkdix, 1)
        topk_onehot_gamma.scatter_(1, topkdix_e, 1)
        topk_onehot_gamma.scatter_(1, topkdix_ed, 1)

        pos_mask_alpha = torch.cat([torch.ones(topk_onehot_alpha.size(0), 1).cuda(), topk_onehot_alpha], dim=1)
        pos_mask_beta = torch.cat([torch.ones(topk_onehot_beta.size(0), 1).cuda(), topk_onehot_beta], dim=1)
        pos_mask_gamma = torch.cat([torch.ones(topk_onehot_gamma.size(0), 1).cuda(), topk_onehot_gamma], dim=1)

        self._dequeue_and_enqueue_alpha(k_alpha)
        self._dequeue_and_enqueue_beta(k_beta)
        self._dequeue_and_enqueue_gamma(k_gamma)

        return logits_alpha, logits_alpha_e, logits_alpha_ed, \
               logits_beta, logits_beta_e, logits_beta_ed, \
               logits_gamma, logits_gamma_e, logits_gamma_ed, \
               pos_mask_alpha, pos_mask_beta, pos_mask_gamma, \
               labels_ddm_alpha, labels_ddm_beta, labels_ddm_gamma
