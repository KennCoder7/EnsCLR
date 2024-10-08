import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.utils.psgraph import PSGraph
from net.utils.ps_graph import psGraph
from net.att_drop import Simam_Drop


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, wo_fc=False, part_modals=None,
                 fusion_method=None, **kwargs):
        super().__init__()
        self.wo_fc = wo_fc
        # load graph
        self.graph = Graph(**graph_args)
        # self.graph = PSGraph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)
        if self.wo_fc:
            z = x.clone()
        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        if self.wo_fc:
            return x, z
        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 se=None):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class STGCNModel(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, part='body', **kwargs):
        super().__init__()

        # # load graph
        # self.graph = psGraph()
        # if part == "hand":
        #     A = self.graph.hand_A
        #     num_point = 13
        # elif part == "leg":
        #     A = self.graph.leg_A
        #     num_point = 9
        # elif part == "body":
        #     A = self.graph.A
        #     num_point = 25
        # elif part == 'body_base':
        #     self.graph = Graph(**graph_args)
        #     A = self.graph.A
        #     num_point = 25
        # else:
        #     raise
        # A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)
        self.graph = Graph(**graph_args)
        # self.graph = PSGraph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        num_point = 25
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)
        self.dropout = Simam_Drop(num_point=num_point, keep_prob=0.7)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x, drop=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        if drop:
            y = self.dropout(x)
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            # global pooling
            y = F.avg_pool2d(y, y.size()[2:])
            y = y.view(N, M, -1).mean(dim=1)

            # prediction
            y = self.fc(y)
            y = y.view(y.size(0), -1)

            return x, y
        else:
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


class STGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        # self.graph = Graph(**graph_args)
        self.graph = PSGraph(part='body', **graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        # prediction
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


class MMTM(nn.Module):
    """
    args:
        modal_types: list e.g. ['rgb', 'skl]
        modal_channels: dict e.g. {'rgb': 512, 'skl': 256}
        ratio: int
        e.g.
        input: x, y, c=channels
        mmtm = MMTM(['x', 'y'], {'x': c, 'y': c})
        z = {'x': x, 'y': y}
        z = mmtm(z)
        z = z['x']+z['y']
    """

    def __init__(self, modal_types, modal_channels, ratio=2, device='cuda'):
        super().__init__()
        self.modal_types = modal_types
        self.modal_channels = modal_channels
        assert self.modal_types == list(self.modal_channels.keys())
        dim = sum(self.modal_channels.values())
        dim_out = int(dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out).to(device)
        self.fc = nn.ModuleDict({})
        for modal in modal_types:
            self.fc[modal] = nn.Linear(dim_out, self.modal_channels[modal]).to(device)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert list(x.keys()) == self.modal_types
        squeeze_array = []
        for tensor in x.values():
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        modal_out = {}
        for modal in self.modal_types:
            modal_out[modal] = self.fc[modal](excitation)
            modal_out[modal] = self.sigmoid(modal_out[modal])
            dim_diff = len(x[modal].shape) - len(modal_out[modal].shape)
            modal_out[modal] = modal_out[modal].view(modal_out[modal].shape + (1,) * dim_diff)
            x[modal] = x[modal] * modal_out[modal]

        return x



class FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature='', **kwargs):
        super().__init__()

        # load hand graph
        self.graph_hand = PSGraph(part='hand', **graph_args)
        A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_hand', A_hand)
        # build hand networks
        spatial_kernel_size = A_hand.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_hand = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        # load leg graph
        self.graph_leg = PSGraph(part='leg', **graph_args)
        A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_leg', A_leg)
        # build leg networks
        spatial_kernel_size = A_leg.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_leg = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_hand = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_hand.size()))
                for i in self.st_gcn_networks_hand
            ])
            self.edge_importance_leg = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_leg.size()))
                for i in self.st_gcn_networks_leg
            ])
        else:
            self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
            self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        # self.fusion = MMTM(modal_types=['hand', 'leg'], modal_channels={'hand': hidden_dim, 'leg': hidden_dim})
        self.fusion = nn.ModuleDict({
            '7': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 2, 'leg': hidden_channels * 2}),
            '8': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
            '9': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
        })
        self.fc = nn.Linear(hidden_dim * 2, num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in ['', 'before fusion', 'after fusion']
        if self.return_each_feature == 'before fusion':
            self.fc_hand = nn.Linear(hidden_channels * 2, num_class)
            self.fc_leg = nn.Linear(hidden_channels * 2, num_class)
        elif self.return_each_feature == 'after fusion':
            self.fc_hand = nn.Linear(hidden_dim, num_class)
            self.fc_leg = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        assert x.keys() == {'hand', 'leg'}

        # data normalization
        for part in ['hand', 'leg']:
            N, C, T, V, M = x[part].size()
            x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
            x[part] = x[part].view(N * M, V * C, T)
            if part == 'hand':
                x[part] = self.data_bn_hand(x[part])
            else:
                x[part] = self.data_bn_leg(x[part])
            x[part] = x[part].view(N, M, V, C, T)
            x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
            x[part] = x[part].view(N * M, C, T, V)

        # forward
        gcn_layer_n = 1  # total = 10
        y_hand = None
        y_leg = None
        for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                      self.edge_importance_hand,
                                                                      self.st_gcn_networks_leg,
                                                                      self.edge_importance_leg):
            if 6 < gcn_layer_n < 10:
                # start fusion from layer 7 to 9
                if self.return_each_feature == 'before fusion' and y_hand is None:
                    # save the output of layer 6
                    y_hand = x['hand'].clone()
                    y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                    y_hand = y_hand.view(N, M, -1).mean(dim=1)
                    y_hand = self.fc_hand(y_hand)
                    y_hand = y_hand.view(y_hand.size(0), -1)
                    y_leg = x['leg'].clone()
                    y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                    y_leg = y_leg.view(N, M, -1).mean(dim=1)
                    y_leg = self.fc_leg(y_leg)
                    y_leg = y_leg.view(y_leg.size(0), -1)
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                x = self.fusion[str(gcn_layer_n)](x)
            else:
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
            gcn_layer_n += 1

        for part in ['hand', 'leg']:
            x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
            x[part] = x[part].view(N, M, -1).mean(dim=1)

        if self.return_each_feature == 'after fusion':
            y_hand = x['hand'].clone()
            y_hand = self.fc_hand(y_hand)
            y_hand = y_hand.view(y_hand.size(0), -1)
            y_leg = x['leg'].clone()
            y_leg = self.fc_leg(y_leg)
            y_leg = y_leg.view(y_leg.size(0), -1)

        x = torch.cat([x['hand'], x['leg']], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if self.return_each_feature != '':
            return x, y_hand, y_leg
        else:
            return x


class FUSION_STGCN_v2(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, **kwargs):
        super().__init__()

        # load hand graph
        self.graph_hand = PSGraph(part='hand', **graph_args)
        A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_hand', A_hand)
        # build hand networks
        spatial_kernel_size = A_hand.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_hand = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        # load leg graph
        self.graph_leg = PSGraph(part='leg', **graph_args)
        A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_leg', A_leg)
        # build leg networks
        spatial_kernel_size = A_leg.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_leg = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_hand = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_hand.size()))
                for i in self.st_gcn_networks_hand
            ])
            self.edge_importance_leg = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_leg.size()))
                for i in self.st_gcn_networks_leg
            ])
        else:
            self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
            self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        # self.fusion = MMTM(modal_types=['hand', 'leg'], modal_channels={'hand': hidden_dim, 'leg': hidden_dim})
        self.fusion = nn.ModuleDict({
            '7': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 2, 'leg': hidden_channels * 2}),
            '8': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
            '9': MMTM(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
        })
        self.fc = nn.Linear(hidden_dim * 2, num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_hand_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_leg_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_hand_af = nn.Linear(hidden_dim, num_class)
            self.fc_leg_af = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        assert x.keys() == {'hand', 'leg'}

        # data normalization
        for part in ['hand', 'leg']:
            N, C, T, V, M = x[part].size()
            x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
            x[part] = x[part].view(N * M, V * C, T)
            if part == 'hand':
                x[part] = self.data_bn_hand(x[part])
            else:
                x[part] = self.data_bn_leg(x[part])
            x[part] = x[part].view(N, M, V, C, T)
            x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
            x[part] = x[part].view(N * M, C, T, V)

        # forward
        gcn_layer_n = 1  # total = 10
        y_hand_bf = None
        y_leg_bf = None
        for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                      self.edge_importance_hand,
                                                                      self.st_gcn_networks_leg,
                                                                      self.edge_importance_leg):
            if 6 < gcn_layer_n < 10:
                # start fusion from layer 7 to 9
                if y_hand_bf is None and self.return_each_feature:
                    # save the output of layer 6
                    y_hand = x['hand'].clone()
                    y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                    y_hand = y_hand.view(N, M, -1).mean(dim=1)
                    y_hand = self.fc_hand_bf(y_hand)
                    y_hand_bf = y_hand.view(y_hand.size(0), -1)
                    y_leg = x['leg'].clone()
                    y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                    y_leg = y_leg.view(N, M, -1).mean(dim=1)
                    y_leg = self.fc_leg_bf(y_leg)
                    y_leg_bf = y_leg.view(y_leg.size(0), -1)
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                x = self.fusion[str(gcn_layer_n)](x)
            else:
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
            gcn_layer_n += 1

        for part in ['hand', 'leg']:
            x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
            x[part] = x[part].view(N, M, -1).mean(dim=1)

        if self.return_each_feature:
            y_hand = x['hand'].clone()
            y_hand = self.fc_hand_af(y_hand)
            y_hand_af = y_hand.view(y_hand.size(0), -1)
            y_leg = x['leg'].clone()
            y_leg = self.fc_leg_af(y_leg)
            y_leg_af = y_leg.view(y_leg.size(0), -1)

        x = torch.cat([x['hand'], x['leg']], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if self.return_each_feature:
            return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
        else:
            return x


class FUSION_STGCN_aimclr(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, **kwargs):
        super().__init__()

        # load hand graph
        self.graph_hand = PSGraph(part='hand', **graph_args)
        A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_hand', A_hand)
        # build hand networks
        spatial_kernel_size = A_hand.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_hand = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        # load leg graph
        self.graph_leg = PSGraph(part='leg', **graph_args)
        A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_leg', A_leg)
        # build leg networks
        spatial_kernel_size = A_leg.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_leg = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_hand = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_hand.size()))
                for i in self.st_gcn_networks_hand
            ])
            self.edge_importance_leg = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_leg.size()))
                for i in self.st_gcn_networks_leg
            ])
        else:
            self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
            self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        # self.fusion = MMTM(modal_types=['hand', 'leg'], modal_channels={'hand': hidden_dim, 'leg': hidden_dim})
        self.fusion = nn.ModuleDict({
            '7': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 2, 'leg': hidden_channels * 2}),
            '8': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
            '9': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
        })
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        self.dropout = nn.ModuleDict({
            'hand': Simam_Drop(num_point=16, keep_prob=0.7),
            'leg': Simam_Drop(num_point=10, keep_prob=0.7)
        })
        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_hand_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_leg_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_hand_af = nn.Linear(hidden_dim, num_class)
            self.fc_leg_af = nn.Linear(hidden_dim, num_class)

    def forward(self, x, drop=False):
        assert x.keys() == {'hand', 'leg'}

        # data normalization
        for part in ['hand', 'leg']:
            N, C, T, V, M = x[part].size()
            x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
            x[part] = x[part].view(N * M, V * C, T)
            if part == 'hand':
                x[part] = self.data_bn_hand(x[part])
            else:
                x[part] = self.data_bn_leg(x[part])
            x[part] = x[part].view(N, M, V, C, T)
            x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
            x[part] = x[part].view(N * M, C, T, V)

        # forward
        gcn_layer_n = 1  # total = 10
        y_hand_bf = None
        y_leg_bf = None
        dp = {}
        for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                      self.edge_importance_hand,
                                                                      self.st_gcn_networks_leg,
                                                                      self.edge_importance_leg):
            if 6 < gcn_layer_n < 10:
                # start fusion from layer 7 to 9
                if y_hand_bf is None and self.return_each_feature:
                    # save the output of layer 6
                    y_hand = x['hand'].clone()
                    y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                    y_hand = y_hand.view(N, M, -1).mean(dim=1)
                    y_hand = self.fc_hand_bf(y_hand)
                    y_hand_bf = y_hand.view(y_hand.size(0), -1)
                    y_leg = x['leg'].clone()
                    y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                    y_leg = y_leg.view(N, M, -1).mean(dim=1)
                    y_leg = self.fc_leg_bf(y_leg)
                    y_leg_bf = y_leg.view(y_leg.size(0), -1)
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                x = self.fusion[str(gcn_layer_n)](x)
            else:
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
            gcn_layer_n += 1

        for part in ['hand', 'leg']:
            if drop:
                dp[part] = self.dropout[part](x[part])
                dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
                dp[part] = dp[part].view(N, M, -1).mean(dim=1)
            x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
            x[part] = x[part].view(N, M, -1).mean(dim=1)

        if self.return_each_feature:
            y_hand = x['hand'].clone()
            y_hand = self.fc_hand_af(y_hand)
            y_hand_af = y_hand.view(y_hand.size(0), -1)
            y_leg = x['leg'].clone()
            y_leg = self.fc_leg_af(y_leg)
            y_leg_af = y_leg.view(y_leg.size(0), -1)

        x = torch.cat([x['hand'], x['leg']], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        if drop:
            dp = torch.cat([dp['hand'], dp['leg']], dim=1)
            dp = self.fc(dp)
            dp = dp.view(dp.size(0), -1)
        if self.return_each_feature:
            return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
        else:
            if drop:
                return x, dp
            return x


class RAWFUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature='', **kwargs):
        super().__init__()
        # load hand graph
        self.graph_hand = PSGraph(part='hand', **graph_args)
        A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_hand', A_hand)
        # build hand networks
        spatial_kernel_size = A_hand.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_hand = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        # load leg graph
        self.graph_leg = PSGraph(part='leg', **graph_args)
        A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_leg', A_leg)
        # build leg networks
        spatial_kernel_size = A_leg.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_leg = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_hand = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_hand.size()))
                for i in self.st_gcn_networks_hand
            ])
            self.edge_importance_leg = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_leg.size()))
                for i in self.st_gcn_networks_leg
            ])
        else:
            self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
            self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        self.fc = nn.Linear(hidden_dim * 2, num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in ['', 'before fusion', 'after fusion']
        if self.return_each_feature == 'before fusion':
            self.fc_hand = nn.Linear(hidden_channels * 2, num_class)
            self.fc_leg = nn.Linear(hidden_channels * 2, num_class)
        elif self.return_each_feature == 'after fusion':
            self.fc_hand = nn.Linear(hidden_dim, num_class)
            self.fc_leg = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        assert x.keys() == {'hand', 'leg'}

        # data normalization
        for part in ['hand', 'leg']:
            N, C, T, V, M = x[part].size()
            x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
            x[part] = x[part].view(N * M, V * C, T)
            if part == 'hand':
                x[part] = self.data_bn_hand(x[part])
            else:
                x[part] = self.data_bn_leg(x[part])
            x[part] = x[part].view(N, M, V, C, T)
            x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
            x[part] = x[part].view(N * M, C, T, V)

        # forward
        gcn_layer_n = 1  # total = 10
        y_hand = None
        y_leg = None
        for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                      self.edge_importance_hand,
                                                                      self.st_gcn_networks_leg,
                                                                      self.edge_importance_leg):
            if 6 < gcn_layer_n < 10:
                # start fusion from layer 7 to 9
                if self.return_each_feature == 'before fusion' and y_hand is None:
                    # save the output of layer 6
                    y_hand = x['hand'].clone()
                    y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                    y_hand = y_hand.view(N, M, -1).mean(dim=1)
                    y_hand = self.fc_hand(y_hand)
                    y_hand = y_hand.view(y_hand.size(0), -1)
                    y_leg = x['leg'].clone()
                    y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                    y_leg = y_leg.view(N, M, -1).mean(dim=1)
                    y_leg = self.fc_leg(y_leg)
                    y_leg = y_leg.view(y_leg.size(0), -1)
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                # x = self.fusion[str(gcn_layer_n)](x)
            else:
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
            gcn_layer_n += 1

        for part in ['hand', 'leg']:
            x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
            x[part] = x[part].view(N, M, -1).mean(dim=1)

        if self.return_each_feature == 'after fusion':
            y_hand = x['hand'].clone()
            y_hand = self.fc_hand(y_hand)
            y_hand = y_hand.view(y_hand.size(0), -1)
            y_leg = x['leg'].clone()
            y_leg = self.fc_leg(y_leg)
            y_leg = y_leg.view(y_leg.size(0), -1)

        x = torch.cat([x['hand'], x['leg']], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if self.return_each_feature != '':
            return x, y_hand, y_leg
        else:
            return x


class CMAF(nn.Module):
    """
                args:
                    modal_types: list e.g. ['rgb', 'skl]
                    modal_channels: dict e.g. {'rgb': 512, 'skl': 256}
            """

    def __init__(self, modal_types, modal_channels, n_heads=8, dropout=0.5, device='cuda'):
        super().__init__()

        self.modal_types = modal_types
        self.modal_channels = modal_channels
        self.n_heads = n_heads
        for c in self.modal_channels:
            assert self.modal_channels[c] % self.n_heads == 0, 'wrong n_heads settings'
        self.fc = nn.ModuleDict({})
        for modal in self.modal_types:
            self.fc[modal] = nn.ModuleDict({
                'k': nn.Linear(self.modal_channels[modal], self.modal_channels[modal]).to(device),
                'q': nn.Linear(self.modal_channels[modal], self.modal_channels[modal]).to(device),
                'v': nn.Linear(self.modal_channels[modal], self.modal_channels[modal]).to(device)
            })

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    # def init_weights(self):
    #     """Initiate the parameters from scratch."""
    #     for modal in self.modal_types:
    #         for m in ['k', 'q', 'v']:
    #             normal_init(self.fc[modal][m], std=0.01)

    def forward(self, x):
        assert list(x.keys()) == self.modal_types
        bs = x[self.modal_types[0]].shape[0]
        dt = {}
        for modal in self.modal_types:
            dt[modal + '_flatten'] = x[modal].view(x[modal].shape[:2] + (-1,))  # (Ca, Na)
            _squeeze = F.adaptive_avg_pool1d(dt[modal + '_flatten'], 1).squeeze()  # (Ca)
            dt[modal] = dict(
                k=self.fc[modal]['k'](_squeeze).unsqueeze(1),  # (1, Ca)
                q=self.fc[modal]['q'](_squeeze).unsqueeze(1),  # (1, Ca)
                v=self.fc[modal]['v'](_squeeze).unsqueeze(1),  # (1, Ca)
            )
            for i in ['k', 'q', 'v']:
                dt[modal][i] = dt[modal][i].view(bs, 1, self.n_heads,  # (n_heads, 1, Ca // n_heads)
                                                 self.modal_channels[modal] // self.n_heads).permute(0, 2, 1, 3)

        for modal in self.modal_types:
            rest_modals = self.modal_types.copy()
            rest_modals.remove(modal)
            attn_lst = []
            for rest_modal in rest_modals:
                # (n_heads, Ca // n_heads, Cb // n_heads)
                co_sim = torch.matmul(dt[modal]['k'].permute(0, 1, 3, 2), dt[rest_modal]['q'])
                co_sim_new = co_sim / torch.sqrt(torch.tensor((self.modal_channels[modal] +
                                                               self.modal_channels[rest_modal]) / 2.))
                co_sim_new = self.softmax(co_sim_new)
                co_sim_new = self.dropout(co_sim_new)
                co_attn = torch.matmul(co_sim_new,
                                       dt[rest_modal]['v'].permute(0, 1, 3, 2))  # (n_heads, Ca // n_heads, 1)
                attn_lst.append(co_attn.view(bs, self.modal_channels[modal], 1))  # (Ca, 1)
            dt[f'E_{modal}'] = torch.mean(torch.stack(attn_lst), dim=0) + dt[f'{modal}_flatten']  # (Ca, Na)
            x[modal] = dt[f'E_{modal}'].view(x[modal].shape)  # (Ca, La, Ha, Wa)

        return x


class FUSION_STGCN_CMAF(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, **kwargs):
        super().__init__()

        # load hand graph
        self.graph_hand = PSGraph(part='hand', **graph_args)
        A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_hand', A_hand)
        # build hand networks
        spatial_kernel_size = A_hand.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_hand = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        # load leg graph
        self.graph_leg = PSGraph(part='leg', **graph_args)
        A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_leg', A_leg)
        # build leg networks
        spatial_kernel_size = A_leg.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks_leg = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance_hand = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_hand.size()))
                for i in self.st_gcn_networks_hand
            ])
            self.edge_importance_leg = nn.ParameterList([
                nn.Parameter(torch.ones(self.A_leg.size()))
                for i in self.st_gcn_networks_leg
            ])
        else:
            self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
            self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        # self.fusion = MMTM(modal_types=['hand', 'leg'], modal_channels={'hand': hidden_dim, 'leg': hidden_dim})
        self.fusion = nn.ModuleDict({
            '7': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 2, 'leg': hidden_channels * 2}),
            '8': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
            '9': CMAF(modal_types=['hand', 'leg'],
                      modal_channels={'hand': hidden_channels * 4, 'leg': hidden_channels * 4}),
        })
        self.fc = nn.Linear(hidden_dim * 2, num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_hand_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_leg_bf = nn.Linear(hidden_channels * 2, num_class)
            self.fc_hand_af = nn.Linear(hidden_dim, num_class)
            self.fc_leg_af = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        assert x.keys() == {'hand', 'leg'}

        # data normalization
        for part in ['hand', 'leg']:
            N, C, T, V, M = x[part].size()
            x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
            x[part] = x[part].view(N * M, V * C, T)
            if part == 'hand':
                x[part] = self.data_bn_hand(x[part])
            else:
                x[part] = self.data_bn_leg(x[part])
            x[part] = x[part].view(N, M, V, C, T)
            x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
            x[part] = x[part].view(N * M, C, T, V)

        # forward
        gcn_layer_n = 1  # total = 10
        y_hand_bf = None
        y_leg_bf = None
        for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                      self.edge_importance_hand,
                                                                      self.st_gcn_networks_leg,
                                                                      self.edge_importance_leg):
            if 6 < gcn_layer_n < 10:
                # start fusion from layer 7 to 9
                if y_hand_bf is None and self.return_each_feature:
                    # save the output of layer 6
                    y_hand = x['hand'].clone()
                    y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                    y_hand = y_hand.view(N, M, -1).mean(dim=1)
                    y_hand = self.fc_hand_bf(y_hand)
                    y_hand_bf = y_hand.view(y_hand.size(0), -1)
                    y_leg = x['leg'].clone()
                    y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                    y_leg = y_leg.view(N, M, -1).mean(dim=1)
                    y_leg = self.fc_leg_bf(y_leg)
                    y_leg_bf = y_leg.view(y_leg.size(0), -1)
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                x = self.fusion[str(gcn_layer_n)](x)
            else:
                x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
            gcn_layer_n += 1

        for part in ['hand', 'leg']:
            x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
            x[part] = x[part].view(N, M, -1).mean(dim=1)

        if self.return_each_feature:
            y_hand = x['hand'].clone()
            y_hand = self.fc_hand_af(y_hand)
            y_hand_af = y_hand.view(y_hand.size(0), -1)
            y_leg = x['leg'].clone()
            y_leg = self.fc_leg_af(y_leg)
            y_leg_af = y_leg.view(y_leg.size(0), -1)

        x = torch.cat([x['hand'], x['leg']], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        if self.return_each_feature:
            return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
        else:
            return x


class PART_FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
                 fusion_method='MMTM', **kwargs):
        super().__init__()
        self.part_modals = part_modals
        self.fusion_method = fusion_method
        assert fusion_method in ['MMTM', 'CMAF', 'NONE']

        if 'hand' in self.part_modals:
            # load hand graph
            self.graph_hand = PSGraph(part='hand', **graph_args)
            A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_hand', A_hand)
            # build hand networks
            spatial_kernel_size = A_hand.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_hand = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_hand = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_hand.size()))
                    for i in self.st_gcn_networks_hand
                ])
            else:
                self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
        if 'leg' in self.part_modals:
            # load leg graph
            self.graph_leg = PSGraph(part='leg', **graph_args)
            A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_leg', A_leg)
            # build leg networks
            spatial_kernel_size = A_leg.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_leg = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_leg = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_leg.size()))
                    for i in self.st_gcn_networks_leg
                ])
            else:
                self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)
        if 'left' in self.part_modals:
            # load left graph
            self.graph_left = PSGraph(part='left', **graph_args)
            A_left = torch.tensor(self.graph_left.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_left', A_left)
            # build left networks
            spatial_kernel_size = A_left.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_left = nn.BatchNorm1d(in_channels * A_left.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_left = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_left = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_left.size()))
                    for i in self.st_gcn_networks_left
                ])
            else:
                self.edge_importance_left = [1] * len(self.st_gcn_networks_left)
        if 'right' in self.part_modals:
            # load right graph
            self.graph_right = PSGraph(part='right', **graph_args)
            A_right = torch.tensor(self.graph_right.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_right', A_right)
            # build right networks
            spatial_kernel_size = A_right.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_right = nn.BatchNorm1d(in_channels * A_right.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_right = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_right = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_right.size()))
                    for i in self.st_gcn_networks_right
                ])
            else:
                self.edge_importance_right = [1] * len(self.st_gcn_networks_right)

        if fusion_method == 'CMAF':
            self.fusion = nn.ModuleDict({
                '7': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'MMTM':
            self.fusion = nn.ModuleDict({
                '7': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'NONE':
            pass
        else:
            raise ValueError('Unknown fusion method: {}'.format(fusion_method))

        self.fc = nn.Linear(hidden_dim * len(part_modals), num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_bf = nn.ModuleDict()
            self.fc_af = nn.ModuleDict()
            for part in part_modals:
                self.fc_bf[part] = nn.Linear(hidden_channels * 2, num_class)
                self.fc_af[part] = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        if self.part_modals == ['hand', 'leg']:
            assert x.keys() == {'hand', 'leg'}
            # data normalization
            for part in ['hand', 'leg']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                else:
                    x[part] = self.data_bn_leg(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            y_hand_bf = None
            y_leg_bf = None
            for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                          self.edge_importance_hand,
                                                                          self.st_gcn_networks_leg,
                                                                          self.edge_importance_leg):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_hand_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_hand = x['hand'].clone()
                        y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                        y_hand = y_hand.view(N, M, -1).mean(dim=1)
                        y_hand = self.fc_bf['hand'](y_hand)
                        y_hand_bf = y_hand.view(y_hand.size(0), -1)
                        y_leg = x['leg'].clone()
                        y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                        y_leg = y_leg.view(N, M, -1).mean(dim=1)
                        y_leg = self.fc_bf['leg'](y_leg)
                        y_leg_bf = y_leg.view(y_leg.size(0), -1)
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                gcn_layer_n += 1

            for part in ['hand', 'leg']:
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)

            x = torch.cat([x['hand'], x['leg']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            if self.return_each_feature:
                return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
            else:
                return x
        elif self.part_modals == ['left', 'right']:
            assert x.keys() == {'left', 'right'}
            # data normalization
            for part in ['left', 'right']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'left':
                    x[part] = self.data_bn_left(x[part])
                else:
                    x[part] = self.data_bn_right(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            y_left_bf = None
            y_right_bf = None
            for gcn_left, importance_left, gcn_right, importance_right in zip(self.st_gcn_networks_left,
                                                                              self.edge_importance_left,
                                                                              self.st_gcn_networks_right,
                                                                              self.edge_importance_right):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_left_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_left = x['left'].clone()
                        y_left = F.avg_pool2d(y_left, y_left.size()[2:])
                        y_left = y_left.view(N, M, -1).mean(dim=1)
                        y_left = self.fc_bf['left'](y_left)
                        y_left_bf = y_left.view(y_left.size(0), -1)
                        y_right = x['right'].clone()
                        y_right = F.avg_pool2d(y_right, y_right.size()[2:])
                        y_right = y_right.view(N, M, -1).mean(dim=1)
                        y_right = self.fc_bf['right'](y_right)
                        y_right_bf = y_right.view(y_right.size(0), -1)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                gcn_layer_n += 1

            for part in ['left', 'right']:
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_left = x['left'].clone()
                y_left = self.fc_af['left'](y_left)
                y_left_af = y_left.view(y_left.size(0), -1)
                y_right = x['right'].clone()
                y_right = self.fc_af['right'](y_right)
                y_right_af = y_right.view(y_right.size(0), -1)

            x = torch.cat([x['left'], x['right']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            if self.return_each_feature:
                return x, y_left_bf, y_right_bf, y_left_af, y_right_af
            else:
                return x
        elif self.part_modals == ['hand', 'leg', 'left', 'right']:
            assert x.keys() == {'hand', 'leg', 'left', 'right'}
            # data normalization
            for part in ['hand', 'leg', 'left', 'right']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                elif part == 'leg':
                    x[part] = self.data_bn_leg(x[part])
                elif part == 'left':
                    x[part] = self.data_bn_left(x[part])
                else:
                    x[part] = self.data_bn_right(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1
            y_hand_bf = None
            y_leg_bf = None
            y_left_bf = None
            y_right_bf = None
            for gcn_hand, importance_hand, gcn_leg, importance_leg, \
                gcn_left, importance_left, gcn_right, importance_right in zip(
                self.st_gcn_networks_hand, self.edge_importance_hand,
                self.st_gcn_networks_leg, self.edge_importance_leg,
                self.st_gcn_networks_left, self.edge_importance_left,
                self.st_gcn_networks_right, self.edge_importance_right):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_hand_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_hand = x['hand'].clone()
                        y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                        y_hand = y_hand.view(N, M, -1).mean(dim=1)
                        y_hand = self.fc_bf['hand'](y_hand)
                        y_hand_bf = y_hand.view(y_hand.size(0), -1)
                        y_leg = x['leg'].clone()
                        y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                        y_leg = y_leg.view(N, M, -1).mean(dim=1)
                        y_leg = self.fc_bf['leg'](y_leg)
                        y_leg_bf = y_leg.view(y_leg.size(0), -1)
                        y_left = x['left'].clone()
                        y_left = F.avg_pool2d(y_left, y_left.size()[2:])
                        y_left = y_left.view(N, M, -1).mean(dim=1)
                        y_left = self.fc_bf['left'](y_left)
                        y_left_bf = y_left.view(y_left.size(0), -1)
                        y_right = x['right'].clone()
                        y_right = F.avg_pool2d(y_right, y_right.size()[2:])
                        y_right = y_right.view(N, M, -1).mean(dim=1)
                        y_right = self.fc_bf['right'](y_right)
                        y_right_bf = y_right.view(y_right.size(0), -1)
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                gcn_layer_n += 1

            for part in ['hand', 'leg', 'left', 'right']:
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                y_left = x['left'].clone()
                y_left = self.fc_af['left'](y_left)
                y_left_af = y_left.view(y_left.size(0), -1)
                y_right = x['right'].clone()
                y_right = self.fc_af['right'](y_right)
                y_right_af = y_right.view(y_right.size(0), -1)

            x = torch.cat([x['hand'], x['leg'], x['left'], x['right']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            if self.return_each_feature:
                return x, y_hand_bf, y_hand_af, y_leg_bf, y_leg_af, y_left_bf, y_left_af, y_right_bf, y_right_af
            else:
                return x

        else:
            raise ValueError('Unknown fusion format')


class AIMCLR_PART_FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
                 fusion_method='MMTM', **kwargs):
        super().__init__()
        self.part_modals = part_modals
        self.fusion_method = fusion_method
        assert fusion_method in ['MMTM', 'CMAF', 'NONE']

        if 'hand' in self.part_modals:
            # load hand graph
            self.graph_hand = PSGraph(part='hand', **graph_args)
            A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_hand', A_hand)
            # build hand networks
            spatial_kernel_size = A_hand.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_hand = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_hand = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_hand.size()))
                    for i in self.st_gcn_networks_hand
                ])
            else:
                self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
        if 'leg' in self.part_modals:
            # load leg graph
            self.graph_leg = PSGraph(part='leg', **graph_args)
            A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_leg', A_leg)
            # build leg networks
            spatial_kernel_size = A_leg.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_leg = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_leg = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_leg.size()))
                    for i in self.st_gcn_networks_leg
                ])
            else:
                self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)
        if 'left' in self.part_modals:
            # load left graph
            self.graph_left = PSGraph(part='left', **graph_args)
            A_left = torch.tensor(self.graph_left.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_left', A_left)
            # build left networks
            spatial_kernel_size = A_left.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_left = nn.BatchNorm1d(in_channels * A_left.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_left = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_left = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_left.size()))
                    for i in self.st_gcn_networks_left
                ])
            else:
                self.edge_importance_left = [1] * len(self.st_gcn_networks_left)
        if 'right' in self.part_modals:
            # load right graph
            self.graph_right = PSGraph(part='right', **graph_args)
            A_right = torch.tensor(self.graph_right.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_right', A_right)
            # build right networks
            spatial_kernel_size = A_right.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_right = nn.BatchNorm1d(in_channels * A_right.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_right = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_right = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_right.size()))
                    for i in self.st_gcn_networks_right
                ])
            else:
                self.edge_importance_right = [1] * len(self.st_gcn_networks_right)

        if fusion_method == 'CMAF':
            self.fusion = nn.ModuleDict({
                '7': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'MMTM':
            self.fusion = nn.ModuleDict({
                '7': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'NONE':
            pass
        else:
            raise ValueError('Unknown fusion method: {}'.format(fusion_method))

        self.fc = nn.Linear(hidden_dim * len(part_modals), num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_bf = nn.ModuleDict()
            self.fc_af = nn.ModuleDict()
            for part in part_modals:
                self.fc_bf[part] = nn.Linear(hidden_channels * 2, num_class)
                self.fc_af[part] = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.ModuleDict({
            'hand': Simam_Drop(num_point=16, keep_prob=0.7),
            'leg': Simam_Drop(num_point=10, keep_prob=0.7)
        })

    def forward(self, x, drop=False):
        # assert self.return_each_feature is False
        assert self.part_modals == ['hand', 'leg']
        if self.part_modals == ['hand', 'leg']:
            assert x.keys() == {'hand', 'leg'}
            # data normalization
            for part in ['hand', 'leg']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                else:
                    x[part] = self.data_bn_leg(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            y_hand_bf = None
            y_leg_bf = None
            dp = {}
            for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                          self.edge_importance_hand,
                                                                          self.st_gcn_networks_leg,
                                                                          self.edge_importance_leg):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_hand_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_hand = x['hand'].clone()
                        if drop:
                            dp_hand_bf = self.dropout['hand'](y_hand)
                            y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                            y_hand = y_hand.view(N, M, -1).mean(dim=1)
                            y_hand = self.fc_bf['hand'](y_hand)
                            y_hand_bf = y_hand.view(y_hand.size(0), -1)
                            dp_hand_bf = F.avg_pool2d(dp_hand_bf, dp_hand_bf.size()[2:])
                            dp_hand_bf = dp_hand_bf.view(N, M, -1).mean(dim=1)
                            dp_hand_bf = self.fc_bf['hand'](dp_hand_bf)
                            dp_hand_bf = dp_hand_bf.view(dp_hand_bf.size(0), -1)
                        else:
                            y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                            y_hand = y_hand.view(N, M, -1).mean(dim=1)
                            y_hand = self.fc_bf['hand'](y_hand)
                            y_hand_bf = y_hand.view(y_hand.size(0), -1)
                        y_leg = x['leg'].clone()
                        if drop:
                            dp_leg_bf = self.dropout['leg'](y_leg)
                            y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                            y_leg = y_leg.view(N, M, -1).mean(dim=1)
                            y_leg = self.fc_bf['leg'](y_leg)
                            y_leg_bf = y_leg.view(y_leg.size(0), -1)
                            dp_leg_bf = F.avg_pool2d(dp_leg_bf, dp_leg_bf.size()[2:])
                            dp_leg_bf = dp_leg_bf.view(N, M, -1).mean(dim=1)
                            dp_leg_bf = self.fc_bf['leg'](dp_leg_bf)
                            dp_leg_bf = dp_leg_bf.view(dp_leg_bf.size(0), -1)
                        else:
                            y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                            y_leg = y_leg.view(N, M, -1).mean(dim=1)
                            y_leg = self.fc_bf['leg'](y_leg)
                            y_leg_bf = y_leg.view(y_leg.size(0), -1)
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                gcn_layer_n += 1

            for part in ['hand', 'leg']:
                if drop:
                    dp[part] = self.dropout[part](x[part])
                    dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
                    dp[part] = dp[part].view(N, M, -1).mean(dim=1)
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                if drop:
                    dp_hand_af = dp['hand'].clone()
                    dp_hand_af = self.fc_af['hand'](dp_hand_af)
                    dp_hand_af = dp_hand_af.view(dp_hand_af.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                if drop:
                    dp_leg_af = dp['leg'].clone()
                    dp_leg_af = self.fc_af['leg'](dp_leg_af)
                    dp_leg_af = dp_leg_af.view(dp_leg_af.size(0), -1)

            x = torch.cat([x['hand'], x['leg']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            if drop:
                dp = torch.cat([dp['hand'], dp['leg']], dim=1)
                dp = self.fc(dp)
                dp = dp.view(dp.size(0), -1)
            if self.return_each_feature:
                if drop:
                    return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af, dp, dp_hand_bf, dp_leg_bf, dp_hand_af, dp_leg_af
                return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
            else:
                if drop:
                    return x, dp
                return x
        elif self.part_modals == ['left', 'right']:
            assert x.keys() == {'left', 'right'}
            # data normalization
            for part in ['left', 'right']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'left':
                    x[part] = self.data_bn_left(x[part])
                else:
                    x[part] = self.data_bn_right(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            y_left_bf = None
            y_right_bf = None
            for gcn_left, importance_left, gcn_right, importance_right in zip(self.st_gcn_networks_left,
                                                                              self.edge_importance_left,
                                                                              self.st_gcn_networks_right,
                                                                              self.edge_importance_right):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_left_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_left = x['left'].clone()
                        y_left = F.avg_pool2d(y_left, y_left.size()[2:])
                        y_left = y_left.view(N, M, -1).mean(dim=1)
                        y_left = self.fc_bf['left'](y_left)
                        y_left_bf = y_left.view(y_left.size(0), -1)
                        y_right = x['right'].clone()
                        y_right = F.avg_pool2d(y_right, y_right.size()[2:])
                        y_right = y_right.view(N, M, -1).mean(dim=1)
                        y_right = self.fc_bf['right'](y_right)
                        y_right_bf = y_right.view(y_right.size(0), -1)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                gcn_layer_n += 1

            for part in ['left', 'right']:
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_left = x['left'].clone()
                y_left = self.fc_af['left'](y_left)
                y_left_af = y_left.view(y_left.size(0), -1)
                y_right = x['right'].clone()
                y_right = self.fc_af['right'](y_right)
                y_right_af = y_right.view(y_right.size(0), -1)

            x = torch.cat([x['left'], x['right']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            if self.return_each_feature:
                return x, y_left_bf, y_right_bf, y_left_af, y_right_af
            else:
                return x
        elif self.part_modals == ['hand', 'leg', 'left', 'right']:
            assert x.keys() == {'hand', 'leg', 'left', 'right'}
            # data normalization
            for part in ['hand', 'leg', 'left', 'right']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                elif part == 'leg':
                    x[part] = self.data_bn_leg(x[part])
                elif part == 'left':
                    x[part] = self.data_bn_left(x[part])
                else:
                    x[part] = self.data_bn_right(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1
            y_hand_bf = None
            y_leg_bf = None
            y_left_bf = None
            y_right_bf = None
            for gcn_hand, importance_hand, gcn_leg, importance_leg, \
                gcn_left, importance_left, gcn_right, importance_right in zip(
                self.st_gcn_networks_hand, self.edge_importance_hand,
                self.st_gcn_networks_leg, self.edge_importance_leg,
                self.st_gcn_networks_left, self.edge_importance_left,
                self.st_gcn_networks_right, self.edge_importance_right):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    if y_hand_bf is None and self.return_each_feature:
                        # save the output of layer 6
                        y_hand = x['hand'].clone()
                        y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
                        y_hand = y_hand.view(N, M, -1).mean(dim=1)
                        y_hand = self.fc_bf['hand'](y_hand)
                        y_hand_bf = y_hand.view(y_hand.size(0), -1)
                        y_leg = x['leg'].clone()
                        y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
                        y_leg = y_leg.view(N, M, -1).mean(dim=1)
                        y_leg = self.fc_bf['leg'](y_leg)
                        y_leg_bf = y_leg.view(y_leg.size(0), -1)
                        y_left = x['left'].clone()
                        y_left = F.avg_pool2d(y_left, y_left.size()[2:])
                        y_left = y_left.view(N, M, -1).mean(dim=1)
                        y_left = self.fc_bf['left'](y_left)
                        y_left_bf = y_left.view(y_left.size(0), -1)
                        y_right = x['right'].clone()
                        y_right = F.avg_pool2d(y_right, y_right.size()[2:])
                        y_right = y_right.view(N, M, -1).mean(dim=1)
                        y_right = self.fc_bf['right'](y_right)
                        y_right_bf = y_right.view(y_right.size(0), -1)
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
                    x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
                gcn_layer_n += 1

            for part in ['hand', 'leg', 'left', 'right']:
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                y_left = x['left'].clone()
                y_left = self.fc_af['left'](y_left)
                y_left_af = y_left.view(y_left.size(0), -1)
                y_right = x['right'].clone()
                y_right = self.fc_af['right'](y_right)
                y_right_af = y_right.view(y_right.size(0), -1)

            x = torch.cat([x['hand'], x['leg'], x['left'], x['right']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            if self.return_each_feature:
                return x, y_hand_bf, y_hand_af, y_leg_bf, y_leg_af, y_left_bf, y_left_af, y_right_bf, y_right_af
            else:
                return x
        else:
            raise ValueError('Unknown fusion format')


# class FinalFusion(nn.Module):
#     def __init__(self, modal_types, modal_channels, n_heads=8, dropout=0.5, device='cuda'):
#         super().__init__()
#         self.cmaf = CMAF(modal_types, modal_channels, n_heads, dropout, device)
#
#     def forward(self, x):
#         for modal in x.keys():
#             x[modal] = x[modal].unsqueeze(-1).unsqueeze(-1)
#         x = self.cmaf(x)
#         for modal in x.keys():
#             x[modal] = x[modal].squeeze(-1)
#         # stack all modal and add
#         x = torch.cat([x[modal] for modal in x.keys()], dim=-1)
#         x = x.mean(dim=-1)
#         return x
#
# class AIMCLR_PART_FUSIONPLUS_STGCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
#                  edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
#                  fusion_method='MMTM', **kwargs):
#         super().__init__()
#         self.part_modals = part_modals
#         self.fusion_method = fusion_method
#         assert fusion_method in ['MMTM', 'CMAF', 'NONE']
#
#         if 'hand' in self.part_modals:
#             # load hand graph
#             self.graph_hand = PSGraph(part='hand', **graph_args)
#             A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
#             self.register_buffer('A_hand', A_hand)
#             # build hand networks
#             spatial_kernel_size = A_hand.size(0)
#             temporal_kernel_size = 9
#             kernel_size = (temporal_kernel_size, spatial_kernel_size)
#             self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
#             kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#             self.st_gcn_networks_hand = nn.ModuleList((
#                 st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
#             ))
#             if edge_importance_weighting:
#                 self.edge_importance_hand = nn.ParameterList([
#                     nn.Parameter(torch.ones(self.A_hand.size()))
#                     for i in self.st_gcn_networks_hand
#                 ])
#             else:
#                 self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
#         if 'leg' in self.part_modals:
#             # load leg graph
#             self.graph_leg = PSGraph(part='leg', **graph_args)
#             A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
#             self.register_buffer('A_leg', A_leg)
#             # build leg networks
#             spatial_kernel_size = A_leg.size(0)
#             temporal_kernel_size = 9
#             kernel_size = (temporal_kernel_size, spatial_kernel_size)
#             self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
#             kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#             self.st_gcn_networks_leg = nn.ModuleList((
#                 st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
#             ))
#             if edge_importance_weighting:
#                 self.edge_importance_leg = nn.ParameterList([
#                     nn.Parameter(torch.ones(self.A_leg.size()))
#                     for i in self.st_gcn_networks_leg
#                 ])
#             else:
#                 self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)
#         if 'left' in self.part_modals:
#             # load left graph
#             self.graph_left = PSGraph(part='left', **graph_args)
#             A_left = torch.tensor(self.graph_left.A, dtype=torch.float32, requires_grad=False)
#             self.register_buffer('A_left', A_left)
#             # build left networks
#             spatial_kernel_size = A_left.size(0)
#             temporal_kernel_size = 9
#             kernel_size = (temporal_kernel_size, spatial_kernel_size)
#             self.data_bn_left = nn.BatchNorm1d(in_channels * A_left.size(1))
#             kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#             self.st_gcn_networks_left = nn.ModuleList((
#                 st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
#             ))
#             if edge_importance_weighting:
#                 self.edge_importance_left = nn.ParameterList([
#                     nn.Parameter(torch.ones(self.A_left.size()))
#                     for i in self.st_gcn_networks_left
#                 ])
#             else:
#                 self.edge_importance_left = [1] * len(self.st_gcn_networks_left)
#         if 'right' in self.part_modals:
#             # load right graph
#             self.graph_right = PSGraph(part='right', **graph_args)
#             A_right = torch.tensor(self.graph_right.A, dtype=torch.float32, requires_grad=False)
#             self.register_buffer('A_right', A_right)
#             # build right networks
#             spatial_kernel_size = A_right.size(0)
#             temporal_kernel_size = 9
#             kernel_size = (temporal_kernel_size, spatial_kernel_size)
#             self.data_bn_right = nn.BatchNorm1d(in_channels * A_right.size(1))
#             kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
#             self.st_gcn_networks_right = nn.ModuleList((
#                 st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
#                 st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
#             ))
#             if edge_importance_weighting:
#                 self.edge_importance_right = nn.ParameterList([
#                     nn.Parameter(torch.ones(self.A_right.size()))
#                     for i in self.st_gcn_networks_right
#                 ])
#             else:
#                 self.edge_importance_right = [1] * len(self.st_gcn_networks_right)
#
#         if fusion_method == 'CMAF':
#             self.fusion = nn.ModuleDict({
#                 '7': CMAF(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 2 for part in part_modals}),
#                 '8': CMAF(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 4 for part in part_modals}),
#                 '9': CMAF(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 4 for part in part_modals}),
#             })
#         elif fusion_method == 'MMTM':
#             self.fusion = nn.ModuleDict({
#                 '7': MMTM(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 2 for part in part_modals}),
#                 '8': MMTM(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 4 for part in part_modals}),
#                 '9': MMTM(modal_types=part_modals,
#                           modal_channels={part: hidden_channels * 4 for part in part_modals}),
#             })
#         elif fusion_method == 'NONE':
#             pass
#         else:
#             raise ValueError('Unknown fusion method: {}'.format(fusion_method))
#
#         self.fc = nn.Linear(hidden_dim, num_class)
#         self.final_fusion = FinalFusion(modal_types=part_modals,
#                                  modal_channels={part: hidden_dim for part in part_modals})
#         self.return_each_feature = return_each_feature
#         assert self.return_each_feature in [False, True]
#
#         if self.return_each_feature:
#             self.fc_bf = nn.ModuleDict()
#             self.fc_af = nn.ModuleDict()
#             for part in part_modals:
#                 self.fc_bf[part] = nn.Linear(hidden_channels * 2, num_class)
#                 self.fc_af[part] = nn.Linear(hidden_dim, num_class)
#
#         self.dropout = nn.ModuleDict({
#             'hand': Simam_Drop(num_point=16, keep_prob=0.7),
#             'leg': Simam_Drop(num_point=10, keep_prob=0.7)
#         })
#
#     def forward(self, x, drop=False):
#         # assert self.return_each_feature is False
#         assert self.part_modals == ['hand', 'leg']
#         if self.part_modals == ['hand', 'leg']:
#             assert x.keys() == {'hand', 'leg'}
#             # data normalization
#             for part in ['hand', 'leg']:
#                 N, C, T, V, M = x[part].size()
#                 x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
#                 x[part] = x[part].view(N * M, V * C, T)
#                 if part == 'hand':
#                     x[part] = self.data_bn_hand(x[part])
#                 else:
#                     x[part] = self.data_bn_leg(x[part])
#                 x[part] = x[part].view(N, M, V, C, T)
#                 x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
#                 x[part] = x[part].view(N * M, C, T, V)
#
#             # forward
#             gcn_layer_n = 1  # total = 10
#             y_hand_bf = None
#             y_leg_bf = None
#             dp = {}
#             for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
#                                                                           self.edge_importance_hand,
#                                                                           self.st_gcn_networks_leg,
#                                                                           self.edge_importance_leg):
#                 if 6 < gcn_layer_n < 10:
#                     # start fusion from layer 7 to 9
#                     if y_hand_bf is None and self.return_each_feature:
#                         # save the output of layer 6
#                         y_hand = x['hand'].clone()
#                         if drop:
#                             dp_hand_bf = self.dropout['hand'](y_hand)
#                             dp_hand_bf = F.avg_pool2d(dp_hand_bf, dp_hand_bf.size()[2:])
#                             dp_hand_bf = dp_hand_bf.view(N, M, -1).mean(dim=1)
#                             dp_hand_bf = self.fc_bf['hand'](dp_hand_bf)
#                             dp_hand_bf = dp_hand_bf.view(dp_hand_bf.size(0), -1)
#                         y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
#                         y_hand = y_hand.view(N, M, -1).mean(dim=1)
#                         y_hand = self.fc_bf['hand'](y_hand)
#                         y_hand_bf = y_hand.view(y_hand.size(0), -1)
#                         y_leg = x['leg'].clone()
#                         if drop:
#                             dp_leg_bf = self.dropout['leg'](y_leg)
#                             dp_leg_bf = F.avg_pool2d(dp_leg_bf, dp_leg_bf.size()[2:])
#                             dp_leg_bf = dp_leg_bf.view(N, M, -1).mean(dim=1)
#                             dp_leg_bf = self.fc_bf['leg'](dp_leg_bf)
#                             dp_leg_bf = dp_leg_bf.view(dp_leg_bf.size(0), -1)
#                         y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
#                         y_leg = y_leg.view(N, M, -1).mean(dim=1)
#                         y_leg = self.fc_bf['leg'](y_leg)
#                         y_leg_bf = y_leg.view(y_leg.size(0), -1)
#                     x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
#                     x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
#                     if self.fusion_method != 'NONE':
#                         x = self.fusion[str(gcn_layer_n)](x)
#                 else:
#                     x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
#                     x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
#                 gcn_layer_n += 1
#
#             for part in ['hand', 'leg']:
#                 if drop:
#                     dp[part] = self.dropout[part](x[part])
#                     dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
#                     dp[part] = dp[part].view(N, M, -1).mean(dim=1)
#                 x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
#                 x[part] = x[part].view(N, M, -1).mean(dim=1)
#
#             if self.return_each_feature:
#                 y_hand = x['hand'].clone()
#                 if drop:
#                     dp_hand_af = self.fc_af['hand'](dp['hand'])
#                     dp_hand_af = dp_hand_af.view(dp_hand_af.size(0), -1)
#                 y_hand = self.fc_af['hand'](y_hand)
#                 y_hand_af = y_hand.view(y_hand.size(0), -1)
#                 y_leg = x['leg'].clone()
#                 if drop:
#                     dp_leg_af = self.fc_af['leg'](dp['leg'])
#                     dp_leg_af = dp_leg_af.view(dp_leg_af.size(0), -1)
#                 y_leg = self.fc_af['leg'](y_leg)
#                 y_leg_af = y_leg.view(y_leg.size(0), -1)
#
#             if self.fusion_method != 'NONE':
#                 x = self.final_fusion(x)
#             else:
#                 x = (x['hand'] + x['leg']) / 2.
#             # print(x.shape)
#             x = self.fc(x)
#             x = x.view(x.size(0), -1)
#             if drop:
#                 dp = torch.cat([dp['hand'], dp['leg']], dim=1)
#                 dp = self.fc(dp)
#                 dp = dp.view(dp.size(0), -1)
#             if self.return_each_feature:
#                 if drop:
#                     return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af, dp, dp_hand_bf, dp_leg_bf, dp_hand_af, dp_leg_af
#                 return x, y_hand_bf, y_leg_bf, y_hand_af, y_leg_af
#             else:
#                 if drop:
#                     return x, dp
#                 return x
#         elif self.part_modals == ['left', 'right']:
#             assert x.keys() == {'left', 'right'}
#             # data normalization
#             for part in ['left', 'right']:
#                 N, C, T, V, M = x[part].size()
#                 x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
#                 x[part] = x[part].view(N * M, V * C, T)
#                 if part == 'left':
#                     x[part] = self.data_bn_left(x[part])
#                 else:
#                     x[part] = self.data_bn_right(x[part])
#                 x[part] = x[part].view(N, M, V, C, T)
#                 x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
#                 x[part] = x[part].view(N * M, C, T, V)
#
#             # forward
#             gcn_layer_n = 1  # total = 10
#             y_left_bf = None
#             y_right_bf = None
#             for gcn_left, importance_left, gcn_right, importance_right in zip(self.st_gcn_networks_left,
#                                                                               self.edge_importance_left,
#                                                                               self.st_gcn_networks_right,
#                                                                               self.edge_importance_right):
#                 if 6 < gcn_layer_n < 10:
#                     # start fusion from layer 7 to 9
#                     if y_left_bf is None and self.return_each_feature:
#                         # save the output of layer 6
#                         y_left = x['left'].clone()
#                         y_left = F.avg_pool2d(y_left, y_left.size()[2:])
#                         y_left = y_left.view(N, M, -1).mean(dim=1)
#                         y_left = self.fc_bf['left'](y_left)
#                         y_left_bf = y_left.view(y_left.size(0), -1)
#                         y_right = x['right'].clone()
#                         y_right = F.avg_pool2d(y_right, y_right.size()[2:])
#                         y_right = y_right.view(N, M, -1).mean(dim=1)
#                         y_right = self.fc_bf['right'](y_right)
#                         y_right_bf = y_right.view(y_right.size(0), -1)
#                     x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
#                     x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
#                     if self.fusion_method != 'NONE':
#                         x = self.fusion[str(gcn_layer_n)](x)
#                 else:
#                     x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
#                     x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
#                 gcn_layer_n += 1
#
#             for part in ['left', 'right']:
#                 x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
#                 x[part] = x[part].view(N, M, -1).mean(dim=1)
#
#             if self.return_each_feature:
#                 y_left = x['left'].clone()
#                 y_left = self.fc_af['left'](y_left)
#                 y_left_af = y_left.view(y_left.size(0), -1)
#                 y_right = x['right'].clone()
#                 y_right = self.fc_af['right'](y_right)
#                 y_right_af = y_right.view(y_right.size(0), -1)
#
#             x = torch.cat([x['left'], x['right']], dim=1)
#             x = self.fc(x)
#             x = x.view(x.size(0), -1)
#
#             if self.return_each_feature:
#                 return x, y_left_bf, y_right_bf, y_left_af, y_right_af
#             else:
#                 return x
#         elif self.part_modals == ['hand', 'leg', 'left', 'right']:
#             assert x.keys() == {'hand', 'leg', 'left', 'right'}
#             # data normalization
#             for part in ['hand', 'leg', 'left', 'right']:
#                 N, C, T, V, M = x[part].size()
#                 x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
#                 x[part] = x[part].view(N * M, V * C, T)
#                 if part == 'hand':
#                     x[part] = self.data_bn_hand(x[part])
#                 elif part == 'leg':
#                     x[part] = self.data_bn_leg(x[part])
#                 elif part == 'left':
#                     x[part] = self.data_bn_left(x[part])
#                 else:
#                     x[part] = self.data_bn_right(x[part])
#                 x[part] = x[part].view(N, M, V, C, T)
#                 x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
#                 x[part] = x[part].view(N * M, C, T, V)
#
#             # forward
#             gcn_layer_n = 1
#             y_hand_bf = None
#             y_leg_bf = None
#             y_left_bf = None
#             y_right_bf = None
#             for gcn_hand, importance_hand, gcn_leg, importance_leg, \
#                 gcn_left, importance_left, gcn_right, importance_right in zip(
#                 self.st_gcn_networks_hand, self.edge_importance_hand,
#                 self.st_gcn_networks_leg, self.edge_importance_leg,
#                 self.st_gcn_networks_left, self.edge_importance_left,
#                 self.st_gcn_networks_right, self.edge_importance_right):
#                 if 6 < gcn_layer_n < 10:
#                     # start fusion from layer 7 to 9
#                     if y_hand_bf is None and self.return_each_feature:
#                         # save the output of layer 6
#                         y_hand = x['hand'].clone()
#                         y_hand = F.avg_pool2d(y_hand, y_hand.size()[2:])
#                         y_hand = y_hand.view(N, M, -1).mean(dim=1)
#                         y_hand = self.fc_bf['hand'](y_hand)
#                         y_hand_bf = y_hand.view(y_hand.size(0), -1)
#                         y_leg = x['leg'].clone()
#                         y_leg = F.avg_pool2d(y_leg, y_leg.size()[2:])
#                         y_leg = y_leg.view(N, M, -1).mean(dim=1)
#                         y_leg = self.fc_bf['leg'](y_leg)
#                         y_leg_bf = y_leg.view(y_leg.size(0), -1)
#                         y_left = x['left'].clone()
#                         y_left = F.avg_pool2d(y_left, y_left.size()[2:])
#                         y_left = y_left.view(N, M, -1).mean(dim=1)
#                         y_left = self.fc_bf['left'](y_left)
#                         y_left_bf = y_left.view(y_left.size(0), -1)
#                         y_right = x['right'].clone()
#                         y_right = F.avg_pool2d(y_right, y_right.size()[2:])
#                         y_right = y_right.view(N, M, -1).mean(dim=1)
#                         y_right = self.fc_bf['right'](y_right)
#                         y_right_bf = y_right.view(y_right.size(0), -1)
#                     x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
#                     x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
#                     x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
#                     x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
#                     if self.fusion_method != 'NONE':
#                         x = self.fusion[str(gcn_layer_n)](x)
#                 else:
#                     x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
#                     x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
#                     x['left'], _ = gcn_left(x['left'], self.A_left * importance_left)
#                     x['right'], _ = gcn_right(x['right'], self.A_right * importance_right)
#                 gcn_layer_n += 1
#
#             for part in ['hand', 'leg', 'left', 'right']:
#                 x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
#                 x[part] = x[part].view(N, M, -1).mean(dim=1)
#
#             if self.return_each_feature:
#                 y_hand = x['hand'].clone()
#                 y_hand = self.fc_af['hand'](y_hand)
#                 y_hand_af = y_hand.view(y_hand.size(0), -1)
#                 y_leg = x['leg'].clone()
#                 y_leg = self.fc_af['leg'](y_leg)
#                 y_leg_af = y_leg.view(y_leg.size(0), -1)
#                 y_left = x['left'].clone()
#                 y_left = self.fc_af['left'](y_left)
#                 y_left_af = y_left.view(y_left.size(0), -1)
#                 y_right = x['right'].clone()
#                 y_right = self.fc_af['right'](y_right)
#                 y_right_af = y_right.view(y_right.size(0), -1)
#
#             x = torch.cat([x['hand'], x['leg'], x['left'], x['right']], dim=1)
#             x = self.fc(x)
#             x = x.view(x.size(0), -1)
#
#             if self.return_each_feature:
#                 return x, y_hand_bf, y_hand_af, y_leg_bf, y_leg_af, y_left_bf, y_left_af, y_right_bf, y_right_af
#             else:
#                 return x
#
#         else:
#             raise ValueError('Unknown fusion format')


class MM_PART_FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
                 fusion_method='MMTM', **kwargs):
        super().__init__()
        self.part_modals = part_modals
        self.fusion_method = fusion_method
        assert fusion_method in ['MMTM', 'CMAF', 'NONE']

        if 'hand' in self.part_modals:
            # load hand graph
            self.graph_hand = PSGraph(part='hand', **graph_args)
            A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_hand', A_hand)
            # build hand networks
            spatial_kernel_size = A_hand.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_hand = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_hand = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_hand.size()))
                    for i in self.st_gcn_networks_hand
                ])
            else:
                self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
        if 'leg' in self.part_modals:
            # load leg graph
            self.graph_leg = PSGraph(part='leg', **graph_args)
            A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_leg', A_leg)
            # build leg networks
            spatial_kernel_size = A_leg.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_leg = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_leg = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_leg.size()))
                    for i in self.st_gcn_networks_leg
                ])
            else:
                self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        if fusion_method == 'CMAF':
            self.fusion = nn.ModuleDict({
                '7': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'MMTM':
            self.fusion = nn.ModuleDict({
                '7': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'NONE':
            pass
        else:
            raise ValueError('Unknown fusion method: {}'.format(fusion_method))

        self.fc = nn.Linear(hidden_dim * len(part_modals), num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            self.fc_af_mm = nn.ModuleDict()
            self.fc_af = nn.ModuleDict()
            for part in part_modals:
                self.fc_af_mm[part] = nn.Linear(hidden_dim, num_class)
                self.fc_af[part] = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.ModuleDict({
            'hand': Simam_Drop(num_point=16, keep_prob=0.7),
            'leg': Simam_Drop(num_point=10, keep_prob=0.7)
        })

    def forward(self, x, drop=False):
        # assert self.return_each_feature is False
        assert self.part_modals == ['hand', 'leg']
        if self.part_modals == ['hand', 'leg']:
            assert x.keys() == {'hand', 'leg'}
            # data normalization
            for part in ['hand', 'leg']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                else:
                    x[part] = self.data_bn_leg(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            dp = {}
            for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                          self.edge_importance_hand,
                                                                          self.st_gcn_networks_leg,
                                                                          self.edge_importance_leg):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                gcn_layer_n += 1

            for part in ['hand', 'leg']:
                if drop:
                    dp[part] = self.dropout[part](x[part])
                    dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
                    dp[part] = dp[part].view(N, M, -1).mean(dim=1)
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                if drop:
                    dp_hand_af = dp['hand'].clone()
                    dp_hand_af = self.fc_af['hand'](dp_hand_af)
                    dp_hand_af = dp_hand_af.view(dp_hand_af.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                if drop:
                    dp_leg_af = dp['leg'].clone()
                    dp_leg_af = self.fc_af['leg'](dp_leg_af)
                    dp_leg_af = dp_leg_af.view(dp_leg_af.size(0), -1)

                y_hand_mm = x['hand'].clone()
                y_hand_mm = self.fc_af_mm['hand'](y_hand_mm)
                y_hand_af_mm = y_hand_mm.view(y_hand_mm.size(0), -1)
                if drop:
                    dp_hand_mm = dp['hand'].clone()
                    dp_hand_mm = self.fc_af_mm['hand'](dp_hand_mm)
                    dp_hand_af_mm = dp_hand_mm.view(dp_hand_mm.size(0), -1)
                y_leg_mm = x['leg'].clone()
                y_leg_mm = self.fc_af_mm['leg'](y_leg_mm)
                y_leg_af_mm = y_leg_mm.view(y_leg_mm.size(0), -1)
                if drop:
                    dp_leg_mm = dp['leg'].clone()
                    dp_leg_mm = self.fc_af_mm['leg'](dp_leg_mm)
                    dp_leg_af_mm = dp_leg_mm.view(dp_leg_mm.size(0), -1)

            x = torch.cat([x['hand'], x['leg']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            if drop:
                dp = torch.cat([dp['hand'], dp['leg']], dim=1)
                dp = self.fc(dp)
                dp = dp.view(dp.size(0), -1)
            if self.return_each_feature:
                if drop:
                    return x, y_hand_af_mm, y_leg_af_mm, y_hand_af, y_leg_af, dp, dp_hand_af_mm, dp_leg_af_mm, dp_hand_af, dp_leg_af
                return x, y_hand_af_mm, y_leg_af_mm, y_hand_af, y_leg_af
            else:
                if drop:
                    return x, dp
                return x


class MM2_PART_FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
                 fusion_method='MMTM', **kwargs):
        super().__init__()
        self.part_modals = part_modals
        self.fusion_method = fusion_method
        assert fusion_method in ['MMTM', 'CMAF', 'NONE']

        if 'hand' in self.part_modals:
            # load hand graph
            self.graph_hand = PSGraph(part='hand', **graph_args)
            A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_hand', A_hand)
            # build hand networks
            spatial_kernel_size = A_hand.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_hand = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_hand = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_hand.size()))
                    for i in self.st_gcn_networks_hand
                ])
            else:
                self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
        if 'leg' in self.part_modals:
            # load leg graph
            self.graph_leg = PSGraph(part='leg', **graph_args)
            A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_leg', A_leg)
            # build leg networks
            spatial_kernel_size = A_leg.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_leg = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_leg = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_leg.size()))
                    for i in self.st_gcn_networks_leg
                ])
            else:
                self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        if fusion_method == 'CMAF':
            self.fusion = nn.ModuleDict({
                '7': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'MMTM':
            self.fusion = nn.ModuleDict({
                '7': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'NONE':
            pass
        else:
            raise ValueError('Unknown fusion method: {}'.format(fusion_method))

        self.fc = nn.Linear(hidden_dim * len(part_modals), num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            # self.fc_af_mm = nn.ModuleDict()
            self.fc_af = nn.ModuleDict()
            for part in part_modals:
                # self.fc_af_mm[part] = nn.Linear(hidden_dim, num_class)
                self.fc_af[part] = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.ModuleDict({
            'hand': Simam_Drop(num_point=16, keep_prob=0.7),
            'leg': Simam_Drop(num_point=10, keep_prob=0.7)
        })

    def forward(self, x, drop=False):
        # assert self.return_each_feature is False
        assert self.part_modals == ['hand', 'leg']
        if self.part_modals == ['hand', 'leg']:
            assert x.keys() == {'hand', 'leg'}
            # data normalization
            for part in ['hand', 'leg']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                else:
                    x[part] = self.data_bn_leg(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            dp = {}
            for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                          self.edge_importance_hand,
                                                                          self.st_gcn_networks_leg,
                                                                          self.edge_importance_leg):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                gcn_layer_n += 1

            for part in ['hand', 'leg']:
                if drop:
                    dp[part] = self.dropout[part](x[part])
                    dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
                    dp[part] = dp[part].view(N, M, -1).mean(dim=1)
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                if drop:
                    dp_hand_af = dp['hand'].clone()
                    dp_hand_af = self.fc_af['hand'](dp_hand_af)
                    dp_hand_af = dp_hand_af.view(dp_hand_af.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                if drop:
                    dp_leg_af = dp['leg'].clone()
                    dp_leg_af = self.fc_af['leg'](dp_leg_af)
                    dp_leg_af = dp_leg_af.view(dp_leg_af.size(0), -1)

                # y_hand_mm = x['hand'].clone()
                # y_hand_mm = self.fc_af_mm['hand'](y_hand_mm)
                # y_hand_af_mm = y_hand_mm.view(y_hand_mm.size(0), -1)
                # if drop:
                #     dp_hand_mm = dp['hand'].clone()
                #     dp_hand_mm = self.fc_af_mm['hand'](dp_hand_mm)
                #     dp_hand_af_mm = dp_hand_mm.view(dp_hand_mm.size(0), -1)
                # y_leg_mm = x['leg'].clone()
                # y_leg_mm = self.fc_af_mm['leg'](y_leg_mm)
                # y_leg_af_mm = y_leg_mm.view(y_leg_mm.size(0), -1)
                # if drop:
                #     dp_leg_mm = dp['leg'].clone()
                #     dp_leg_mm = self.fc_af_mm['leg'](dp_leg_mm)
                #     dp_leg_af_mm = dp_leg_mm.view(dp_leg_mm.size(0), -1)

            x = torch.cat([x['hand'], x['leg']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            if drop:
                dp = torch.cat([dp['hand'], dp['leg']], dim=1)
                dp = self.fc(dp)
                dp = dp.view(dp.size(0), -1)
            if self.return_each_feature:
                if drop:
                    return x, None, None, y_hand_af, y_leg_af, dp, None, None, dp_hand_af, dp_leg_af
                return x, None, None, y_hand_af, y_leg_af
            else:
                if drop:
                    return x, dp
                return x


class FULL_FUSION_STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, return_each_feature=False, part_modals=['hand', 'leg', 'left', 'right'],
                 fusion_method='MMTM', **kwargs):
        super().__init__()
        self.part_modals = part_modals
        self.fusion_method = fusion_method
        assert fusion_method in ['MMTM', 'CMAF', 'NONE']

        if 'joint' in self.part_modals:
            # load hand graph
            self.graph_hand = PSGraph(part='body', **graph_args)
            A_hand = torch.tensor(self.graph_hand.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_hand', A_hand)
            # build hand networks
            spatial_kernel_size = A_hand.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_hand = nn.BatchNorm1d(in_channels * A_hand.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_hand = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_hand = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_hand.size()))
                    for i in self.st_gcn_networks_hand
                ])
            else:
                self.edge_importance_hand = [1] * len(self.st_gcn_networks_hand)
        if 'leg' in self.part_modals:
            # load leg graph
            self.graph_leg = PSGraph(part='leg', **graph_args)
            A_leg = torch.tensor(self.graph_leg.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A_leg', A_leg)
            # build leg networks
            spatial_kernel_size = A_leg.size(0)
            temporal_kernel_size = 9
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.data_bn_leg = nn.BatchNorm1d(in_channels * A_leg.size(1))
            kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
            self.st_gcn_networks_leg = nn.ModuleList((
                st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
                st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
                st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
            ))
            if edge_importance_weighting:
                self.edge_importance_leg = nn.ParameterList([
                    nn.Parameter(torch.ones(self.A_leg.size()))
                    for i in self.st_gcn_networks_leg
                ])
            else:
                self.edge_importance_leg = [1] * len(self.st_gcn_networks_leg)

        if fusion_method == 'CMAF':
            self.fusion = nn.ModuleDict({
                '7': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': CMAF(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'MMTM':
            self.fusion = nn.ModuleDict({
                '7': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 2 for part in part_modals}),
                '8': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
                '9': MMTM(modal_types=part_modals,
                          modal_channels={part: hidden_channels * 4 for part in part_modals}),
            })
        elif fusion_method == 'NONE':
            pass
        else:
            raise ValueError('Unknown fusion method: {}'.format(fusion_method))

        self.fc = nn.Linear(hidden_dim * len(part_modals), num_class)

        self.return_each_feature = return_each_feature
        assert self.return_each_feature in [False, True]

        if self.return_each_feature:
            # self.fc_af_mm = nn.ModuleDict()
            self.fc_af = nn.ModuleDict()
            for part in part_modals:
                # self.fc_af_mm[part] = nn.Linear(hidden_dim, num_class)
                self.fc_af[part] = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.ModuleDict({
            'hand': Simam_Drop(num_point=16, keep_prob=0.7),
            'leg': Simam_Drop(num_point=10, keep_prob=0.7)
        })

    def forward(self, x, drop=False):
        # assert self.return_each_feature is False
        assert self.part_modals == ['hand', 'leg']
        if self.part_modals == ['hand', 'leg']:
            assert x.keys() == {'hand', 'leg'}
            # data normalization
            for part in ['hand', 'leg']:
                N, C, T, V, M = x[part].size()
                x[part] = x[part].permute(0, 4, 3, 1, 2).contiguous()
                x[part] = x[part].view(N * M, V * C, T)
                if part == 'hand':
                    x[part] = self.data_bn_hand(x[part])
                else:
                    x[part] = self.data_bn_leg(x[part])
                x[part] = x[part].view(N, M, V, C, T)
                x[part] = x[part].permute(0, 1, 3, 4, 2).contiguous()
                x[part] = x[part].view(N * M, C, T, V)

            # forward
            gcn_layer_n = 1  # total = 10
            dp = {}
            for gcn_hand, importance_hand, gcn_leg, importance_leg in zip(self.st_gcn_networks_hand,
                                                                          self.edge_importance_hand,
                                                                          self.st_gcn_networks_leg,
                                                                          self.edge_importance_leg):
                if 6 < gcn_layer_n < 10:
                    # start fusion from layer 7 to 9
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                    if self.fusion_method != 'NONE':
                        x = self.fusion[str(gcn_layer_n)](x)
                else:
                    x['hand'], _ = gcn_hand(x['hand'], self.A_hand * importance_hand)
                    x['leg'], _ = gcn_leg(x['leg'], self.A_leg * importance_leg)
                gcn_layer_n += 1

            for part in ['hand', 'leg']:
                if drop:
                    dp[part] = self.dropout[part](x[part])
                    dp[part] = F.avg_pool2d(dp[part], dp[part].size()[2:])
                    dp[part] = dp[part].view(N, M, -1).mean(dim=1)
                x[part] = F.avg_pool2d(x[part], x[part].size()[2:])
                x[part] = x[part].view(N, M, -1).mean(dim=1)

            if self.return_each_feature:
                y_hand = x['hand'].clone()
                y_hand = self.fc_af['hand'](y_hand)
                y_hand_af = y_hand.view(y_hand.size(0), -1)
                if drop:
                    dp_hand_af = dp['hand'].clone()
                    dp_hand_af = self.fc_af['hand'](dp_hand_af)
                    dp_hand_af = dp_hand_af.view(dp_hand_af.size(0), -1)
                y_leg = x['leg'].clone()
                y_leg = self.fc_af['leg'](y_leg)
                y_leg_af = y_leg.view(y_leg.size(0), -1)
                if drop:
                    dp_leg_af = dp['leg'].clone()
                    dp_leg_af = self.fc_af['leg'](dp_leg_af)
                    dp_leg_af = dp_leg_af.view(dp_leg_af.size(0), -1)

                # y_hand_mm = x['hand'].clone()
                # y_hand_mm = self.fc_af_mm['hand'](y_hand_mm)
                # y_hand_af_mm = y_hand_mm.view(y_hand_mm.size(0), -1)
                # if drop:
                #     dp_hand_mm = dp['hand'].clone()
                #     dp_hand_mm = self.fc_af_mm['hand'](dp_hand_mm)
                #     dp_hand_af_mm = dp_hand_mm.view(dp_hand_mm.size(0), -1)
                # y_leg_mm = x['leg'].clone()
                # y_leg_mm = self.fc_af_mm['leg'](y_leg_mm)
                # y_leg_af_mm = y_leg_mm.view(y_leg_mm.size(0), -1)
                # if drop:
                #     dp_leg_mm = dp['leg'].clone()
                #     dp_leg_mm = self.fc_af_mm['leg'](dp_leg_mm)
                #     dp_leg_af_mm = dp_leg_mm.view(dp_leg_mm.size(0), -1)

            x = torch.cat([x['hand'], x['leg']], dim=1)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            if drop:
                dp = torch.cat([dp['hand'], dp['leg']], dim=1)
                dp = self.fc(dp)
                dp = dp.view(dp.size(0), -1)
            if self.return_each_feature:
                if drop:
                    return x, None, None, y_hand_af, y_leg_af, dp, None, None, dp_hand_af, dp_leg_af
                return x, None, None, y_hand_af, y_leg_af
            else:
                if drop:
                    return x, dp
                return x