import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple


class PointSamplingNet(nn.Module):
    """
    Point Sampling Net PyTorch Module.

    Attributes:
        num_to_sample: the number to sample, int
        max_local_num: the max number of local area, int
        mlp: the channels of feature transform function, List[int]
        global_geature: whether enable global feature, bool
    """

    def __init__(self, num_to_sample: int = 512, max_local_num: int = 32, mlp: List[int] = [32, 64, 256], global_feature: bool = False) -> None:
        """
        Initialization of Point Sampling Net.
        """
        super(PointSamplingNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        assert len(mlp) > 1, "The number of MLP layers must greater than 1 !"

        self.mlp_convs.append(
            nn.Conv1d(in_channels=3, out_channels=mlp[0], kernel_size=1))
        self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[0]))

        for i in range(len(mlp)-1):
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[i], out_channels=mlp[i+1], kernel_size=1))

        for i in range(len(mlp)-1):
            self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[i+1]))

        self.global_feature = global_feature

        if self.global_feature:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1] * 2, out_channels=num_to_sample, kernel_size=1))
        else:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1], out_channels=num_to_sample, kernel_size=1))

        self.s = num_to_sample
        self.n = max_local_num

    def forward(self, coordinate: Tensor, feature: Tensor, train: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward propagation of Point Sampling Net

        Args:
            coordinate: input points position data, [B, m, 3]
            feature: input points feature, [B, m, d]
        Returns:
            sampled indices: the indices of sampled points, [B, s]
            grouped_indices: the indices of grouped points, [B, s, n]
        """
        _, m, _ = coordinate.size()

        assert self.s < m, "The number to sample must less than input points !"

        x = coordinate.transpose(2, 1)  # Channel First

        for i in range(len(self.mlp_convs) - 1):
            x = F.relu(self.mlp_bns[i](self.mlp_convs[i](x)))

        if self.global_feature:
            max_feature = torch.max(x, 2, keepdim=True)[0]
            max_feature = max_feature.repeat(1, 1, m)   # [B, mlp[-1], m]
            x = torch.cat([x, max_feature], 1)  # [B, mlp[-1] * 2, m]

        x = self.mlp_convs[-1](x)   # [B,s,m]

        Q = torch.sigmoid(x)  # [B, s, m]

        _, indices = torch.sort(input=Q, dim=2, descending=True)    # [B, s, m]
        grouped_indices = indices[:,:,:self.n]
        grouped_points = index_points(coordinate, grouped_indices)[:,:,:self.n,:]  #[B,s,n,3]
        if feature is not None:
            grouped_feature = index_points(feature, grouped_indices)[:,:,:self.n,:]  #[B,s,n,d]
            if not train:
                sampled_points = grouped_points[:,:,0,:]  # [B,s,3]
                sampled_feature = grouped_feature[:,:,0,:]  #[B,s,d]
            else:
                Q = gumbel_softmax_sample(Q)  # [B, s, m]
                sampled_points = torch.matmul(Q, coordinate)  # [B,s,3]
                sampled_feature = torch.matmul(Q, feature)  # [B,s,d]
                grouped_feature[:,:,0,:] = sampled_feature
        else:
            if not train:
                sampled_points = grouped_points[:,:,0,:]  # [B,s,3]
                sampled_feature = None  #[B,s,d]
            else:
                Q = gumbel_softmax_sample(Q)  # [B, s, m]
                sampled_points = torch.matmul(Q, coordinate)  # [B,s,3]
                sampled_feature = None
            grouped_feature = None

        return sampled_points, grouped_points, sampled_feature, grouped_feature

class PointSamplingNetRadius(nn.Module):
    """
    Point Sampling Net with heuristic condition PyTorch Module.
    This example is radius query
    You may replace function C(x) by your own function

    Attributes:
        num_to_sample: the number to sample, int
        radius: radius to query, float
        max_local_num: the max number of local area, int
        mlp: the channels of feature transform function, List[int]
        global_geature: whether enable global feature, bool
    """

    def __init__(self, num_to_sample: int = 512, radius: float = 1.0, max_local_num: int = 32, mlp: List[int] = [32, 64, 256], global_feature: bool = False) -> None:
        """
        Initialization of Point Sampling Net.
        """
        super(PointSamplingNetRadius, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.radius = radius

        assert len(mlp) > 1, "The number of MLP layers must greater than 1 !"

        self.mlp_convs.append(
            nn.Conv1d(in_channels=3, out_channels=mlp[0], kernel_size=1))
        self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[0]))

        for i in range(len(mlp)-1):
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[i], out_channels=mlp[i+1], kernel_size=1))

        for i in range(len(mlp)-1):
            self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[i+1]))

        self.global_feature = global_feature

        if self.global_feature:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1] * 2, out_channels=num_to_sample, kernel_size=1))
        else:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1], out_channels=num_to_sample, kernel_size=1))

        self.softmax = nn.Softmax(dim=1)

        self.s = num_to_sample
        self.n = max_local_num

    def forward(self, coordinate: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagation of Point Sampling Net

        Args:
            coordinate: input points position data, [B, m, 3]
        Returns:
            sampled indices: the indices of sampled points, [B, s]
            grouped_indices: the indices of grouped points, [B, s, n]
        """
        _, m, _ = coordinate.size()

        assert self.s < m, "The number to sample must less than input points !"

        x = coordinate.transpose(2, 1)  # Channel First

        for i in range(len(self.mlp_convs) - 1):
            x = F.relu(self.mlp_bns[i](self.mlp_convs[i](x)))

        if self.global_feature:
            max_feature = torch.max(x, 2, keepdim=True)[0]
            max_feature = max_feature.repeat(1, 1, m)   # [B, mlp[-1], m]
            x = torch.cat([x, max_feature], 1)  # [B, mlp[-1] * 2, m]

        x = self.mlp_convs[-1](x)   # [B,s,m]

        Q = self.softmax(x)  # [B, s, m]

        _, indices = torch.sort(input=Q, dim=2, descending=True)    # [B, s, m]

        grouped_indices = indices[:, :, 0:self.n]   # [B, s, n]

        sampled_indices = indices[:, :, 0]  # [B, s]

        # function C(x)
        # you may replace C(x) by your heuristic condition
        sampled_coordinate = torch.unsqueeze(index_points(coordinate, sampled_indices), dim=2)  # [B, s, 1, 3]
        grouped_coordinate = index_points(coordinate, grouped_indices)  # [B, s, m, 3]

        diff = grouped_coordinate - sampled_coordinate
        diff = diff ** 2
        diff = torch.sum(diff, dim=3) #[B, s, m]
        mask = diff > self.radius ** 2

        sampled_indices_expand = torch.unsqueeze(sampled_indices, dim=2).repeat(1, 1, self.n)  #[B, s, n]
        grouped_indices[mask] = sampled_indices_expand[mask]
        # function C(x) end

        return sampled_indices, grouped_indices

class PointSamplingNetMSG(nn.Module):
    """
    Point Sampling Net with Multi-scale Grouping PyTorch Module.

    Attributes:
        num_to_sample: the number to sample, int
        msg_n: the list of mutil-scale grouping n values, List[int]
        mlp: the channels of feature transform function, List[int]
        global_geature: whether enable global feature, bool
    """

    def __init__(self, num_to_sample: int = 512, msg_n: List[int] = [32, 64], mlp: List[int] = [32, 64, 256], global_feature: bool = False) -> None:
        """
        Initialization of Point Sampling Net.
        """
        super(PointSamplingNetMSG, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        assert len(mlp) > 1, "The number of MLP layers must greater than 1 !"

        self.mlp_convs.append(
            nn.Conv1d(in_channels=3, out_channels=mlp[0], kernel_size=1))
        self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[0]))

        for i in range(len(mlp)-1):
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[i], out_channels=mlp[i+1], kernel_size=1))

        for i in range(len(mlp)-1):
            self.mlp_bns.append(nn.BatchNorm1d(num_features=mlp[i+1]))

        self.global_feature = global_feature

        if self.global_feature:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1] * 2, out_channels=num_to_sample, kernel_size=1))
        else:
            self.mlp_convs.append(
                nn.Conv1d(in_channels=mlp[-1], out_channels=num_to_sample, kernel_size=1))

        self.softmax = nn.Softmax(dim=1)

        self.s = num_to_sample
        self.msg_n = msg_n

    def forward(self, coordinate: Tensor, feature: Tensor, train: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Forward propagation of Point Sampling Net

        Args:
            coordinate: input points position data, [B, m, 3]
        Returns:
            sampled indices: the indices of sampled points, [B, s]
            grouped_indices_msg: the multi-scale grouping indices of grouped points, List[Tensor]
        """
        _, m, _ = coordinate.size()

        assert self.s < m, "The number to sample must less than input points !"

        x = coordinate.transpose(2, 1)  # Channel First

        for i in range(len(self.mlp_convs) - 1):
            x = F.relu(self.mlp_bns[i](self.mlp_convs[i](x)))

        if self.global_feature:
            max_feature = torch.max(x, 2, keepdim=True)[0]
            max_feature = max_feature.repeat(1, 1, m)   # [B, mlp[-1], m]
            x = torch.cat([x, max_feature], 1)  # [B, mlp[-1] * 2, m]

        x = self.mlp_convs[-1](x)   # [B,s,m]


        Q = torch.sigmoid(x)  # [B, s, m]

        _, indices = torch.sort(input=Q, dim=2, descending=True)    # [B, s, m]
        grouped_indices = indices[:,:,:self.n]
        grouped_points_msg = []
        for n in self.msg_n:
            grouped_points_msg.append(index_points(coordinate, grouped_indices)[:,:,:n,:])
        if feature is not None:
            grouped_feature_msg = []
            for n in self.msg_n:
                grouped_feature_msg.append(index_points(feature, grouped_indices)[:,:,:n,:])
            if not train:
                sampled_points = grouped_points_msg[0][:,:,0,:]  # [B,s,3]
                sampled_feature = grouped_feature_msg[-1][:,:,0,:]  #[B,s,d]
            else:
                Q = gumbel_softmax_sample(Q)  # [B, s, m]
                sampled_points = torch.matmul(Q, coordinate)  # [B,s,3]
                sampled_feature = torch.matmul(Q, feature)  # [B,s,d]
                for n in self.msg_n:
                    grouped_feature_msg[n][:,:,0,:] = sampled_feature
        else:
            if not train:
                sampled_points = grouped_points_msg[0][:,:,0,:]  # [B,s,3]
                sampled_feature = None  #[B,s,d]
            else:
                Q = gumbel_softmax_sample(Q)  # [B, s, m]
                sampled_points = torch.matmul(Q, coordinate)  # [B,s,3]
                sampled_feature = None
            grouped_feature_msg = None

        return sampled_points, grouped_points_msg, sampled_feature, grouped_feature_msg


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, dim=-1, temperature=0.001):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=dim)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        #print(y_hard[0], "random")
        y = (y_hard - y).detach() + y
    return y

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    #print(logits[0],"a")
    #print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs
