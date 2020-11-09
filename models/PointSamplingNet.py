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

        return sampled_indices, grouped_indices

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

    def forward(self, coordinate: Tensor) -> Tuple[Tensor, List[Tensor]]:
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

        Q = self.softmax(x)  # [B, s, m]

        _, indices = torch.sort(input=Q, dim=2, descending=True)    # [B, s, m]

        grouped_indices_msg = []
        for n in self.msg_n:
            grouped_indices_msg.append(indices[:, :, :n])

        sampled_indices = indices[:, :, 0]  # [B, s]

        return sampled_indices, grouped_indices_msg


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
