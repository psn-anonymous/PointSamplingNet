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

        self.mlp_convs.append(nn.Conv1d(3, mlp[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(mlp[0]))

        for i in range(len(mlp)-1):
            self.mlp_convs.append(nn.Conv1d(mlp[i], mlp[i+1], 1))

        for i in range(len(mlp)-1):
            self.mlp_bns.append(nn.BatchNorm1d(mlp[i+1]))

        self.global_feature = global_feature

        if self.global_feature:
            self.mlp_convs.append(nn.Conv1d(mlp[-1] * 2, num_to_sample, 1))
        else:
            self.mlp_convs.append(nn.Conv1d(mlp[-1], num_to_sample, 1))

        self.softmax = nn.Softmax(1)

        self.S = num_to_sample
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
        _, N, _ = coordinate.size()

        assert self.S < N, "The number to sample must less than input points !"

        x = coordinate.transpose(2, 1)  # Channel First

        for i in range(len(self.mlp_convs) - 1):
            x = F.relu(self.mlp_bns[i](self.mlp_convs[i](x)))

        if self.global_feature:
            max_feature = torch.max(x, 2, keepdim=True)[0]
            max_feature = max_feature.repeat(1, 1, N)  # [B, mlp[-1], m]
            x = torch.cat([x, max_feature], 1)  # [B, mlp[-1]*2, m]

        x = self.mlp_convs[-1](x)  # [B,S,m]

        Q = self.softmax(x)  # [B, S, N] 归一化

        _, indices = torch.sort(input=Q, dim=2, descending=True)  # [B, S, m]

        grouped_indices = indices[:, :, 0:self.n]  # [B, S, n]

        sampled_indices = indices[:, :, 0]  # [B, S]

        return sampled_indices, grouped_indices
