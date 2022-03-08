import torch
import torch.nn as nn


class MultiAdjGNN(nn.Module):
    def __init__(self, gnns, in_dim, out_dim, num_supports,
                 order, bias_start=0.0, dropout=0):
        super(MultiAdjGNN, self).__init__()
        assert len(gnns) == num_supports
        self.gnns = gnns
        self.linear = nn.Conv1d(
            in_dim * (order * num_supports + 1), out_dim, 1)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        x = torch.cat([x,  # add GNN with zero order
                       *[gnn(x, A) for gnn, A in zip(self.gnns, adjs)]], dim=1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GCN(nn.Module):
    """Some Information about GCN"""

    def __init__(self, order, operation='ncvl, vw -> ncwl'):
        # follow the original implementation. The correct operation should be
        # 'ncvl, wv -> ncwl' but performs worse (don't know why).
        super(GCN, self).__init__()
        self.order = order
        self.conv_operation = operation

    def gconv(self, x, A):
        A = A.to(x.device)
        return torch.einsum(self.conv_operation, x, A)

    def forward(self, x, A):
        out = []
        x0 = x
        x1 = self.gconv(x0, A)
        out.append(x1)
        for _ in range(2, self.order + 1):
            x2 = 2 * self.gconv(x1, A) - x0
            out.append(x2)
            x1, x0 = x2, x1
        x = torch.cat(out, dim=1)
        return x
