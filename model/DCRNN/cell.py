import torch
import torch.nn as nn

from model.GNN.model import MultiAdjGNN, GCN


class DCGRUCell(nn.Module):
    def __init__(self, in_dim, hid_dim, gcn_order, num_supports):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()

        # support other nonlinearities up here?
        gcns = nn.ModuleList([GCN(gcn_order, operation='bcn, mn -> bcm')
                              for _ in range(num_supports)])
        self.gcn_ru = MultiAdjGNN(gcns,
                                  in_dim=in_dim + hid_dim,
                                  out_dim=hid_dim * 2,
                                  num_supports=num_supports,
                                  order=gcn_order,
                                  bias_start=1.0)

        self.gcn_C = MultiAdjGNN(gcns,
                                 in_dim=in_dim + hid_dim,
                                 out_dim=hid_dim,
                                 num_supports=num_supports,
                                 order=gcn_order,
                                 )
        self.hid_dim = hid_dim

    def forward(self, inputs, hx, supports):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """

        concat_input = torch.cat([inputs, hx], dim=1)
        value = torch.sigmoid(self.gcn_ru(concat_input, supports))
        r = value[:, :self.hid_dim, ...]
        u = value[:, self.hid_dim:, ...]
        C = torch.tanh(self.gcn_C(
            torch.cat([inputs, r * hx], dim=1), supports))
        new_state = u * hx + (1.0 - u) * C
        return new_state
