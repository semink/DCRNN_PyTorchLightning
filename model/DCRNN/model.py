import torch
import torch.nn as nn

from lib import utils


from model.DCRNN.cell import DCGRUCell


class DCGRUModel(nn.Module):
    def __init__(self, in_dim, hid_dim, num_rnn_layers,
                 gcn_order, num_supports):
        super().__init__()
        dcgru_layers = nn.ModuleList(
            [DCGRUCell(in_dim, hid_dim, gcn_order, num_supports)])
        for _ in range(num_rnn_layers - 1):
            dcgru_layers.append(
                DCGRUCell(
                    hid_dim,
                    hid_dim,
                    gcn_order,
                    num_supports))
        self.dcgru = DCGRU(dcgru_layers, hid_dim)

    def forward(self, inputs, supports, hidden_state=None):
        hidden_states = self.dcgru(inputs, hidden_state, supports)
        return hidden_states


class DCGRU(nn.Module):
    def __init__(self, dcgru_layers, hid_dim):
        super().__init__()
        self.dcgru_layers = dcgru_layers
        self.hid_dim = hid_dim
        self.num_rnn_layers = len(self.dcgru_layers)

    def forward(self, inputs, hidden_state, supports):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, input_dim, num_nodes)
        :param hidden_state: (num_layers, batch_size, self.hid_dim, num_nodes)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hid_dim, num_nodes)
                 (lower indices mean lower layers)
        """

        hidden_states = []
        next_hidden_state = inputs
        for hs_layer, dcgru_layer in zip(hidden_state, self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                next_hidden_state, hs_layer, supports)
            hidden_states.append(next_hidden_state)
        # runs in O(num_layers) so not too slow
        return hidden_states


class DCRNNModel(nn.Module):
    def __init__(self, adj_mx, input_dim, output_dim, rnn_units, num_rnn_layers,
                 max_diffusion_step, horizon):
        super().__init__()
        self.out_dim = output_dim
        self.horizon = horizon
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units

        self._supports = utils.double_transition_matrix(adj_mx)

        self.encoder_model = DCGRUModel(input_dim, rnn_units, num_rnn_layers, max_diffusion_step,
                                        len(self._supports))
        self.decoder_model = DCGRUModel(output_dim, rnn_units, num_rnn_layers, max_diffusion_step,
                                        len(self._supports))
        self.projector = nn.Conv1d(rnn_units, output_dim, 1)

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (batch_size, input_dim, num_sensors, seq_len)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size, num_sensors)
        """
        batch_size, _, num_sensors, _ = inputs.size()
        encoder_hidden_state = [inputs.new_zeros(batch_size,
                                                 self.rnn_units,
                                                 num_sensors) for _ in range(self.num_rnn_layers)]
        for seq in range(inputs.size(-1)):
            encoder_hidden_state = self.encoder_model(
                inputs[..., seq], self._supports, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None,
                teacher_forcing=False):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size, _, num_sensors = encoder_hidden_state[0].size()
        go_symbol = encoder_hidden_state[0].new_zeros(
            batch_size,
            self.out_dim,
            num_sensors
        )
        decoder_hidden_state = encoder_hidden_state

        decoder_input = go_symbol
        outputs = []
        for t in range(self.horizon):
            decoder_hidden_state = self.decoder_model(decoder_input,
                                                      self._supports,
                                                      decoder_hidden_state)
            decoder_input = self.projector(decoder_hidden_state[-1])
            outputs.append(decoder_input)
            if teacher_forcing:
                if teacher_forcing():  # ugly...
                    decoder_input = labels[..., t]
        outputs = torch.stack(outputs, dim=-1)
        return outputs

    def forward(self, inputs, labels=None, teacher_forcing=False):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state,
                               labels, teacher_forcing)

        return outputs
