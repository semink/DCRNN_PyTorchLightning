import torch
import numpy as np

from model.DCRNN.model import DCRNNModel
from lib.utils import masked_MAE, masked_RMSE, masked_MAPE
import pytorch_lightning as pl


class Supervisor(pl.LightningModule):
    def __init__(self, adj_mx, scaler, hparams):
        super().__init__()
        self._model_kwargs = hparams.get('MODEL')
        self._tf_kwargs = hparams.get('TEACHER_FORCING')
        self._optim_kwargs = hparams.get('OPTIMIZER')
        self._metric_kwargs = hparams.get('METRIC')
        # data set
        self.standard_scaler = scaler
        self.input_dim = int(
            self._model_kwargs.get(
                'input_dim',
                1))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._tf_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get(
            'horizon', 1))  # for the decoder

        # setup model
        self.model = DCRNNModel(
            adj_mx,
            **self._model_kwargs)
        self.monitor_metric_name = self._metric_kwargs['monitor_metric_name']
        self.training_metric_name = self._metric_kwargs['training_metric_name']

        # optimizer setting
        self.lr = self._optim_kwargs['base_lr']
        self.example_input_array = torch.rand(
            64, self.input_dim, adj_mx.size(0), 12)
        self.save_hyperparameters('hparams')

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams, {
                self.monitor_metric_name: 0})

    def forward(self, x):
        x = self.model(x)
        return x

    def validation_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def validation_epoch_end(self, outputs):
        true = torch.cat([output['true'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        loss = self._compute_loss(true, pred)
        self.log_dict({self.monitor_metric_name: loss,
                       'step': float(self.current_epoch)}, prog_bar=True)

    def training_step(self, batch, idx):
        batches_seen = self.current_epoch * self.trainer.num_training_batches + idx
        sampling_p = self._compute_sampling_threshold(
            self._tf_kwargs['cl_decay_steps'], batches_seen)
        self.log('training/teacher_forcing_probability',
                 float(sampling_p), prog_bar=False)
        x, y = batch
        output = self.model(
            x, y, lambda: self.teacher_forcing(
                sampling_p, self.use_curriculum_learning))
        loss = self._compute_loss(y, output)
        self.log(self.training_metric_name, loss, prog_bar=False)
        return loss

    def test_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        return {'true': y, 'pred': pred}

    def test_epoch_end(self, outputs):
        true = torch.cat([output['true'] for output in outputs], dim=0)
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        losses = self._compute_all_loss(true, pred)
        loss = {'mae': losses[0],
                'rmse': losses[1],
                'mape': losses[2]}

        # error for each horizon
        for h in range(len(loss["mae"])):
            print(f"Horizon {h+1} ({5*(h+1)} min) - ", end="")
            print(f"MAE: {loss['mae'][h]:.2f}", end=", ")
            print(f"RMSE: {loss['rmse'][h]:.2f}", end=", ")
            print(f"MAPE: {loss['mape'][h]:.2f}")
            if self.logger:
                for m in loss:
                    self.logger.experiment.add_scalar(
                        f"Test/{m}", loss[m][h], h)

        # aggregated error
        print("Aggregation - ", end="")
        print(f"MAE: {loss['mae'].mean():.2f}", end=", ")
        print(f"RMSE: {loss['rmse'].mean():.2f}", end=", ")
        print(f"MAPE: {loss['mape'].mean():.2f}")

        self.test_results = pred.cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps=self._optim_kwargs['eps'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._optim_kwargs['milestones'],
                                                            gamma=self._optim_kwargs['gamma'])

        return [optimizer], [lr_scheduler]

    @ staticmethod
    def _compute_sampling_threshold(cl_decay_steps, batches_seen):
        return cl_decay_steps / (
            cl_decay_steps + np.exp(batches_seen / cl_decay_steps))

    @ staticmethod
    def teacher_forcing(sampling_p, curricular_learning_flag):
        go_flag = (torch.rand(1).item() < sampling_p) & (
            curricular_learning_flag)
        return go_flag

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x, y

    def _compute_loss(self, y_true, y_predicted, dim=(0, 1, 2, 3)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_MAE(y_predicted, y_true, dim=dim)

    def _compute_all_loss(self, y_true, y_predicted, dim=(0, 1, 2)):
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return (masked_MAE(y_predicted, y_true, dim=dim),
                masked_RMSE(y_predicted, y_true, dim=dim),
                masked_MAPE(y_predicted, y_true, dim=dim))
