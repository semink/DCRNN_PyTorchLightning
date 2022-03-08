from model.DCRNN.dataset import DataModule

from argparse import ArgumentParser
import yaml

from model.DCRNN.supervisor import Supervisor as DCRNNSupervisor

from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import os


def data_load(dataset, batch_size, seq_len, horizon,
              num_workers=os.cpu_count(), **kwargs):
    dm = DataModule(dataset, batch_size, seq_len,
                    horizon, num_workers)
    dm.prepare_data()
    return dm


def train_model(dataset, dparams, hparams):
    dm = data_load(dataset, **dparams['DATA'],
                   horizon=hparams['MODEL']['horizon'])

    model = DCRNNSupervisor(
        hparams=config['HPARAMS'],
        adj_mx=dm.get_adj(),
        scaler=dm.get_scaler())

    dparams['LOG']['save_dir'] = os.path.join(
        dparams['LOG']['save_dir'], dataset)
    logger = TensorBoardLogger(
        **dparams['LOG'],
        default_hp_metric=False)

    trainer = Trainer(
        **dparams['TRAINER'],
        callbacks=[RichModelSummary(**dparams['SUMMARY']),
                   RichProgressBar(),
                   LearningRateMonitor(logging_interval='epoch'),
                   ModelCheckpoint(
            monitor=hparams['METRIC']['monitor_metric_name']),
            EarlyStopping(
            monitor=hparams['METRIC']['monitor_metric_name'],
            **dparams['EARLY_STOPPING'])
        ],
        logger=logger)

    trainer.fit(model, dm)
    result = trainer.test(model, dm, ckpt_path='best')


def test_model(dataset, dparams, hparams):
    dm = data_load(dataset, **dparams['DATA'],
                   horizon=hparams['MODEL']['horizon'])
    model = DCRNNSupervisor.load_from_checkpoint(
        dparams['TEST']['checkpoint'][dataset], scaler=dm.get_scaler(), adj_mx=dm.get_adj())

    trainer = Trainer(gpus=dparams['TRAINER']['gpus'],
                      callbacks=[RichProgressBar()],
                      enable_checkpointing=False,
                      logger=False)
    result = trainer.test(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Program specific args
    parser.add_argument("--config", type=str,
                        default="data/dcrnn_config.yaml", help="Configuration file path")
    parser.add_argument("--dataset", type=str,
                        default="la", help="name of the dataset. it should be either la or bay.",
                        choices=['la', 'bay'])
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')

    args = parser.parse_args()

    assert (
        not args.train) | (
        not args.test), "Only one of --train and --test flags can be turned on."
    assert (
        args.train) | (
        args.test), "At least one of --train and --test flags must be turned on."

    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.train:
        train_model(args.dataset, config['NONPARAMS'], config['HPARAMS'])
    elif args.test:
        test_model(args.dataset, config['NONPARAMS'], config['HPARAMS'])
