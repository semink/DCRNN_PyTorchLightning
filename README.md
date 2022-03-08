# [Pytorch Lightning] Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

[Pytorch lightning](https://www.pytorchlightning.ai/) implementation of the original DCRNN ([paper](https://arxiv.org/abs/1707.01926), [code](https://github.com/liyaguang/DCRNN)).

## Dependencies

> **_NOTE:_** [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) should be installed in the system.

### Create a conda environment

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate dcrnn
```

## Training

```bash
# METR-LA
python run_with_lightning.py --config_filename=data/dcrnn_config.yaml --train --dataset=la

# PEMS-bay
python run_with_lightning.py --config_filename=data/dcrnn_config.yaml --train --dataset=bay
```

## Tensorboard

It is possible to run tensorboard with saved logs. Please see an example [here](https://tensorboard.dev/experiment/yzpByETpTWOAxATR8WyVYg/#scalars).

```bash
# METR-LA
tensorboard --logdir=experiments/la 

# PEMS-BAY
tensorboard --logdir=experiments/bay
```

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

```citation
@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```
