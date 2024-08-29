import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.loggers
import logging
import yaml
import json

# local imports
from config import C
import models
from task.data import GlSegDataModule
from task.seg import GlSegTask

# https://github.com/PyTorchLightning/pytorch-lightning/issues/5225
if 'SLURM_NTASKS' in os.environ:
    del os.environ['SLURM_NTASKS']
if 'SLURM_JOB_NAME' in os.environ:
    del os.environ['SLURM_JOB_NAME']


def train_model(settings: dict):
    # Logger (console and TensorBoard)
    tb_logger = pl.loggers.TensorBoardLogger(**settings['logger'])
    Path(tb_logger.log_dir).mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger('pytorch_lightning')
    root_logger.setLevel(logging.INFO)
    fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
    root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
    logger = logging.getLogger('pytorch_lightning.core')
    fh = logging.FileHandler(Path(tb_logger.log_dir, 'console.log'))
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    # fix the seed
    seed = settings['task']['seed']
    pl.seed_everything(seed, workers=True)
    logger.info(f'Initial settings: {settings}')

    # Data
    data_params = settings['data'].copy()
    data_params['input_settings'] = settings['model']['inputs']
    dm = GlSegDataModule(**data_params)

    # Model
    model_class = getattr(models, settings['model']['class'])
    model = model_class(
        input_settings=settings['model']['inputs'] if 'inputs' in settings['model'] else None,
        other_settings=settings['model']['other'] if 'other' in settings['model'] else None,
        model_args=settings['model']['args'],
        model_name=settings['model']['name']
    )

    # Task
    task_params = settings['task']
    task = GlSegTask(model=model, task_params=task_params)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='w_JaccardIndex_val_epoch_avg_per_g',
        filename='ckpt-{epoch:02d}-{w_JaccardIndex_val_epoch_avg_per_g:.4f}',
        save_top_k=1,
        save_last=True,
        mode='max',
        every_n_epochs=1
    )
    summary = pl.callbacks.ModelSummary(max_depth=-1)

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback, summary], **trainer_dict)

    # Save the config file, after adding the slurm job id (if exists)
    path_out = Path(tb_logger.log_dir, 'settings.yaml')
    setting_dict_save = settings.copy()
    setting_dict_save['SLURM_JOBID'] = os.environ['SLURM_JOBID'] if 'SLURM_JOBID' in os.environ else None
    with open(path_out, 'w') as fp:
        yaml.dump(setting_dict_save, fp, sort_keys=False)
    logger.info(f'Settings saved to to {path_out}')
    logger.info(f'Exported settings:\n{json.dumps(settings, sort_keys=False, indent=4)}')

    trainer.fit(task, dm)
    logger.info(f'Best model {checkpoint_callback.best_model_path} with score {checkpoint_callback.best_model_score}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--settings_fp',
        type=str, metavar='path/to/settings.yaml', help='yaml with all the settings', required=True)
    parser.add_argument(
        '--seed',
        type=int, help='training seed (if not given, the one from the config is used)', default=None, required=False)
    parser.add_argument(
        '--split',
        type=str, help='training split (if not given, the one from the config is used)', default=None, required=False,
        choices={f"split_{i + 1}" for i in range(C.NUM_CV_FOLDS)}
    )
    parser.add_argument(
        '--gpu_id',
        type=int, help='GPU ID (if not given, the one from the config is used)', default=None, required=False)
    args = parser.parse_args()

    # read the settings
    assert Path(args.settings_fp).exists(), f'Settings file ({args.settings_fp}) not found.'
    with open(args.settings_fp, 'r') as f:
        all_settings = yaml.load(f, Loader=yaml.FullLoader)

    # overwrite the seed if provided
    if args.seed is not None:
        all_settings['task']['seed'] = args.seed

    # overwrite the split if provided
    if args.split is not None:
        all_settings['data']['data_root_dir'] = str(Path(all_settings['data']['data_root_dir']).parent / args.split)
        all_settings['data']['data_stats_fp'] = str(
            Path(all_settings['data']['data_stats_fp']).parent.parent / args.split / 'stats_train_patches_agg.csv')
        all_settings['logger']['name'] = str(Path(all_settings['logger']['name']).parent / args.split)

    # add the seed as a subfolder
    all_settings['logger']['name'] = str(Path(all_settings['logger']['name']) / f'seed_{all_settings["task"]["seed"]}')

    # overwrite the gpu id if provided
    if args.gpu_id is not None:
        all_settings['trainer']['devices'] = [args.gpu_id]

    train_model(all_settings)
