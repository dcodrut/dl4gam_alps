from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import pytorch_lightning as pl
import yaml
from tqdm import tqdm
import logging

# local imports
import config as C
import models
from task.data import GlSegDataModule
from task.seg import GlSegTask
from utils.general import str2bool

# Logger (console and TensorBoard)
root_logger = logging.getLogger('pytorch_lightning')
root_logger.setLevel(logging.INFO)
fmt = '[%(levelname)s] - %(asctime)s - %(name)s: %(message)s (%(filename)s:%(funcName)s:%(lineno)d)'
root_logger.handlers[0].setFormatter(logging.Formatter(fmt))
logger = logging.getLogger('pytorch_lightning.core')


def test_model(settings, fold, test_per_entry=False, glacier_id_list=None, checkpoint=None):
    logger.info(f'Settings: {settings}')

    # Data
    data_params = settings['data']
    data_params['input_settings'] = settings['model']['inputs']
    dm = GlSegDataModule(**data_params)
    dm.train_shuffle = False  # disable shuffling for the training dataloader

    # Model
    model_class = getattr(models, settings['model']['class'])
    model = model_class(
        input_settings=settings['model']['inputs'] if 'inputs' in settings['model'] else None,
        training_settings=settings['model']['training_settings'] if 'training_settings' in settings['model'] else None,
        model_name=settings['model']['name'],
        model_args=settings['model']['args'],
    )

    # Task
    task_params = settings['task']
    task = GlSegTask(model=model, task_params=task_params)
    logger.info(f'Loading model from {checkpoint}')
    device = f"cuda:{settings['trainer']['devices'][0]}" if settings['trainer']['accelerator'] == 'gpu' else 'cpu'
    if checkpoint is not None:  # allow using non-trainable models
        task.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location=device,
            model=model,
            task_params=task_params
        )

    # Trainer
    trainer_dict = settings['trainer']
    trainer = pl.Trainer(**trainer_dict)

    logger.info(f'test_per_entry = {test_per_entry}')
    checkpoint_root_dir = Path(checkpoint).parent.parent
    if test_per_entry:
        ds_name = Path(data_params['rasters_dir']).parent.name
        subdir = Path(data_params['rasters_dir']).name
        root_outdir = checkpoint_root_dir / 'output' / 'preds' / ds_name / subdir
    else:
        ds_name = Path(data_params['data_root_dir']).parent.name
        root_outdir = checkpoint_root_dir / 'output' / 'stats' / ds_name

    assert fold in ['s_train', 's_valid', 's_test']
    logger.info(f'Testing for fold = {fold}')

    task.outdir = root_outdir / fold
    logger.info(f"output directory = {task.outdir}")
    if not test_per_entry:
        assert fold in ('s_train', 's_valid', 's_test')
        dm.setup('test' if fold == 's_test' else 'fit')
        if fold == 's_train':
            dl = dm.train_dataloader()
        elif fold == 's_valid':
            dl = dm.val_dataloader()
        else:
            dl = dm.test_dataloader()

        results = trainer.validate(model=task, dataloaders=dl)
        logger.info(f'results = {results}')
    else:
        dir_fp = Path(dm.rasters_dir)
        logger.info(f'Reading the entries IDs based on the rasters from {dir_fp}')
        fp_list = list(dir_fp.glob('**/*.nc'))
        glacier_id_list_crt_dir = set([p.name for p in fp_list])
        logger.info(f'#entries in the current rasters dir = {len(glacier_id_list_crt_dir)}')

        dl_list = dm.test_dataloaders_per_image(gid_list=glacier_id_list_crt_dir)
        for dl in tqdm(dl_list, desc='Testing per glacier'):
            trainer.test(model=task, dataloaders=dl)


def get_best_model_ckpt(checkpoint_dir, metric_name='val_loss_epoch', sort_method='min'):
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists(), f'{checkpoint_dir} not found'
    assert sort_method in ('max', 'min')

    ckpt_list = sorted(list(checkpoint_dir.glob('*.ckpt')))
    ens_list = np.array([float(p.stem.split(f'{metric_name}=')[1]) for p in ckpt_list if metric_name in str(p)])

    # get the index of the last best value
    sort_method_f = np.argmax if sort_method == 'max' else np.argmin
    i_best = len(ens_list) - sort_method_f(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_best]

    return ckpt_best


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--settings_fp', type=str, metavar='path/to/settings.yaml', help='yaml with all the settings')
    parser.add_argument('--checkpoint', type=str, metavar='path/to/checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--checkpoint_dir', type=str, metavar='path/to/checkpoint_dir',
                        help='a directory from which the model with the lowest L1 distance will be selected '
                             '(alternative to checkpoint_file)', default=None)
    parser.add_argument('--fold', type=str, metavar='s_train|s_valid|s_test', required=True,
                        help='which subset to test on: either s_train, s_valid or s_test')
    parser.add_argument('--test_per_entry', type=str2bool, required=True,
                        help='whether to apply the model separately for each glacier instead of using the patches'
                             '(by generating in-memory all the patches)')
    parser.add_argument('--rasters_dir', type=str, required=False,
                        help='directory on which to test the model, for the case when test_per_entry is True; '
                             'if not provided, the one from the config file is used'
                        )
    parser.add_argument('--split_fp', type=str, required=False,
                        help='path to a (geo)pandas dataframe containing the list of entries to be tested '
                             'and the corresponding fold of each split. Either .csv or .shp.')
    parser.add_argument('--gpu_id', type=int, default=None, required=False
                        , help='GPU ID (if not given, the one from the config is used)'
                        )
    args = parser.parse_args()

    # prepare the checkpoint
    if args.checkpoint_dir is not None:
        # get the best checkpoint
        checkpoint_file = get_best_model_ckpt(
            checkpoint_dir=args.checkpoint_dir,
            metric_name='BinaryJaccardIndex_val_epoch_avg_per_e',
            sort_method='max'
        )
    else:
        checkpoint_file = args.checkpoint

    # get the settings (assuming it was saved in the model's results directory if not given)
    if args.settings_fp is None:
        model_dir = Path(checkpoint_file).parent.parent
        settings_fp = model_dir / 'settings.yaml'
    else:
        settings_fp = args.settings_fp

    with open(settings_fp, 'r') as fp:
        all_settings = yaml.load(fp, Loader=yaml.FullLoader)

    # overwrite the gpu id if provided
    if args.gpu_id is not None:
        all_settings['trainer']['devices'] = [args.gpu_id]

    # set the raster directory to the command line argument if given, otherwise use the inference dirs from the config
    if args.rasters_dir is not None:
        infer_dir_list = [args.rasters_dir]
    else:
        infer_dir_list = C.S1.DIRS_INFER

    # choose the entries for the specified fold using the given shapefile
    entry_ids = None
    if args.test_per_entry:
        assert args.split_fp is not None
        fp = Path(args.split_fp)
        assert fp.suffix in ('.csv', '.shp')
        logger.info(f"Reading the split dataframe from {fp}")
        split_df = gpd.read_file(fp, dtype={'entry_id': str})

        # get the split on which the current model was trained on
        split_name = f"split_{str(checkpoint_file).split('split_')[1].split('/')[0]}"

        # get the list of entries for the specified fold
        fold_name = f"fold_{args.fold[2:]}"
        entry_ids = sorted(list(split_df[split_df[split_name] == fold_name].entry_id))
        logger.info(f"split = {split_name}; fold = {fold_name}; #entries = {len(entry_ids)}")

    for infer_dir in infer_dir_list:
        assert Path(infer_dir).exists(), f"{infer_dir} does not exist"
        all_settings['data']['rasters_dir'] = infer_dir
        test_model(
            settings=all_settings,
            checkpoint=checkpoint_file,
            test_per_entry=args.test_per_entry,
            glacier_id_list=entry_ids,
            fold=args.fold,
        )
