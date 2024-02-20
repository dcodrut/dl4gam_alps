
from pathlib import Path
from typing import Union

import pandas as pd
import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from utils.sampling_utils import get_patches_gdf


def extract_inputs(ds, fp, input_settings):
    s1_bands = ds.band_data.values.astype(np.float32)

    # fill-in the data gaps with the average and build a mask which will be used for the loss
    mask_no_data = np.zeros_like(s1_bands[0]).astype(bool)
    mask_na_per_band = np.isnan(s1_bands)
    if mask_na_per_band.sum() > 0:
        idx_na = np.where(mask_na_per_band)

        # fill in the gaps with the average
        avg_per_band = np.nansum(np.nansum(s1_bands, axis=-1), axis=-1) / np.prod(s1_bands.shape[-2:])
        s1_bands[idx_na[0], idx_na[1], idx_na[2]] = avg_per_band[idx_na[0]]

        # make sure that these pixels are masked in mask_no_data too
        mask_na = mask_na_per_band.any(axis=0)
        mask_no_data |= mask_na

    # add the glacier mask to the no-data mask s.t. the loss ignores the non-glacierized area
    mask_non_glacier = ~(ds.mask_data_ok.values == 1)
    mask_no_data |= mask_non_glacier

    data = {
        's1_bands': s1_bands,
        'mask_no_data': mask_no_data,
        'mask_crt': ds.mask_crt.values == 1,
        'mask_all': ds.mask_all.values == 1,
        'fp': str(fp),
    }

    if input_settings['elevation']:
        dem = ds.dem.values.astype(np.float32) / 5000  # TODO: fix this
        # fill in the NAs with the average
        dem[np.isnan(dem)] = np.mean(dem[~np.isnan(dem)])
        data['dem'] = dem

    return data


def standardize_inputs(data, stats_df, scale_each_band):
    band_data_sdf = stats_df[stats_df.var_name.apply(lambda s: 'band' in s)]
    mu = band_data_sdf.mu.values[:len(data['s2_bands'])]
    stddev = band_data_sdf.stddev.values[:len(data['s2_bands'])]

    if not scale_each_band:
        mu[:] = mu.mean()
        stddev[:] = stddev.mean()

    data['s2_bands'] -= mu[:, None, None]
    data['s2_bands'] /= stddev[:, None, None]

    # do the same for the static variables
    for v in ['dem']:
        if v in data:
            sdf = stats_df[stats_df.var_name == v]
            mu = sdf.mu.values[0]
            stddev = sdf.stddev.values[0]
            data[v] -= mu
            data[v] /= stddev

    return data


def minmax_scale_inputs(data, stats_df, scale_each_band):
    band_data_sdf = stats_df[stats_df.var_name.apply(lambda s: 'band' in s)]
    vmin = band_data_sdf.vmin.values[:len(data['s2_bands'])]
    vmax = band_data_sdf.vmax.values[:len(data['s2_bands'])]

    if not scale_each_band:
        vmin[:] = vmin.min()
        vmax[:] = vmax.max()

    data['s2_bands'] -= vmin[:, None, None]
    data['s2_bands'] /= (vmax[:, None, None] - vmin[:, None, None])

    # do the same for the static variables
    for v in ['dem']:
        if v in data:
            # fill in the missing values with the average
            data[v][np.isnan(data[v])] = np.nanmean(data[v])

            # apply the scaling
            sdf = stats_df[stats_df.var_name == v]
            vmin = sdf.vmin.values[0]
            vmax = sdf.vmax.values[0]
            data[v] -= vmin
            data[v] /= (vmax - vmin)

    return data


class GlSegPatchDataset(Dataset):
    def __init__(self, input_settings, folder=None, fp_list=None, standardize_data=False, minmax_scale_data=False,
                 scale_each_band=True, data_stats_df=None):
        assert folder is not None or fp_list is not None

        if folder is not None:
            folder = Path(folder)
            self.fp_list = sorted(list(folder.glob('**/*.nc')))

            assert len(self.fp_list) > 0, f'No files found at: {str(folder)}'
        else:
            assert all([Path(fp).exists() for fp in fp_list])
            self.fp_list = fp_list

        self.input_settings = input_settings
        self.standardize_data = standardize_data
        self.minmax_scale_data = minmax_scale_data
        self.scale_each_band = scale_each_band
        self.data_stats_df = data_stats_df

    def process_data(self, ds, fp):
        # extract the inputs
        data = extract_inputs(ds=ds, fp=fp, input_settings=self.input_settings)

        # standardize/scale the inputs if needed
        if self.standardize_data or self.minmax_scale_data:
            assert self.standardize_data != self.minmax_scale_data
        if self.standardize_data:
            data = standardize_inputs(data, stats_df=self.data_stats_df, scale_each_band=self.scale_each_band)
        if self.minmax_scale_data:
            data = minmax_scale_inputs(data, stats_df=self.data_stats_df, scale_each_band=self.scale_each_band)

        return data

    def __getitem__(self, idx):
        # read the current file
        fp = self.fp_list[idx]
        nc = xr.open_dataset(fp, decode_coords='all')

        data = self.process_data(ds=nc, fp=fp)

        return data

    def __len__(self):
        return len(self.fp_list)


class GlSegDataset(GlSegPatchDataset):
    def __init__(self, fp, **kwargs):
        self.fp = fp
        super().__init__(folder=None, fp_list=[fp], **kwargs)

        # get all possible patches for the current entry
        self.nc = xr.open_dataset(fp, decode_coords='all').load()
        self.nc['mask_crt'] = self.nc['mask_all']
        sampling_mask = np.ones_like(self.nc.mask_all).astype(bool)
        self.patches_df = get_patches_gdf(self.nc, patch_radius=64, sampling_step=32, sampling_mask=sampling_mask)

    def __getitem__(self, idx):
        patch_shp = self.patches_df.iloc[idx:idx + 1]
        nc_patch = self.nc.rio.clip(patch_shp.geometry)

        data = self.process_data(ds=nc_patch, fp=self.fp)

        # add information regarding the location of the patch w.r.t. the entire entry
        data['patch_info'] = {k: patch_shp.iloc[0][k] for k in ['x_center', 'y_center', 'bounds_px']}

        return data

    def __len__(self):
        return len(self.patches_df)


class GlSegDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root_dir: Union[Path, str],
                 train_dir_name: str,
                 val_dir_name: str,
                 test_dir_name: str,
                 rasters_dir: str,
                 input_settings: dict,
                 standardize_data: bool,
                 minmax_scale_data: bool,
                 scale_each_band: bool,
                 data_stats_fp: str,
                 train_batch_size: int = 16,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 num_workers: int = 16,
                 pin_memory: bool = False):
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.train_dir_name = train_dir_name
        self.val_dir_name = val_dir_name
        self.test_dir_name = test_dir_name
        self.rasters_dir = rasters_dir
        self.input_settings = input_settings
        self.standardize_data = standardize_data
        self.minmax_scale_data = minmax_scale_data
        self.scale_each_band = scale_each_band
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # the following will be set when calling setup
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self.test_ds_list = None

        # prepare the standardization constants if needed
        self.data_stats_df = None
        if self.standardize_data or self.minmax_scale_data:
            data_stats_fp = Path(data_stats_fp)
            assert data_stats_fp.exists(), f'{data_stats_fp} not found'

            self.data_stats_df = pd.read_csv(data_stats_fp)

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_ds = GlSegPatchDataset(
                folder=self.data_root_dir / self.train_dir_name,
                input_settings=self.input_settings,
                standardize_data=self.standardize_data,
                minmax_scale_data=self.minmax_scale_data,
                scale_each_band=self.scale_each_band,
                data_stats_df=self.data_stats_df
            )
            self.valid_ds = GlSegPatchDataset(
                folder=self.data_root_dir / self.val_dir_name,
                input_settings=self.input_settings,
                standardize_data=self.standardize_data,
                minmax_scale_data=self.minmax_scale_data,
                scale_each_band=self.scale_each_band,
                data_stats_df=self.data_stats_df
            )
        if stage == 'test':
            self.test_ds = GlSegPatchDataset(
                folder=self.data_root_dir / self.test_dir_name,
                input_settings=self.input_settings,
                standardize_data=self.standardize_data,
                minmax_scale_data=self.minmax_scale_data,
                scale_each_band=self.scale_each_band,
                data_stats_df=self.data_stats_df
            )

    def setup_dl_per_image(self, gid_list=None):
        # get the directory of the full images
        cubes_dir = Path(self.rasters_dir)
        assert cubes_dir.exists()

        # get all entries
        cubes_fp = sorted(list(cubes_dir.glob('**/*.nc')))

        # filter the entries by id, if needed
        if gid_list is not None:
            cubes_fp = list(filter(lambda f: f.name in gid_list, cubes_fp))

        test_ds_list = []
        for fp in tqdm(cubes_fp, desc='Preparing datasets per entry'):
            test_ds_list.append(
                GlSegDataset(
                    fp=fp,
                    input_settings=self.input_settings,
                    standardize_data=self.standardize_data,
                    minmax_scale_data=self.minmax_scale_data,
                    scale_each_band=self.scale_each_band,
                    data_stats_df=self.data_stats_df
                )
            )

        return test_ds_list

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def test_dataloaders_per_image(self, gid_list):
        test_ds_list = self.setup_dl_per_image(gid_list=gid_list)

        dloaders = []
        for ds in test_ds_list:
            dloaders.append(
                DataLoader(
                    dataset=ds,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=False
                )
            )
        return dloaders
