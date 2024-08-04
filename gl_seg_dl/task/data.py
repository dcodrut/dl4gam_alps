from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.sampling_utils import get_patches_gdf


def extract_inputs(ds, fp, input_settings):
    band_names = ds.band_data.attrs['long_name']
    idx_bands = [band_names.index(b) for b in input_settings['bands_input']]
    band_data = ds.band_data.isel(band=idx_bands).values.astype(np.float32)

    # build the data mask
    mask_no_data = np.zeros_like(ds.band_data.isel(band=0).values).astype(bool)
    mask_names = input_settings['bands_mask']
    for mask_name in mask_names:
        # check which value do we expect for the mask
        if mask_name[0] == '~':
            mask_name = mask_name[1:]
            ok_value = 1
        else:
            ok_value = 0
        # add the current mask
        assert mask_name in band_names, f"Expecting one of the following values for masks: {band_names}"
        crt_mask_no_data = ~(ds.band_data.values[band_names.index(mask_name)] == ok_value)
        mask_no_data |= crt_mask_no_data

    # include to the nodata mask any NaN pixel
    # (it happened once that a few pixels were missing only from the last band but the mask did not include them)
    mask_na_per_band = np.isnan(band_data)
    if mask_na_per_band.sum() > 0:
        idx_na = np.where(mask_na_per_band)

        # fill in the gaps with the average
        avg_per_band = np.nansum(np.nansum(band_data, axis=-1), axis=-1) / np.prod(band_data.shape[-2:])
        band_data[idx_na[0], idx_na[1], idx_na[2]] = avg_per_band[idx_na[0]]

        # make sure that these pixels are masked in mask_no_data too
        mask_na = mask_na_per_band.any(axis=0)
        mask_no_data |= mask_na

    data = {
        'band_data': band_data,
        'mask_no_data': mask_no_data,
        'mask_crt_g': ds.mask_crt_g.values == 1,
        'mask_all_g': ~np.isnan(ds.mask_all_g_id.values),
        'fp': str(fp),
        'glacier_area': ds.attrs['glacier_area']
    }

    # add the debris masks (priority: SGI)
    mask_debris_sgi_2016_crt_g = (ds.mask_debris_sgi_2016.values == 1) & data['mask_crt_g']
    mask_debris_sgi_2016_all_g = (ds.mask_debris_sgi_2016.values == 1) & data['mask_all_g']
    mask_debris_sherler_2018_crt_g = (ds.mask_debris_scherler_2018.values == 1) & data['mask_crt_g']
    mask_debris_sherler_2018_all_g = (ds.mask_debris_scherler_2018.values == 1) & data['mask_all_g']
    if mask_debris_sgi_2016_crt_g.sum() > 0:
        data['mask_debris_crt_g'] = mask_debris_sgi_2016_crt_g
        data['mask_debris_all_g'] = mask_debris_sgi_2016_all_g
        data['debris_source'] = 'sgi_2016'
    else:
        data['mask_debris_crt_g'] = mask_debris_sherler_2018_crt_g
        data['mask_debris_all_g'] = mask_debris_sherler_2018_all_g
        data['debris_source'] = 'sherler_2018'

    if input_settings['elevation']:
        dem = ds.dem.values.astype(np.float32)
        # fill in the NAs with the average
        dem[np.isnan(dem)] = np.mean(dem[~np.isnan(dem)])
        data['dem'] = dem

    if input_settings['dhdt']:
        dhdt = ds.dhdt.values.astype(np.float32)
        # fill in the NAs with zeros
        dhdt[np.isnan(dhdt)] = 0.0
        data['dhdt'] = dhdt

    if input_settings['optical_indices']:
        # compute the NDSI, NDVI and NDWI indices
        # NDSI = (Green - SWIR) / (Green + SWIR)
        # NDVI = (NIR - Red) / (NIR + Red)
        # NDWI = (Green - NIR) / (Green + NIR)
        swir = band_data[input_settings['bands_input'].index('B12')]
        r = band_data[input_settings['bands_input'].index('B4')]
        g = band_data[input_settings['bands_input'].index('B3')]
        nir = band_data[input_settings['bands_input'].index('B8')]

        # NDSI
        den = g + swir
        den[den == 0] = 1  # avoid division by zero
        data['ndsi'] = (g - swir) / den

        # NDVI
        den = nir + r
        den[den == 0] = 1  # avoid division by zero
        data['ndvi'] = (nir - r) / den

        # NDWI
        den = g + nir
        den[den == 0] = 1  # avoid division by zero
        data['ndwi'] = (g - nir) / den

    if input_settings['dem_features']:
        data['slope'] = ds.slope.values.astype(np.float32) / 90.  # scale the slope to [0, 1]

        # compute the sine and cosine of the aspect
        data['aspect_sin'] = np.sin(ds.aspect.values.astype(np.float32) * np.pi / 180)
        data['aspect_cos'] = np.cos(ds.aspect.values.astype(np.float32) * np.pi / 180)

        # add the planform curvature, profile curvature, terrain ruggedness index, which will be later normalized
        for k in ['planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
            data[k] = ds[k].values.astype(np.float32)

    return data


def standardize_inputs(data, stats_df, scale_each_band):
    band_data_sdf = stats_df[stats_df.var_name.apply(lambda s: 'band' in s)]
    mu = band_data_sdf.mu.values[:len(data['band_data'])]
    stddev = band_data_sdf.stddev.values[:len(data['band_data'])]

    if not scale_each_band:
        mu[:] = mu.mean()
        stddev[:] = stddev.mean()

    data['band_data'] -= mu[:, None, None]
    data['band_data'] /= stddev[:, None, None]

    # do the same for the static variables that need to be standardized
    for v in ['dem', 'dhdt', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
        if v in data:
            sdf = stats_df[stats_df.var_name == v]
            assert len(sdf) == 1, f"Expecting one stats row for {v}"
            mu = sdf.mu.values[0]
            stddev = sdf.stddev.values[0]
            data[v] -= mu
            data[v] /= stddev

    return data


def minmax_scale_inputs(data, stats_df, scale_each_band):
    band_data_sdf = stats_df[stats_df.var_name.apply(lambda s: 'band' in s)]
    vmin = band_data_sdf.vmin.values[:len(data['band_data'])]
    vmax = band_data_sdf.vmax.values[:len(data['band_data'])]

    if not scale_each_band:
        vmin[:] = vmin.min()
        vmax[:] = vmax.max()

    data['band_data'] -= vmin[:, None, None]
    data['band_data'] /= (vmax[:, None, None] - vmin[:, None, None])

    # do the same for the static variables that need to be normalized
    for v in ['dem', 'dhdt', 'planform_curvature', 'profile_curvature', 'terrain_ruggedness_index']:
        if v in data:
            # apply the scaling
            sdf = stats_df[stats_df.var_name == v]
            assert len(sdf) == 1, f"Expecting one stats row for {v}"
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
    def __init__(self, fp, patch_radius=None, sampling_step=None, preload_data=False, **kwargs):
        self.fp = fp
        super().__init__(folder=None, fp_list=[fp], **kwargs)

        # get all possible patches for the glacier
        if preload_data:
            self.nc = xr.load_dataset(fp, decode_coords='all')
        else:
            self.nc = xr.open_dataset(fp, decode_coords='all')
        self.patches_df = get_patches_gdf(
            self.nc,
            patch_radius=patch_radius,
            sampling_step=sampling_step,
            add_center=False,
            add_centroid=True,
            add_extremes=True
        )

    def __getitem__(self, idx):
        patch_shp = self.patches_df.iloc[idx:idx + 1]
        nc_patch = self.nc.rio.clip(patch_shp.geometry)

        data = self.process_data(ds=nc_patch, fp=self.fp)

        # add information regarding the location of the patch w.r.t. the entire glacier
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

    def setup_dl_per_glacier(self, gid_list=None, patch_radius=None, sampling_step=None, preload_data=False):
        # get the directory of the full glacier cubes
        cubes_dir = Path(self.rasters_dir)
        assert cubes_dir.exists()

        # get all glaciers
        cubes_fp = sorted(list(cubes_dir.glob('**/*.nc')))

        # filter the glaciers by id, if needed
        if gid_list is not None:
            cubes_fp = list(filter(lambda f: f.parent.name in gid_list, cubes_fp))

        test_ds_list = []
        for fp in tqdm(cubes_fp, desc='Preparing datasets per glacier'):
            test_ds_list.append(
                GlSegDataset(
                    fp=fp,
                    input_settings=self.input_settings,
                    standardize_data=self.standardize_data,
                    minmax_scale_data=self.minmax_scale_data,
                    scale_each_band=self.scale_each_band,
                    data_stats_df=self.data_stats_df,
                    patch_radius=patch_radius,
                    sampling_step=sampling_step,
                    preload_data=preload_data
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
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            drop_last=False
        )

    def test_dataloaders_per_glacier(self, gid_list, patch_radius=None, sampling_step=None, preload_data=False):
        test_ds_list = self.setup_dl_per_glacier(
            gid_list=gid_list,
            patch_radius=patch_radius,
            sampling_step=sampling_step,
            preload_data=preload_data
        )

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
