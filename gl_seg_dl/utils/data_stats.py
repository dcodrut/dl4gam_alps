import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import json

# local imports
from .data_prep import add_glacier_masks


def compute_normalization_stats(fp):
    """
    Given the filepath to a data patch, it computes the various statistics which will used to build the
    normalization constants (needed either for min-max scaling or standardization).

    :param fp: Filepath to a xarray dataset
    :return: a dictionary with the stats for the current raster
    """
    nc = xr.open_dataset(fp)
    band_data = nc.band_data.values
    data = np.concatenate([band_data, nc.dem.values[None, ...]], axis=0)

    stats = {
        'fp': str(fp),
    }

    # add the stats for the band data
    n_list = []
    s_list = []
    ssq_list = []
    vmin_list = []
    vmax_list = []
    for i_band in range(len(data)):
        data_crt_band = data[i_band, :, :].flatten()
        all_na = np.all(np.isnan(data_crt_band))
        n_list.append(np.sum(~np.isnan(data_crt_band), axis=0) if not all_na else 0)
        s_list.append(np.nansum(data_crt_band, axis=0) if not all_na else np.nan)
        ssq_list.append(np.nansum(data_crt_band ** 2, axis=0) if not all_na else np.nan)
        vmin_list.append(np.nanmin(data_crt_band, axis=0) if not all_na else np.nan)
        vmax_list.append(np.nanmax(data_crt_band, axis=0) if not all_na else np.nan)

    stats['n'] = n_list
    stats['sum_1'] = s_list
    stats['sum_2'] = ssq_list
    stats['vmin'] = vmin_list
    stats['vmax'] = vmax_list
    stats['var_name'] = [f'band_{i}' for i in range(len(band_data))] + ['dem']

    return stats


def aggregate_normalization_stats(df):
    """
    Given the patch statistics computed using compute_normalization_stats, it combines them to estimate the
    normalization constants (needed either for min-max scaling or standardization).

    :param df: Pandas dataframe with the statistics for all the data patches.
    :return: a dataframe containing the normalization constants of each band.
    """

    # compute mean and standard deviation based only on the training folds
    stats_agg = {k: [] for k in ['var_name', 'mu', 'stddev', 'vmin', 'vmax']}
    for var_name in df.var_name.unique():
        df_r1_crt_var = df[df.var_name == var_name]
        n = max(df_r1_crt_var.n.sum(), 1)
        s1 = df_r1_crt_var.sum_2.sum()
        s2 = (df_r1_crt_var.sum_1.sum() ** 2) / n
        std = np.sqrt((s1 - s2) / n)
        mu = df_r1_crt_var.sum_1.sum() / n
        stats_agg['var_name'].append(var_name)
        stats_agg['mu'].append(mu)
        stats_agg['stddev'].append(std)
        stats_agg['vmin'].append(df_r1_crt_var.vmin.quantile(0.025))
        stats_agg['vmax'].append(df_r1_crt_var.vmax.quantile(1 - 0.025))
    df_stats_agg = pd.DataFrame(stats_agg)

    return df_stats_agg


def compute_cloud_stats(gl_sdf, buffer_px):
    assert len(gl_sdf) == 1, 'Expecting a dataframe with a single entry.'
    row = gl_sdf.iloc[0]
    fp = Path(row.fp_img)
    stats = {'fp_img': str(fp)}
    nc = xr.open_dataset(fp)

    # keep only the mask bands
    band_names = list(nc.band_data.attrs['long_name'])
    idx_bands_to_keep = [band_names.index(b) for b in band_names if 'MASK' in b]
    band_names = [band_names[i] for i in idx_bands_to_keep]
    nc = nc.isel(band=idx_bands_to_keep)

    # read the metadata from the same directory
    with open(fp.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)

    # get the glacier ID
    entry_id_i = row.entry_id_i
    dx = nc.rio.resolution()[0]
    buffer = buffer_px * dx
    nc = add_glacier_masks(nc_data=nc, gl_df=gl_sdf, entry_id_int=entry_id_i, buffer=buffer)

    # get the cloud percentage for the entire image which should be automatically computed by geedim
    # if the image comes from multiple tiles, use the one with the highest coverage
    _fill_portion_list = [metadata['imgs_props'][k]['FILL_PORTION'] for k in metadata['imgs_props']]
    k = list(metadata['imgs_props'].keys())[np.argmax(_fill_portion_list)]
    cloudless_p_meta = metadata['imgs_props'][k]['CLOUDLESS_PORTION'] / 100
    fill_p_meta = metadata['imgs_props'][k]['FILL_PORTION'] / 100
    stats['cloud_p_meta'] = 1 - cloudless_p_meta
    stats['fill_p_meta'] = fill_p_meta

    # prepare the cloud masks, either using the provided CLOUDLESS_MASK band
    #   or the one composed by the bands FILL_MASK | CLOUD_MASK | SHADOW_MASK
    fill_mask = nc.band_data.isel(band=band_names.index('FILL_MASK')).data
    fill_p = np.nansum(fill_mask == 1) / np.prod(fill_mask.shape)
    cloud_mask_v1 = (nc.band_data.isel(band=band_names.index('CLOUDLESS_MASK')).data != 1)
    cloud_p_v1 = np.nansum(cloud_mask_v1 == 1) / np.prod(cloud_mask_v1.shape)
    cloud_mask = nc.band_data.isel(band=band_names.index('CLOUD_MASK')).data
    shadow_mask = nc.band_data.isel(band=band_names.index('SHADOW_MASK')).data
    cloud_mask_v2 = (cloud_mask == 1) | (shadow_mask == 1) | (fill_mask != 1)
    cloud_p_v2 = np.nansum(cloud_mask_v2 == 1) / np.prod(cloud_mask_v2.shape)

    # compute the glacier-level cloud percentage using the 20m buffer
    gl_mask = (nc.mask_crt_g_b20.values == 1)
    cloud_mask_v1_gl_only = ((cloud_mask_v1 == 1) & gl_mask)
    cloud_mask_v2_gl_only = ((cloud_mask_v2 == 1) & gl_mask)
    cloud_p_v1_gl_only = np.sum(cloud_mask_v1_gl_only) / np.sum(gl_mask)
    cloud_p_v2_gl_only = np.sum(cloud_mask_v2_gl_only) / np.sum(gl_mask)

    # get the tile-level cloud percentage
    tile_level_cloud_p = metadata['imgs_props_extra'][k]['CLOUDY_PIXEL_PERCENTAGE'] / 100

    # save and return the stats
    stats['fill_p'] = fill_p
    stats['cloud_p_v1'] = cloud_p_v1
    stats['cloud_p_v2'] = cloud_p_v2
    stats['tile_level_cloud_p'] = tile_level_cloud_p
    stats['cloud_p_v1_gl_only'] = cloud_p_v1_gl_only
    stats['cloud_p_v2_gl_only'] = cloud_p_v2_gl_only
    stats['entry_id'] = row.entry_id

    return stats
