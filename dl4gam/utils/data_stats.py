import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# local imports
from .data_prep import add_glacier_masks


def compute_normalization_stats(fp):
    """
    Given the filepath to a data patch, it computes various statistics which will used to build the
    normalization constants (needed either for min-max scaling or standardization).

    :param fp: Filepath to a xarray dataset
    :return: a dictionary with the stats for the current raster
    """
    nc = xr.open_dataset(fp, decode_coords='all')
    band_data = nc.band_data.values[:13]
    list_arrays = [band_data]

    # add the other variables (except the masks and the already added band_data), assuming that they are 2D
    extra_vars = [v for v in nc if 'mask' not in v and v != 'band_data']
    for v in extra_vars:
        list_arrays.append(nc[v].values[None, ...])
    data = np.concatenate(list_arrays, axis=0)

    stats = {
        'fn': fp.name,
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
    stats['var_name'] = [f'band_{i}' for i in range(len(band_data))] + extra_vars

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


def compute_qc_stats(gl_sdf, include_shadows=True):
    assert len(gl_sdf) == 1, 'Expecting a dataframe with a single entry.'
    row = gl_sdf.iloc[0]
    fp = Path(row.fp_img)
    stats = {'fp_img': str(fp), 'entry_id': row.entry_id}
    nc = xr.open_dataset(fp, mask_and_scale=False)

    # get the band names
    band_names = list(nc.band_data.attrs['long_name'])

    # read the metadata from the same directory
    with open(fp.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)

    # get the glacier ID
    entry_id_i = row.entry_id_i
    nc = add_glacier_masks(nc_data=nc, gl_df=gl_sdf, entry_id_int=entry_id_i, buffer=50)

    # get the cloud percentage for the entire image which geedim should automatically compute
    # if the image comes from multiple tiles, use the one with the highest coverage
    _fill_portion_list = [metadata['imgs_props'][k]['FILL_PORTION'] for k in metadata['imgs_props']]
    k = list(metadata['imgs_props'].keys())[np.argmax(_fill_portion_list)]
    cloudless_p_meta = metadata['imgs_props'][k]['CLOUDLESS_PORTION'] / 100
    fill_p_meta = metadata['imgs_props'][k]['FILL_PORTION'] / 100
    stats['cloud_p_meta'] = 1 - cloudless_p_meta
    stats['fill_p_meta'] = fill_p_meta

    # prepare the cloud masks, either using the provided CLOUDLESS_MASK band
    #   or the one composed by the bands FILL_MASK | CLOUD_MASK | SHADOW_MASK (if required)
    fill_mask = nc.band_data.isel(band=band_names.index('FILL_MASK')).data
    fill_p = np.nansum(fill_mask == 1) / np.prod(fill_mask.shape)
    stats['fill_p'] = fill_p
    cloud_mask_scene_v1 = (nc.band_data.isel(band=band_names.index('CLOUDLESS_MASK')).data != 1)
    cloud_mask = nc.band_data.isel(band=band_names.index('CLOUD_MASK')).data
    cloud_mask_scene_v2 = (cloud_mask == 1) | (fill_mask != 1)
    if include_shadows:
        shadow_mask = nc.band_data.isel(band=band_names.index('SHADOW_MASK')).data
        cloud_mask_scene_v2 |= (shadow_mask == 1)

    # get the tile-level cloud percentage
    tile_level_cloud_p = metadata['imgs_props_extra'][k]['CLOUDY_PIXEL_PERCENTAGE'] / 100
    stats['tile_level_cloud_p'] = tile_level_cloud_p

    # compute the albedo, NDSI & Red \ SWIR, over the unclouded surfaces (glacier & non-glacier & within 50m buffer)
    for cloud_mask_v, cloud_mask in zip(['v1', 'v2'], [cloud_mask_scene_v1, cloud_mask_scene_v2]):
        mask_clean = ~cloud_mask
        mask_gl_clean = (nc.mask_crt_g.values == 1) & mask_clean
        mask_gl_buffer_50m_clean = (nc.mask_crt_g_b50.values == 1) & mask_clean
        mask_non_gl_clean = (nc.mask_all_g_id.values == -1) & mask_clean
        mask_non_gl_clean_buffer_50m = (nc.mask_crt_g_b50.values == 1) & (nc.mask_all_g_id.values == -1) & mask_clean
        for name, mask in zip(
                ['scene', 'gl', 'gl_b50m', 'non_gl', 'non_gl_b50m'],
                [mask_clean, mask_gl_clean, mask_gl_buffer_50m_clean, mask_non_gl_clean, mask_non_gl_clean_buffer_50m]
        ):
            if np.sum(mask) < 30:  # at least 30 clean pixels required
                continue

            # albedo
            nc_crt = nc.where(mask)
            img_bgr = nc_crt.isel(band=[band_names.index(x) for x in ['B2', 'B3', 'B4']]).band_data.values / 10000
            albedo = 0.5621 * img_bgr[0] + 0.1479 * img_bgr[1] + 0.2512 * img_bgr[2] + 0.0015
            albedo_avg = np.nanmean(albedo) if np.sum(~np.isnan(albedo)) > 0 else np.nan
            stats[f"albedo_avg_{name}_{cloud_mask_v}"] = albedo_avg

            # NDSI
            img_g_swir = nc_crt.isel(band=[band_names.index(x) for x in ['B3', 'B11']]).band_data.values
            den = (img_g_swir[0] + img_g_swir[1])
            den[den == 0] = 1
            ndsi = (img_g_swir[0] - img_g_swir[1]) / den
            ndsi_avg = np.nanmean(ndsi) if np.sum(~np.isnan(ndsi)) > 0 else np.nan
            stats[f"ndsi_avg_{name}_{cloud_mask_v}"] = ndsi_avg

            # Red / SWIR
            img_r_swir = nc_crt.isel(band=[band_names.index(x) for x in ['B4', 'B11']]).band_data.values
            den = img_r_swir[1].copy()
            den[den == 0] = 1
            r_swir = img_r_swir[0] / den
            r_swir_avg = np.nanmean(r_swir) if np.sum(~np.isnan(r_swir)) > 0 else np.nan
            stats[f"r_swir_avg_{name}_{cloud_mask_v}"] = r_swir_avg

        # add also the cloud coverage stats
        stats[f"cloud_p_scene_{cloud_mask_v}"] = np.sum(cloud_mask) / np.prod(cloud_mask.shape)
        bg_mask = (nc.mask_crt_g.values == 1)
        stats[f"cloud_p_gl_{cloud_mask_v}"] = np.sum(cloud_mask & bg_mask) / np.sum(bg_mask)
        bg_mask = (nc.mask_crt_g_b50.values == 1)
        stats[f"cloud_p_gl_b50m_{cloud_mask_v}"] = np.sum(cloud_mask & bg_mask) / np.sum(bg_mask)

    return stats
