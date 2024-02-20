import shapely
import shapely.geometry
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr

from .data_prep import add_extra_mask


def get_patches_gdf(nc, patch_radius, sampling_mask=None, sampling_step=None, add_center=False, add_centroid=False,
                    add_extremes=False):
    """
    Given a xarray dataset for one image, it returns a geopandas dataframe with the contours of square patches
    extracted from the dataset. The patches are only generated if they have the center pixel on the provided mask.

    :param nc: xarray dataset containing the image data and the masks based on the contours
    :param patch_radius: patch radius (in px)
    :param sampling_mask: binary mask from where to sample the centers of the patches; if not provided, we assume that
                          there is a 'mask' variable in the dataset (so we will only sample patches with the centers in
                          the contours)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :param add_center: whether to add one patch centered in the middle of the contour's box
    :param add_centroid: whether to add one patch centered in the centroid of the contour
    :param add_extremes: whether to add four patches centered in the corners of the contour
    :return: a geopandas dataframe with the contours of the generated patches
    """

    if sampling_step is None:
        assert add_center or add_centroid

    # build a mask containing all feasible patch centers
    if sampling_mask is not None:
        mask_centers = sampling_mask.copy()
    else:
        assert 'mask' in nc.data_vars
        mask_centers = (nc.mask.data == 1)

    # eliminate the edges
    mask_centers[:patch_radius, :] = False
    mask_centers[:, -patch_radius:] = False
    mask_centers[:, :patch_radius] = False
    mask_centers[-patch_radius:, :] = False

    # check if we can still sample any patch
    if mask_centers.sum() == 0:
        return None

    # get all feasible centers
    all_y_centers, all_x_centers = np.where(mask_centers)
    minx = all_x_centers.min()
    miny = all_y_centers.min()
    maxx = all_x_centers.max()
    maxy = all_y_centers.max()

    if sampling_step is not None:
        # sample the feasible centers uniformly; ensure that the first and last feasible centers are always included
        idx_x = np.asarray([p % sampling_step == 0 for p in all_x_centers])
        idx_y = np.asarray([p % sampling_step == 0 for p in all_y_centers])
        idx = idx_x & idx_y
        x_centers = all_x_centers[idx]
        y_centers = all_y_centers[idx]

        # on top of the previously sampled centers, add also the extreme corners
        if add_extremes:
            left = (minx, int(np.mean(all_y_centers[all_x_centers == minx])))
            top = (int(np.mean(all_x_centers[all_y_centers == miny])), miny)
            right = (maxx, int(np.mean(all_y_centers[all_x_centers == maxx])))
            bottom = (int(np.mean(all_x_centers[all_y_centers == maxy])), maxy)
            x_centers = np.concatenate([x_centers, [left[0], top[0], right[0], bottom[0]]])
            y_centers = np.concatenate([y_centers, [left[1], top[1], right[1], bottom[1]]])
    else:
        x_centers = []
        y_centers = []

    if add_centroid:
        x_centers = np.concatenate([x_centers, [int(np.mean(all_x_centers))]]).astype(int)
        y_centers = np.concatenate([y_centers, [int(np.mean(all_y_centers))]]).astype(int)

    if add_center:
        x_centers = np.concatenate([x_centers, [int((minx + maxx) / 2)]]).astype(int)
        y_centers = np.concatenate([y_centers, [int((miny + maxy) / 2)]]).astype(int)

    # build a geopandas dataframe with the sampled patches
    all_patches = {k: [] for k in ['x_center', 'y_center', 'bounds_px', 'bounds_m']}
    for x_center, y_center in zip(x_centers, y_centers):
        minx_patch, maxx_patch = x_center - patch_radius, x_center + patch_radius
        miny_patch, maxy_patch = y_center - patch_radius, y_center + patch_radius
        nc_crt_patch = nc.isel(x=slice(minx_patch, maxx_patch), y=slice(miny_patch, maxy_patch))

        all_patches['x_center'].append(x_center)
        all_patches['y_center'].append(y_center)
        all_patches['bounds_px'].append((minx_patch, miny_patch, maxx_patch, maxy_patch))
        all_patches['bounds_m'].append(nc_crt_patch.rio.bounds())

    patches_df = gpd.GeoDataFrame(all_patches)
    patches_df = gpd.GeoDataFrame(patches_df, geometry=patches_df.bounds_m.apply(lambda x: shapely.geometry.box(*x)))
    patches_df = patches_df.set_crs(nc.rio.crs)

    return patches_df


def data_cv_split(sdf, num_folds, valid_fraction, outlines_split_dir):
    """
    :param sdf: geopandas dataframe with the avalanche contours
    :param num_folds: how many CV folds to generate
    :param valid_fraction: the percentage of each training fold to be used as validation
    :param outlines_split_dir: output directory where the outlines split will be exported
    :return:
    """

    # regional split, assuming W to E direction
    sdf['bound_lim'] = sdf.bounds.maxx
    sdf = sdf.sort_values('bound_lim')
    sdf['Area'] = sdf.area / 1e6

    split_lims = np.linspace(0, 1, num_folds + 1)
    split_lims[-1] += 1e-4  # to include the last contour

    for i_split in range(num_folds):
        # first extract the test fold and the combined train & valid fold
        test_lims = (split_lims[i_split], split_lims[i_split + 1])
        area_cumsumf = sdf.Area.cumsum() / sdf.Area.sum()
        idx_test = (test_lims[0] <= area_cumsumf) & (area_cumsumf < test_lims[1])
        s2_df_test = sdf[idx_test]

        # compute the size of the validation relative to the entire set
        valid_fraction_adj = valid_fraction * ((num_folds - 1) / num_folds)

        # choose the valid set s.t. it acts as a clear boundary between test and train
        if i_split == 0:
            test_valid_lims = (test_lims[0], test_lims[1] + valid_fraction_adj)
        elif i_split == (num_folds - 1):
            test_valid_lims = (test_lims[0] - valid_fraction_adj, test_lims[1])
        else:
            test_valid_lims = (test_lims[0] - valid_fraction_adj / 2, test_lims[1] + valid_fraction_adj / 2)
        idx_test_valid = (test_valid_lims[0] <= area_cumsumf) & (area_cumsumf < test_valid_lims[1])
        idx_valid = idx_test_valid & (~idx_test)
        idx_train = ~idx_test_valid
        s2_df_train = sdf[idx_train]
        s2_df_valid = sdf[idx_valid]

        df_per_fold = {
            'train': s2_df_train,
            'valid': s2_df_valid,
            'test': s2_df_test,
        }

        for crt_fold, df_crt_fold in df_per_fold.items():
            print(f'Extracting outlines for split = {i_split + 1} / {num_folds}, fold = {crt_fold}')
            outlines_split_fp = Path(outlines_split_dir) / f'split_{i_split + 1}' / f'fold_{crt_fold}.shp'
            outlines_split_fp.parent.mkdir(parents=True, exist_ok=True)
            df_crt_fold.to_file(outlines_split_fp)
            print(
                f'Exported {len(df_crt_fold)} contours '
                f'out of {len(sdf)} ({len(df_crt_fold) / len(sdf) * 100:.2f}%);'
                f' actual area percentage = {df_crt_fold.Area.sum() / sdf.Area.sum() * 100:.2f}%'
                f' ({df_crt_fold.Area.sum():.2f} km^2 from a total of {sdf.Area.sum():.2f} km^2)')


def get_df_sampling_buffers(df, patch_radius, gsd, min_sampling_margin):
    """
        Build the sampling buffers (i.e.all feasible patch centers)
    """
    w = df.geometry.bounds.maxx - df.geometry.bounds.minx
    h = df.geometry.bounds.maxy - df.geometry.bounds.miny
    max_dx_px = patch_radius // 2 - np.ceil((w / 2 / gsd)).astype(int) - min_sampling_margin
    max_dx_px[max_dx_px < 0] = 0.0
    max_dy_px = patch_radius // 2 - np.ceil((h / 2 / gsd)).astype(int) - min_sampling_margin
    max_dy_px[max_dy_px < 0] = 0.0
    minx_b = df.geometry.bounds.minx - max_dx_px * gsd
    maxx_b = df.geometry.bounds.maxx + max_dx_px * gsd
    miny_b = df.geometry.bounds.miny - max_dy_px * gsd
    maxy_b = df.geometry.bounds.maxy + max_dy_px * gsd
    sampling_boxes = [shapely.box(a, b, c, d) for (a, b, c, d) in zip(minx_b, miny_b, maxx_b, maxy_b)]
    df_sampling_buffer = df.copy()
    df_sampling_buffer.geometry = sampling_boxes
    return df_sampling_buffer


def patchify_raster(fp, outlines_split_dir, num_folds, patches_dir, patch_radius, sampling_step,
                    max_n_samples_per_event, seed, pbar=False):
    """
    Using the get_patches_gdf function, it exports patches to disk for each cross-validation split, with each split
    separated into training-validation-test.
    When generating the patches, add_centroid and add_extremes will be set to True (see get_patches_gdf), which means
    at least five patches will be generated per contour.

    :param fp: netcdf file containing the raster to be patchified
    :param outlines_split_dir: directory containing the cross-validation splits
    :param num_folds: number of cross-validation folds
    :param patches_dir: output directory where the extracted patches will be saved
    :param patch_radius: patch radius (in px)
    :param sampling_step: sampling step applied both on x and y (in px);
                          if smaller than 2 * patch_radius, then patches will overlap
    :param max_n_samples_per_event: maximum number patches to sample for each contour (note that is not guaranteed that
                                    this number of patches will be reached)
    :param seed: random seed for patch sampling
    :param pbar: whether to show the progress bar while exporting the patches for each split-fold-event
    :return:
    """

    # read the image data
    nc = xr.open_dataset(fp, decode_coords='all')

    for i_split in range(1, num_folds + 1):
        for crt_fold in ['train', 'valid', 'test']:
            # for crt_fold in ['test']:
            outlines_split_fp = Path(outlines_split_dir) / f'split_{i_split}' / f'fold_{crt_fold}.shp'
            df_crt_fold = gpd.read_file(outlines_split_fp)
            df_crt_fold.fn = df_crt_fold.fn.apply(lambda s: s.replace('-', '_'))

            # keep only the contours of the current image and project them to the same CRS
            fn_to_match = fp.stem.split('GEE_')[1]
            df_crt_img = df_crt_fold[df_crt_fold.fn == fn_to_match].copy()
            df_crt_img = df_crt_img.to_crs(nc.rio.crs)

            # for each event from the current image
            for event_id in df_crt_img.id:
                df_crt_event = df_crt_img[df_crt_img.id == event_id]

                # add the rasterized contour only of the current event to the dataset
                nc = add_extra_mask(nc_data=nc, mask_name='mask_crt', gdf=df_crt_event)

                # get the sampling buffers around the current event
                df_crt_fold_sampling_buffer = get_df_sampling_buffers(
                    df=df_crt_event, patch_radius=patch_radius, gsd=10, min_sampling_margin=2
                )

                # prepare a rasterized mask for the sampling buffer (around the contours)
                _m = nc.band_data.isel(band=0).rio.clip(df_crt_fold_sampling_buffer.geometry, drop=False).values
                sampling_mask = (~np.isnan(_m)).astype(np.int8)

                # get the locations of the sampled patches
                patches_df = get_patches_gdf(
                    nc=nc,
                    sampling_mask=sampling_mask,
                    sampling_step=sampling_step,
                    patch_radius=patch_radius,
                    add_center=False,
                    add_centroid=True,
                    add_extremes=True
                )
                if patches_df is None:
                    print(f"No patches for {fn_to_match} - event {event_id}")
                    continue
                patches_df['sample_type'] = 'event'

                n_sample = max_n_samples_per_event // 2
                if len(patches_df) > n_sample:
                    patches_df = patches_df.sample(n_sample, replace=False, random_state=seed)

                # sample also some bg patches, using a (larger) buffer around the objects from the other images
                # TODO: make this faster, too many patches are being initially generated and then dropped
                df_bg = df_crt_fold[df_crt_fold.fn != fn_to_match].copy()
                df_bg = df_bg.to_crs(nc.rio.crs)
                df_crt_fold_sampling_bg = get_df_sampling_buffers(
                    df=df_bg, patch_radius=patch_radius * 2, gsd=10, min_sampling_margin=2
                )

                # prepare a rasterized mask for the sampling buffer (around the contours)
                _m = nc.band_data.isel(band=0).rio.clip(df_crt_fold_sampling_bg.geometry, drop=False).values
                sampling_mask_bg = (~np.isnan(_m)).astype(np.int8)

                # get the locations of the sampled patches
                patches_df_bg = get_patches_gdf(
                    nc=nc,
                    sampling_mask=sampling_mask_bg,
                    sampling_step=sampling_step,
                    patch_radius=patch_radius,
                    add_center=False,
                    add_centroid=True,
                    add_extremes=True
                )
                patches_df_bg['sample_type'] = 'no_event'
                n_sample = min(max_n_samples_per_event // 2, len(patches_df))
                if len(patches_df_bg) > n_sample:
                    patches_df_bg = patches_df_bg.sample(n_sample, replace=False, random_state=seed)

                patches_df_all = pd.concat((patches_df, patches_df_bg))

                # build the patches
                pbar_desc = (
                    f"Exporting patches "
                    f"(fold = {crt_fold}; split = {i_split}; image = {fn_to_match}; event = {event_id})"
                )
                for i in tqdm(range(len(patches_df_all)), desc=pbar_desc, disable=not pbar):
                    patch_shp = patches_df_all.iloc[i:i + 1]
                    crt_s2_data = nc.rio.clip(patch_shp.geometry)

                    r = patch_shp.iloc[0]
                    fn = f'{fn_to_match}_patch_{i}_xc_{r.x_center}_yc_{r.y_center}.nc'

                    patch_fp = Path(patches_dir) / f'split_{i_split}' / f'fold_{crt_fold}'
                    patch_fp = patch_fp / r.sample_type / event_id / fn
                    patch_fp.parent.mkdir(parents=True, exist_ok=True)

                    crt_s2_data.to_netcdf(patch_fp)
