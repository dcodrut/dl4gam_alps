import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import rasterio
import numpy as np

# local imports
from utils.sampling_utils import patchify_raster, data_cv_split
from utils.data_prep import add_extra_mask
from utils.general import run_in_parallel
import config as C

if __name__ == "__main__":
    data_masks = {}
    dir_raw_data = Path(C.S1.DIR_RAW_DATA)
    for mask_name in ['DESC', 'ASC']:
        fp_mask = dir_raw_data / 'mask' / f"Mask_all_{mask_name}.shp"
        print(f'Reading the data masks from {fp_mask}')
        mask_gdf = gpd.read_file(fp_mask)

        # remove the empty geometries (TODO: check this later)
        mask_gdf = mask_gdf[~mask_gdf.geometry.isna()]

        data_masks[mask_name] = mask_gdf

    avalanche_outlines_fp = dir_raw_data / 'labels/processed/all_orig/all_orig.shp'
    print(f"Reading the avalanche contours from {avalanche_outlines_fp}")
    avalanches_gdf = gpd.read_file(avalanche_outlines_fp)
    print(f"#events = {len(avalanches_gdf)}")

    # read the DEM
    dem_fp = list((dir_raw_data / 'DEM').glob('**/*.tif'))[0]
    print(f'Reading the DEM from {dem_fp}')
    nc_dem = xr.open_dataset(dem_fp).isel(band=0)

    # build netcdf files with both the images and the labels
    fp_list = sorted(list((dir_raw_data / 'images').glob('**/*.tif')))
    avalanches_gdf.fn = avalanches_gdf.fn.apply(lambda s: s.replace('-', '_'))

    for crt_fp in tqdm(fp_list, desc='Building the rasters'):
        # read the image data
        nc = xr.open_dataset(crt_fp).astype(np.float32)
        nc.band_data.rio.write_crs(nc.rio.crs, inplace=True)

        # keep only the contours of the current image and project them to the same CRS
        fn_to_match = crt_fp.stem.split('GEE_')[1]
        df_crt_img = avalanches_gdf[avalanches_gdf.fn == fn_to_match].copy()
        if len(df_crt_img) == 0:
            print(f"No matching labels for {fn_to_match}. Skipping.")
            continue
        # assert len(df_crt_img) > 0

        # add the DEM
        nc['dem'] = nc_dem.band_data.rio.reproject_match(nc, resampling=rasterio.enums.Resampling.bilinear)

        # add the rasterized contours to the dataset (using all events)
        nc = add_extra_mask(nc_data=nc, mask_name='mask_all', gdf=df_crt_img)

        # add the corresponding data mask
        mask_gdf = data_masks[crt_fp.name.split('_')[1]]
        nc = add_extra_mask(nc_data=nc, mask_name='mask_data_ok', gdf=mask_gdf)

        fp = Path(C.S1.DIR_NC_RASTERS) / crt_fp.with_suffix('.nc').name
        fp.parent.mkdir(parents=True, exist_ok=True)
        nc.to_netcdf(fp)

    # split the data regionally
    data_cv_split(
        sdf=avalanches_gdf,
        num_folds=C.S1.NUM_CV_FOLDS,
        valid_fraction=C.S1.VALID_FRACTION,
        outlines_split_dir=C.S1.DIR_OUTLINES_SPLIT
    )

    # save one dataframe which maps each glacier to the corresponding fold of each split (will be used at inference)
    df_list = []
    for i_split in range(1, C.S1.NUM_CV_FOLDS + 1):
        crt_split_dir = Path(C.S1.DIR_OUTLINES_SPLIT) / f"split_{i_split}"
        for fold in ['train', 'valid', 'test']:
            fp = crt_split_dir / f"fold_{fold}.shp"
            df = gpd.read_file(fp)[['id']].rename(columns={'id': 'entry_id'})
            df['split'] = f"split_{i_split}"
            df['fold'] = f"fold_{fold}"
            df_list.append(df)

    df_all = pd.concat(df_list)
    df_all = df_all.pivot(index='entry_id', columns='split', values='fold').reset_index()
    fp = Path(C.S1.DIR_OUTLINES_SPLIT) / 'map_all_splits_all_folds.csv'
    df_all.to_csv(fp, index=False)
    print(f"Dataframe with the split-fold mapping saved to {fp}")

    # prepare the list of rasters to be patchified
    fp_list = sorted(list((Path(C.S1.DIR_NC_RASTERS).glob('**/*.nc'))))
    assert len(fp_list) > 0, f'No netcdf files found in {fp_list}'

    # assign a unique seed to each, to increase the chances that sampled patches are different
    seed_list = [42 + i for i in range(len(fp_list))]

    # patchify the data in parallel
    run_in_parallel(
        patchify_raster,
        fp=fp_list,
        outlines_split_dir=C.S1.DIR_OUTLINES_SPLIT,
        num_folds=C.S1.NUM_CV_FOLDS,
        patches_dir=C.S1.DIR_GL_PATCHES,
        patch_radius=C.S1.PATCH_RADIUS,
        sampling_step=C.S1.SAMPLING_STEP,
        max_n_samples_per_event=10,
        num_cores=C.S1.NUM_CORES_PREP_DATA,
        seed=seed_list,
        pbar=True
    )
