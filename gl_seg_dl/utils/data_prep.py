import numpy as np


def add_extra_mask(nc_data, mask_name, gdf):
    # project the outlines on the same CRS as the raster data
    gdf_proj = gdf.to_crs(nc_data.rio.crs)

    # get all the polys that intersect the current raster (multiple contours can be covered)
    tmp_raster = nc_data.band_data.isel(band=0).rio.clip(gdf_proj.geometry, drop=False).values
    mask = (~np.isnan(tmp_raster)).astype(np.int8)
    nc_data[mask_name] = (('y', 'x'), mask)
    nc_data[mask_name].attrs['_FillValue'] = -1
    nc_data[mask_name].rio.write_crs(nc_data.rio.crs, inplace=True)

    return nc_data
