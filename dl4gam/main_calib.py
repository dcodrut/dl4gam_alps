import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

# local imports
from config import C
from utils.general import run_in_parallel
from utils.reliability_diagrams import reliability_diagram


def temperature_scale_binary_probs(probs, t):
    """
    Applies temperature scaling to binary sigmoid probabilities.
    """
    probs = np.clip(probs, 1e-6, 1. - 1e-6)
    logits = np.log(probs / (1 - probs))  # recompute the login by inverting the sigmoid
    scaled_logits = logits / t
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))
    return scaled_probs


def optimize_temperature_binary(probs, true_labels, t_lims=(0.5, 5.0)):
    """
    Finds optimal temperature for binary classification to minimize NLL.
    """

    def objective(t):
        scaled = temperature_scale_binary_probs(probs, t)
        return log_loss(true_labels, scaled)

    result = minimize_scalar(objective, bounds=t_lims, method='bounded', options={'disp': True})
    return result.x


def sample_points(fp, fraction=0.05):
    """
    Reads the ensemble predictions (containing individual members predictions) and
    return a dataframe with the sampled points.
    """
    with xr.open_dataset(fp, mask_and_scale=False) as ds:
        mask_gt = (ds.mask_all_g_id.values != -1)  # ground truth for the entire raster (could have multiple glaciers)
        mask_preds_exist = ~np.isnan(ds.pred.values)  # we might not have predictions for the entire raster

        mask_ok = ~(ds.mask_nok == 1) & mask_preds_exist
        y_true = mask_gt[mask_ok]
        y_pred_avg = ds.pred.values[mask_ok]
        y_pred_all = ds.pred_all.values[:, mask_ok]
        y_std = ds.pred_std.values[mask_ok]

        # sample points uniformly
        idx = np.arange(0, len(y_true), step=int(1 / fraction))
        y_pred_avg = y_pred_avg[idx]
        y_pred_all = y_pred_all[:, idx]
        y_true = y_true[idx]
        y_std = y_std[idx]

        tdf = pd.DataFrame({
            'entry_id': fp.parent.name,
            'y_pred_avg': y_pred_avg,
            'y_true': y_true,
            'y_std': y_std
        })

        for i in range(len(y_pred_all)):
            tdf[f"y_pred_{i + 1}"] = y_pred_all[i]

        return tdf


def sample_points_all_glaciers(fp_list, fraction=0.05, num_procs=1):
    # get a sample of the pixel-wise predictions from all glaciers
    df_all = run_in_parallel(
        sample_points,
        fp=fp_list,
        fraction=fraction,
        num_procs=num_procs,
        pbar_desc='Sampling pixels from glacier-wide predictions'
    )
    df = pd.concat(df_all)
    df = df.sort_values('entry_id')
    df['conf'] = df.y_pred_avg.apply(lambda x: 1 - x if x <= 0.5 else x)
    df['conf - y_std'] = df.conf - df.y_std
    p = df.y_pred_avg.clip(1e-5, 1 - 1e-5)
    df['shannon'] = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / 2
    df['y_pred_b'] = (df.y_pred_avg >= 0.5)
    df['pred_ok'] = (df.y_pred_b == df.y_true)
    df['pred_nok'] = ~df.pred_ok

    return df


def estimate_area_bounds_with_inc_thr(fp_pred, thresholds):
    with xr.open_dataset(fp_pred, mask_and_scale=False) as ds:
        stats_crt_g = {
            'entry_id': fp_pred.parent.name,
            'thr': [],
            'area_pred': [],
            'area_true': [],
            'area_lb': [],
            'area_ub': [],
        }

        # keep only the predictions within the 50m buffer and discard the other glaciers
        mask_all_g = (ds.mask_all_g_id.values != -1)
        mask_crt_g = (ds.mask_crt_g.values == 1)
        mask_other_g = (mask_all_g != mask_crt_g)
        ds['mask_other_g'] = (('y', 'x'), mask_other_g)
        ds = ds.where((ds.mask_crt_g_b50 == 1) & ~ds.mask_other_g)

        img_avg = ds.pred.values
        f_area = 1e2 / 1e6
        area_pred = np.nansum(img_avg >= 0.5) * f_area
        img_std = ds.pred_std.values

        all_areas_lb = []
        all_areas_ub = []

        for thr in thresholds:
            img_ub = (img_avg + img_std) >= thr
            img_lb = (img_avg - img_std) >= (1 - thr)

            area_lb = np.nansum(img_lb) * f_area
            area_ub = np.nansum(img_ub) * f_area
            all_areas_lb.append(area_lb)
            all_areas_ub.append(area_ub)

            stats_crt_g['thr'].append(thr)
            stats_crt_g['area_pred'].append(area_pred)
            stats_crt_g['area_true'].append(np.sum(mask_crt_g) * f_area)
            stats_crt_g['area_lb'].append(area_lb)
            stats_crt_g['area_ub'].append(area_ub)

        stats_crt_g_df = pd.DataFrame(stats_crt_g)
        return stats_crt_g_df


def estimate_area_bounds_with_inc_thr_all_glaciers(fp_pred_list, thresholds, num_procs=1):
    df_all = run_in_parallel(
        estimate_area_bounds_with_inc_thr,
        fp_pred=fp_pred_list,
        thresholds=thresholds,
        num_procs=num_procs,
        pbar_desc='Estimating glacier area bounds with increasing thresholds'
    )
    df = pd.concat(df_all)
    df = df.sort_values('entry_id')

    return df


def calibrate_preds(fp_in, fp_out, stats_df, ensemble_size):
    with xr.open_dataset(fp_in, mask_and_scale=False, decode_coords='all') as ds:
        scaled_probs_all = []
        for i in range(1, ensemble_size + 1):
            # calibrate the individual members
            probs = ds.pred_all.values[i - 1, :, :]
            t = stats_df.loc[stats_df.member == f"member_{i}", 'best_t'].values[0]
            scaled_probs = temperature_scale_binary_probs(probs, t)
            scaled_probs_all.append(scaled_probs)
        ds['pred_all'] = (('seed', 'y', 'x'), np.stack(scaled_probs_all, axis=0))

        # recompute the average and std deviations across ensemble members
        ds['pred'] = (('y', 'x'), np.mean(scaled_probs_all, axis=0))
        ds['pred_b'] = (ds.pred >= 0.5)
        ds['pred_std'] = (('y', 'x'), np.std(scaled_probs_all, axis=0))

        # (re)set the CRS - QGIS issues
        for k in ds.data_vars:
            ds[k].rio.write_crs(ds.rio.crs, inplace=True)

        # export the calibrated raster
        fp_out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(fp_out)


def calibrate_bounds(fp_in, fp_out, thr):
    with xr.open_dataset(fp_in, mask_and_scale=False, decode_coords='all') as ds:
        # recalculate the bounds using the calibrated ensemble average and stddev + the area-dependent threshold
        ds['pred_low_b'] = ((ds.pred - ds.pred_std) >= (1 - thr))
        ds['pred_high_b'] = ((ds.pred + ds.pred_std) >= thr)

        # (re)set the CRS - QGIS issues
        for k in ds.data_vars:
            ds[k].rio.write_crs(ds.rio.crs, inplace=True)

        # save the threshold in the attributes
        ds.attrs['decision_threshold'] = thr

        # export
        fp_out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(fp_out)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inference_dir', type=str, metavar='path/to/inference_dir', required=True,
        help='directory where the model predictions are stored',
    )
    parser.add_argument(
        '--fold', type=str, metavar='s_train|s_valid|s_test', required=True,
        help='which subset to evaluate on: either s_train, s_valid or s_test'
    )
    parser.add_argument(
        '--rasters_dir', type=str, required=False,
        help='directory where the original images are stored; if not provided, the one from the config file is used'
    )
    parser.add_argument(
        '--ensemble_size', type=int, required=True,
        help='number of ensemble members'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_dir_root = Path(args.inference_dir)
    assert inference_dir_root.exists(), f'Inference directory not found: {inference_dir_root}'

    fold = args.fold
    ensemble_size = args.ensemble_size

    # set the raster directory to the command line argument if given, otherwise use the inference dirs from the config
    if args.rasters_dir is not None:
        rasters_dir = args.rasters_dir
    else:
        rasters_dir = C.DIR_GL_RASTERS

    # replace the 'preds' subdirectory with 'calib'
    p = list(inference_dir_root.parts)
    stats_dir_root = Path(*p[:p.index('preds')]) / 'calib' / Path(*p[p.index('preds') + 1:])
    print(f'stats_dir_root = {stats_dir_root}')

    # flag whether we are using the inventory year (in which case calibration stats are computed)
    # if it's the test year, we use the calibration stats from the validation set of the inventory year
    is_inventory_year = (Path(inference_dir_root).name == 'inv')
    print(f'is_inventory_year = {is_inventory_year}')

    preds_dir = inference_dir_root / fold
    stats_dir = stats_dir_root / fold

    # get all the glacier-wide predictions
    fp_list = sorted(list(preds_dir.rglob('*.nc')))
    print(f'Found {len(fp_list)} predictions found in {preds_dir}')
    if len(fp_list) == 0:
        print(f'No predictions found for fold = {fold}. Skipping.')
        exit(0)

    # if calculated, we always use the validation set of the inventory year for the calibrated temperature(s)
    if not is_inventory_year or fold != 's_valid':
        fp = stats_dir_root.parent / 'inv' / 's_valid' / 'calibration_stats_px.csv'
        print(f"Loading calibration stats (pixel-level) from {fp}")
        stats_df_inv_val = pd.read_csv(fp)
        print(stats_df_inv_val)

    ####################################################################################################################
    # in the 1st part, we sample pixels from the glacier-wide predictions and compute the temperature scaling for each
    # ensemble member; we also plot the reliability diagrams before and after calibration

    if is_inventory_year:
        # sample pixels if we are on the test set or the validation set of the inventory year
        # (the test set is used only for showing the reliability diagrams)
        print(f"Sampling pixels from {len(fp_list)} glaciers...")
        df = sample_points_all_glaciers(fp_list, fraction=0.05 if fold == 's_test' else 0.1, num_procs=C.NUM_PROCS)
        print(df.describe())

        # export ~100k pixels to a csv file for later inspection
        fp = stats_dir / 'sampled_pixels.csv'
        fp.parent.mkdir(parents=True, exist_ok=True)
        if len(df) > 200_000:
            idx = np.arange(0, len(df), step=int(len(df) / 100_000))
            sdf = df.iloc[idx]
        else:
            sdf = df
        sdf.to_csv(fp, index=False)
        print(f"Exported {len(sdf)} pixels to {fp}")

        # calibrate each ensemble member and plot the reliability diagrams before and after calibration
        sdf = df.filter(regex=r'^y_pred.*_\d*$', axis=1)
        stats_all = []
        y_true = df.y_true.values
        for i in range(1, ensemble_size + 1):
            probs = sdf[f"y_pred_{i}"].values
            y_pred = probs >= 0.5
            y_conf = np.maximum(probs, 1 - probs)

            # plot the reliability diagram for the uncalibrated member
            calib_stats_before, fig = reliability_diagram(
                y_true, y_pred, y_conf,
                num_bins=40, draw_ece=True, draw_bin_importance="alpha", draw_averages=True,
                title="Reliability diagram - validation (uncalibrated)",
                figsize=(6, 6), dpi=200, return_fig=True, return_stats=True
            )
            fp = stats_dir / f"reliability_diagram_member_{i}_v0_uncalib.png"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fp, dpi=200, bbox_inches='tight')

            # find the best temperature for the current member
            print(f"Finding best temperature for member {i}...")
            best_t = optimize_temperature_binary(probs, y_true)

            # if we are using the validation set, we use the temperature we just computed
            if fold == 's_valid':
                actual_t = best_t
            else:
                actual_t = stats_df_inv_val.loc[stats_df_inv_val.member == f"member_{i}", 'best_t'].values[0]

            print(f"fold = {fold}; member = {i}; best_t = {best_t:.3f}; actual_t = {actual_t:.3f}")

            scaled_probs = temperature_scale_binary_probs(probs, actual_t)

            # plot the reliability diagram for the calibrated member
            y_pred = scaled_probs >= 0.5
            y_conf = np.maximum(scaled_probs, 1 - scaled_probs)
            calib_stats_after, fig = reliability_diagram(
                y_true, y_pred, y_conf,
                num_bins=40, draw_ece=True, draw_bin_importance="alpha", draw_averages=True,
                title="Reliability diagram (after calibration)",
                figsize=(6, 6), dpi=200, return_fig=True, return_stats=True
            )
            fp = stats_dir / f"reliability_diagram_member_{i}_v1_calib.png"
            fig.savefig(fp, dpi=200, bbox_inches='tight')

            # save the calibrated probabilities and the calibration stats
            df[f"y_pred_{i}_c"] = scaled_probs
            stats_all.append({
                "member": f"member_{i}",
                "best_t": best_t,
                "actual_t": actual_t,
                "ece_before": calib_stats_before['expected_calibration_error'],
                "ece_after": calib_stats_after['expected_calibration_error'],
            })

            plt.close('all')

        # recompute the average and std deviations across ensemble members
        df["y_pred_avg_c"] = df.filter(regex=r'^y_pred.*_c$').mean(axis=1)
        df["y_pred_std_c"] = df.filter(regex=r'^y_pred.*_c$').std(axis=1, ddof=0)

        # check the calibration of the ensemble average before and after calibration
        y_pred = df.y_pred_avg.values >= 0.5
        y_conf = np.maximum(df.y_pred_avg.values, 1 - df.y_pred_avg.values)
        calib_stats_before, fig = reliability_diagram(
            y_true, y_pred, y_conf,
            num_bins=40, draw_ece=True, draw_bin_importance="alpha", draw_averages=True,
            title="Reliability diagram - validation (uncalibrated)",
            figsize=(6, 6), dpi=200, return_fig=True, return_stats=True
        )
        fp = stats_dir / 'reliability_diagram_ensemble_v0_uncalib.png'
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=200, bbox_inches='tight')

        y_pred = df.y_pred_avg_c.values >= 0.5
        y_conf = np.maximum(df.y_pred_avg_c.values, 1 - df.y_pred_avg_c.values)
        calib_stats_after, fig = reliability_diagram(
            y_true, y_pred, y_conf,
            num_bins=40, draw_ece=True, draw_bin_importance="alpha", draw_averages=True,
            title="Reliability diagram (after calibration)",
            figsize=(6, 6), dpi=200, return_fig=True, return_stats=True
        )
        fp = stats_dir / f"reliability_diagram_ensemble_v1_calib.png"
        fig.savefig(fp, dpi=200, bbox_inches='tight')

        # save the calibration stats
        stats_all.append({
            "member": "ensemble",
            "best_t": None,  # we assume the ensemble average is calibrated once the individual members are calibrated
            "actual_t": None,
            "ece_before": calib_stats_before['expected_calibration_error'],
            "ece_after": calib_stats_after['expected_calibration_error'],
        })

        # export all the calibration stats
        stats_df = pd.DataFrame(stats_all)
        fp = stats_dir / f"calibration_stats_px.csv"
        stats_df.to_csv(fp, index=False)
        print(f"Calibration stats saved to {fp}")
        print(stats_df)

        # mark it as validation fold of the inventory year if needed
        if fold == 's_valid':
            stats_df_inv_val = stats_df

    ####################################################################################################################
    # in the 2nd part, we calibrate each ensemble member using the temperatures previously computed;
    # the calibrated rasters will contain the calibrated probabilities and the calibrated ensemble average
    # we export the calibrated rasters to a new directory

    preds_dir_calib_px = Path(*p[:p.index('preds')]) / 'preds_calib_px' / Path(*p[p.index('preds') + 1:]) / fold

    run_in_parallel(
        fun=calibrate_preds,
        fp_in=fp_list,
        fp_out=[preds_dir_calib_px / fp.relative_to(preds_dir) for fp in fp_list],
        stats_df=stats_df_inv_val,
        ensemble_size=ensemble_size,
        num_procs=C.NUM_PROCS,
        pbar_desc="Calibrating the rasters (pixel-wise)"
    )

    ####################################################################################################################
    # in the 3rd part, we calibrate a probability threshold for the ensemble average ± one stddev (pixelwise) s.t.
    # the resulting lower and upper bounds of the glacier areas are also calibrated
    # (i.e. capture the errors w.r.t the ground truth)

    # if calculated, we always use the validation set of the inventory year for the calibrated threshold
    if not is_inventory_year or fold != 's_valid':
        fp = stats_dir_root.parent / 'inv' / 's_valid' / 'calibration_stats_areas_thr.json'
        print(f"Loading calibration stats (area-level) from {fp}")
        with open(fp, 'r') as f:
            stats_d_inv_val = json.load(f)
            print(json.dumps(stats_d_inv_val, indent=4))
            thr_actual = stats_d_inv_val['thr_best']

    if is_inventory_year:
        # we run this step only if we are in the inventory year
        # (the test set is used only for showing the calibration diagrams)

        # read the previously calibrated rasters
        fp_list = sorted(list(preds_dir_calib_px.rglob('*.nc')))

        thr_step = 1e-3
        stats_df = estimate_area_bounds_with_inc_thr_all_glaciers(
            fp_pred_list=fp_list,
            thresholds=np.arange(0.0, 0.5 + thr_step, thr_step),
            num_procs=C.NUM_PROCS
        )

        stats_df['area_pred_error'] = stats_df.area_pred - stats_df.area_true
        stats_df['area_pred_error_abs'] = stats_df['area_pred_error'].abs()
        stats_df['area_pred_error_est'] = (stats_df.area_ub - stats_df.area_lb) / 2
        stats_df['est_err_vs_actual_err'] = (stats_df.area_pred_error_est - stats_df.area_pred_error_abs).abs()
        stats_df['pred_is_within_bounds'] = (
                (stats_df.area_lb <= stats_df.area_true) &
                (stats_df.area_true <= stats_df.area_ub)
        )

        # export the stats
        fp = stats_dir / 'calibration_stats_areas_all.csv'
        stats_df.to_csv(fp, index=False)
        print(f"Calibration stats for area bounds calibration saved to {fp}")

        # compute the best threshold
        p_target = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)  # probability mass within ±1 sigma, ~68.2%
        tdf = stats_df.groupby(['thr']).mean(numeric_only=True).reset_index()
        r_best = tdf.iloc[np.argmin(np.abs(tdf.pred_is_within_bounds.values - p_target))]
        thr_best = r_best['thr']

        # if we are using the validation set, we use the threshold we just computed
        if fold == 's_valid':
            thr_actual = thr_best
            r_actual = r_best
        else:
            # we're on testing; find the actual threshold using the validation set
            r_actual = tdf[tdf.thr == thr_actual].iloc[0]
            thr_actual = r_actual['thr']

        print(f"fold = {fold}; thr_best = {thr_best:.3f}; thr_actual = {thr_actual:.3f}")

        # export to a json file
        fp = stats_dir / 'calibration_stats_areas_thr.json'
        with open(fp, 'w') as f:  # TODO: save also the other stats for test
            stats_area_dict = {
                'thr_best': thr_best,
                'thr_actual': thr_actual,
                'p_target': p_target,
                'p_actual': r_actual.pred_is_within_bounds,
                'p_best': r_best.pred_is_within_bounds,
                'stats_actual': r_actual.to_dict(),
                'stats_best': r_best.to_dict()
            }
            json.dump(stats_area_dict, f, indent=4)
            print(f"Calibration threshold for area bounds saved to {fp}")
            print(json.dumps(stats_area_dict, indent=4))

        plt.figure(dpi=150, figsize=(15, 5))
        plt.scatter(tdf['thr'], tdf['pred_is_within_bounds'], color='C0')
        plt.axhline(y=p_target, color='C1', linestyle='--', label='p_target')
        plt.axvline(x=thr_best, color='C2', linestyle='--', label='thr_best')
        plt.axvline(x=thr_actual, color='C3', linestyle='--', label='thr_actual')
        plt.xlabel('thr')
        plt.ylabel('pred_is_within_bounds (avg)')
        plt.title('Calibration threshold for glacier areas')
        plt.legend()
        plt.grid()
        fp = stats_dir / f"calibration_thr_areas.png"
        plt.savefig(fp, bbox_inches='tight')

    ####################################################################################################################
    # 4th part: calibrate the lower and upper bounds for the glacier areas and export the rasters

    # prepare the paths to the (pixel-level) calibrated rasters
    fp_list = sorted(list(preds_dir_calib_px.rglob('*.nc')))
    preds_dir_calib = Path(*p[:p.index('preds')]) / 'preds_calib' / Path(*p[p.index('preds') + 1:]) / fold

    run_in_parallel(
        fun=calibrate_bounds,
        fp_in=fp_list,
        fp_out=[preds_dir_calib / fp.relative_to(preds_dir_calib_px) for fp in fp_list],
        thr=thr_actual,
        num_procs=C.NUM_PROCS,
        pbar_desc='Calibrating the rasters (area bounds)'
    )
