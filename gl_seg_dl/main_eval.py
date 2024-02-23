from argparse import ArgumentParser
from pathlib import Path
import multiprocessing
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

# local imports
import config as C


def compute_stats(filepath):
    nc = xr.open_dataset(filepath)

    # get the predictions and the ground truth
    y_pred = (nc.pred_b_masked.values == 1)
    y_true = (nc.mask_all.values == 1)

    # apply the mask
    mask_ok = (nc.mask_data_ok.values == 1)
    y_true = y_true[mask_ok]
    y_pred = y_pred[mask_ok]

    tp = (y_true & (y_pred == y_true)).sum()
    tn = (~y_true & (y_pred == y_true)).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()

    if y_true.sum() == 0 or y_pred.sum() == 0:  # no avalanches (ground truth or predicted)
        precision, recall, f1, iou = [np.nan] * 4
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

    stats = {
        'filepath': filepath,
        'img_name': Path(filepath).stem,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

    return stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--inference_dir', type=str, metavar='path/to/inference_dir', required=True,
        help='directory where the model predictions are stored',
    )

    args = parser.parse_args()
    inference_dir_root = Path(args.inference_dir)
    print(f'inference_dir_root = {inference_dir_root}')
    assert inference_dir_root.exists()

    stats_dir_root = Path(inference_dir_root.parent.parent.parent, 'stats', inference_dir_root.name)

    for fold in ('s_train', 's_valid', 's_test'):
        preds_dir = inference_dir_root / fold
        fp_list = list(preds_dir.glob('**/*.nc'))
        print(f'fold = {fold}; #glaciers = {len(fp_list)}')
        if len(fp_list) == 0:
            print(f'No predictions found for fold = {fold}. Skipping.')
            continue

        with multiprocessing.Pool(C.S1.NUM_CORES_EVAL) as pool:
            all_metrics = []
            for metrics in tqdm(
                    pool.imap_unordered(compute_stats, fp_list, chunksize=1), total=len(fp_list),
                    desc=f'Computing evaluation metrics'):
                all_metrics.append(metrics)
            metrics_df = pd.DataFrame.from_records(all_metrics)

            stats_fp = stats_dir_root / fold / f'stats_per_image.csv'
            stats_fp.parent.mkdir(parents=True, exist_ok=True)
            metrics_df = metrics_df.sort_values('fp')
            metrics_df.to_csv(stats_fp, index=False)
            print(f'Evaluation metrics exported to {stats_fp}')
