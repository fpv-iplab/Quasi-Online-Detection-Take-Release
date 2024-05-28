from typing import Tuple, List, Union

import numpy as np
import pandas as pd

from config.config_loader import cfg


def _interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> Union[float, np.floating]:
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        prec (np.ndarray): Precision values.
        rec (np.ndarray): Recall values.

    Returns:
        Union[float, np.floating]: The interpolated average precision.
    """
    mprec = np.hstack([[0.0], prec, [0.0]])
    mrec = np.hstack([[0.0], rec, [1.0]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(float(mprec[i]), float(mprec[i + 1]))
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1

    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return float(ap)


def _temporal_offset(target_AS: float, candidate_AS: np.ndarray) -> np.ndarray:
    """Compute the temporal offset between a target AS and all the test AS.

    Args:
        target_AS (float): Starting point of the target action segment.
        candidate_AS (np.ndarray): Array of starting points of candidate action segments.

    Returns:
        np.ndarray: Array of temporal offsets.
    """
    # return np.absolute(target_AS - candidate_AS)
    result = np.absolute(candidate_AS - target_AS)
    return result


def _ap_depth_at_recall_x(prec: np.ndarray, rec: np.ndarray, x: float) -> float:
    """Calculate the AP depth at recall X%.

    Args:
        prec (np.ndarray): Precision values.
        rec (np.ndarray): Recall values.
        x (float): Recall threshold.

    Returns:
        float: The average precision at the given recall threshold.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    idx = np.max(np.where(mprec <= x))
    ap = np.mean(mrec[:idx + 1])

    return ap


def compute_average_precision_detection(ground_truth: pd.DataFrame,
                                        prediction: pd.DataFrame,
                                        tOffset_thresholds: np.ndarray,
                                        latency_threshold: bool = False,
                                        latency_threshold_chunk: float = 1.0
                                        ) -> Tuple[np.ndarray, List[float]]:
    """Compute average precision between ground truth and
    predictions data frames. If multiple predictions occur for the same
    predicted segment, only the one with the smallest offset is matched as
    true positive.

    Args:
        ground_truth (pd.DataFrame): Data frame containing the ground truth instances.
            Required fields: ['video-id', 't-start']
        prediction (pd.DataFrame): Data frame containing the prediction instances.
            Required fields: ['video-id', 't-start', 'score']
        tOffset_thresholds (np.ndarray): Temporal offset threshold in seconds.
        latency_threshold (bool, optional): Whether to apply latency threshold. Defaults to False.
        latency_threshold_chunk (float, optional): Latency threshold chunk size. Defaults to 1.0.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing average precision scores and latency values.
    """
    # since we will use indexes, we reset them in order to avoid problems
    # (e.g., if the indexes are not in order)
    ground_truth = ground_truth.reset_index(drop=True)
    prediction = prediction.reset_index(drop=True)

    fps = cfg['fps']

    # Convert thresholds seconds to chunks.
    tOffset_thresholds = np.array(tOffset_thresholds) * fps

    ap = np.zeros(len(tOffset_thresholds))
    if prediction.empty:
        return ap, []

    num_pos = float(len(ground_truth))

    # initialize the lock array, -1 means not assigned.
    lock_gt = np.ones((len(tOffset_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction_sorted = prediction.loc[sort_idx].reset_index(drop=True)

    size = (len(tOffset_thresholds), len(prediction))
    tp = np.zeros(size)
    fp = np.zeros(size)
    latency = []

    ground_truth_gbvn = ground_truth.groupby('video-id')

    for idx, this_pred in prediction_sorted.iterrows():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
            # In case there is no ground truth for this video-id
            # everything is considered as false positive
        except KeyError:
            fp[:, idx] = 1
            continue

        # in order to have index as a column
        this_gt = ground_truth_videoid.reset_index()

        toff_arr = _temporal_offset(this_pred['t-start'],
                                    this_gt['t-start'].values)

        # first prediction with the smallest offset
        tOffset_sorted_idx = toff_arr.argsort()

        for tidx, toff_thr in enumerate(tOffset_thresholds):
            for jdx in tOffset_sorted_idx:
                # false positive if the offset is greater than the threshold
                if (toff_arr[jdx] > toff_thr or
                        (latency_threshold and
                         (this_pred['t-pred'] - this_gt['t-start'][jdx]) > latency_threshold_chunk)):
                    fp[tidx, idx] = 1
                    break

                # it has been already assigned a true positive to the jdx-th ground truth
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue

                # Assign as true positive otherwise
                tp[tidx, idx] = 1
                latency.append(max(0, this_pred['t-pred'] - this_gt['t-start'][jdx]))

                # lock this ground truth
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)

    recall_cumsum = tp_cumsum / num_pos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tOffset_thresholds)):
        ap[tidx] = _interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap, latency
