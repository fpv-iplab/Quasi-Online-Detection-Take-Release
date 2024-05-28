from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import pandas as pd

from src.metrics.detection_metrics import compute_average_precision_detection


def _get_class_p_ap_and_latency(gt: pd.DataFrame,
                                prediction: pd.DataFrame,
                                tOffset_thresholds: np.ndarray,
                                latency_threshold: bool = False,
                                latency_threshold_chunk: float = 1.0,
                                name_class: Optional[str] = None,
                                show: bool = False) -> Tuple[np.ndarray, List[float]]:
    """Compute the per-class average precision and latency.

    Args:
        gt (pd.DataFrame): Ground truth instances.
        prediction (pd.DataFrame): Prediction instances.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds.
        latency_threshold (bool, optional): Whether to apply latency threshold. Defaults to False.
        latency_threshold_chunk (float, optional): Latency threshold chunk size. Defaults to 1.0.
        name_class (Optional[str], optional): Name of the class. Defaults to None.
        show (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[float]]: Per-class average precision and latency values.
    """
    ap, mean_latency = compute_average_precision_detection(gt,
                                                           prediction,
                                                           tOffset_thresholds,
                                                           latency_threshold,
                                                           latency_threshold_chunk)

    if name_class is None:
        name_class = "class"

    if show:
        print(f"{name_class} p-Average Precision: {ap}")
        print(f"mean {name_class} p-Average Precision: {np.mean(ap)}")
        print(f"{name_class} mean latency: {mean_latency}")

    return ap, mean_latency


def get_p_ap_and_latency_classes(gt_action_starts: pd.DataFrame,
                                 pred_df: pd.DataFrame,
                                 tOffset_thresholds: np.ndarray,
                                 latency_threshold: bool = False,
                                 latency_threshold_chunk: float = 1.0,
                                 show: bool = False) -> Tuple[List[np.ndarray], List[List[float]]]:
    """Compute the average precision and latency for each class.

    Args:
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        pred_df (pd.DataFrame): Prediction data frame.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds.
        latency_threshold (bool, optional): Whether to apply latency threshold. Defaults to False.
        latency_threshold_chunk (float, optional): Latency threshold chunk size. Defaults to 1.0.
        show (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[List[float]]]: Average precision and latency values for each class.
    """
    gt_action_starts_gpc = gt_action_starts.groupby('action_name')
    ap_classes = []
    latency_classes = []

    for class_name, pred_df_class in pred_df.groupby('action_name'):
        pred_df_class = pred_df_class.reset_index(drop=True)

        gt_action_starts_class = gt_action_starts_gpc.get_group(class_name).reset_index(drop=True)

        ap, mean_latency = _get_class_p_ap_and_latency(gt_action_starts_class,
                                                       pred_df_class,
                                                       tOffset_thresholds,
                                                       latency_threshold=latency_threshold,
                                                       latency_threshold_chunk=latency_threshold_chunk,
                                                       name_class=class_name,
                                                       show=show)

        ap_classes.append(ap)
        if len(mean_latency):
            latency_classes.append(mean_latency)

    return ap_classes, latency_classes


detection_config_type = Dict[str, Union[int, float]]


def get_formatted_performances(tOffset_thresholds: np.ndarray,
                               gt_action_starts: pd.DataFrame,
                               pred_df: pd.DataFrame,
                               detection_config: detection_config_type
                               ) -> Dict[str, Union[float, np.ndarray, detection_config_type]]:
    """Get formatted performance metrics.

    Args:
        tOffset_thresholds (np.ndarray): Temporal offset thresholds.
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        pred_df (pd.DataFrame): Prediction data frame.
        detection_config (detection_config_type): Detection configuration.

    Returns:
        Dict[str, Union[float, np.ndarray, detection_config_type]]: Formatted performance metrics.
    """
    ap_classes, latency_classes = get_p_ap_and_latency_classes(gt_action_starts,
                                                               pred_df,
                                                               tOffset_thresholds,
                                                               latency_threshold=False,
                                                               show=False)

    mean_class_latency = np.mean([np.mean(latency) for latency in latency_classes])
    mean_flat_latency = np.mean([lat for latency in latency_classes for lat in latency])

    p_mAP = np.mean(ap_classes, axis=0)
    mp_mAP = np.mean(p_mAP)

    results = {'p_mAP_tOffset': p_mAP,
               'mp_mAP': mp_mAP,
               'mean_class_latency': mean_class_latency,
               'mean_flat_latency': mean_flat_latency,
               'cfg': detection_config}

    return results
