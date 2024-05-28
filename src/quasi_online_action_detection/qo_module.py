from typing import Optional, Union, List

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sortedcontainers import SortedDict


def _smooth_and_find_peaks(data: np.ndarray,
                           window_size: int,
                           inhibition_time: int,
                           sigma: float,
                           min_dist: Optional[int],
                           latency: bool = True) -> Union[List[int], SortedDict]:
    """Smooth the data using a Gaussian filter and find peaks.

    Args:
        data (np.ndarray): The input data array.
        window_size (int): The size of the window for smoothing.
        inhibition_time (int): The inhibition time to avoid peak detection immediately after another peak.
        sigma (float): The standard deviation for the Gaussian kernel.
        min_dist (Optional[int]): The minimum distance between peaks. If None, it will be set to window_size + inhibition_time.
        latency (bool, optional): Whether to return latency information. Defaults to True.

    Returns:
        Union[List[int], SortedDict]: A list of peak indices if latency is False, otherwise a SortedDict of peaks and their corresponding indices.
    """
    if min_dist is None:
        min_dist = window_size + inhibition_time

    saved_peaks = SortedDict()
    saved_peaks.setdefault(np.iinfo(np.int32).min, 0)

    for i in range(len(data)):
        max_range = min(i, len(data))
        min_range = max(i - window_size, 0)
        window = data[min_range:max_range]
        gau = gaussian_filter1d(window, sigma=sigma)
        peaks = find_peaks(gau)[0] + min_range

        for peak in peaks:
            # we can't have a peak after the inhibition time
            if peak > max_range - inhibition_time:
                continue

            # check if the peak is not near another saved peak
            num_peaks = len(saved_peaks)
            near_peak_idx = saved_peaks.bisect_left(peak)
            if near_peak_idx >= num_peaks:
                if abs(saved_peaks.peekitem(num_peaks - 1)[0] - peak) < min_dist:
                    continue
            elif near_peak_idx == 0:
                if abs(saved_peaks.peekitem(0)[0] - peak) < min_dist:
                    continue
            elif (abs(saved_peaks.peekitem(near_peak_idx - 1)[0] - peak) < min_dist or
                  abs(saved_peaks.peekitem(near_peak_idx)[0] - peak) < min_dist):
                continue

            saved_peaks.setdefault(peak, i)

    saved_peaks.popitem(0)
    if not latency:
        return list(saved_peaks.keys())
    return saved_peaks


def get_class_predictions(pred_scores: np.ndarray,
                          idx_class: int,
                          class_name: str,
                          video_id: str,
                          window_size: int,
                          inhibition_time: int,
                          sigma: float,
                          min_dist: Optional[int] = None,
                          **kwargs) -> Optional[pd.DataFrame]:
    """Generate class predictions from prediction scores.

    Args:
        pred_scores (np.ndarray): The prediction scores array.
        idx_class (int): The index of the class.
        class_name (str): The name of the class.
        video_id (str): The video ID.
        window_size (int): The size of the window for smoothing.
        inhibition_time (int): The inhibition time to avoid peak detection immediately after another peak.
        sigma (float): The standard deviation for the Gaussian kernel.
        min_dist (Optional[int], optional): The minimum distance between peaks. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the class predictions, or None if no peaks are found.
    """
    peaks_pred = _smooth_and_find_peaks(pred_scores[:, idx_class],
                                        window_size=window_size,
                                        inhibition_time=inhibition_time,
                                        sigma=sigma,
                                        min_dist=min_dist,
                                        latency=True)

    if len(peaks_pred) == 0:
        return None

    peaks, peaks_timestamp_pred = zip(*peaks_pred.items())

    class_pred_score = pred_scores[peaks, idx_class]

    class_pred_data = {'action_name': [class_name] * len(peaks),
                       't-start': peaks,
                       't-pred': peaks_timestamp_pred,
                       'score': class_pred_score,
                       'video-id': [video_id] * len(peaks)}

    return pd.DataFrame(class_pred_data)
