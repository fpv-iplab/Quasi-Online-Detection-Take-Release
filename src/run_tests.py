import os
from typing import Optional, Dict, Union

import pandas as pd
import time
import pickle
import numpy as np
import pickle as pkl
import itertools
from config.config_loader import cfg
from src.metrics.detection_performance import get_formatted_performances, get_p_ap_and_latency_classes
from src.quasi_online_action_detection.qo_module import get_class_predictions


def single_test_parameters(filename_predictions: str,
                           filename_gt_action_starts: str,
                           save_test_folder: Optional[str] = None,
                           csv_filename: Optional[str] = None) -> None:
    """Test a single configuration (defined in the config file). If save_test_folder is not None, save the result in
    a single csv file and in a pickle file. Otherwise, print the results.

    Args:
        filename_predictions (str): Path to the predictions file.
        filename_gt_action_starts (str): Path to the ground truth action starts file.
        save_test_folder (Optional[str], optional): Path to the folder to save pickle and csv files. Defaults to None.
        csv_filename (Optional[str], optional): Path to the csv file where the results should or are already saved. Defaults to None.
    """
    with open(filename_predictions, 'rb') as f:
        predictions = pickle.load(f)

    with open(filename_gt_action_starts, 'rb') as f:
        gt_action_starts = pickle.load(f)

    parameters = cfg['detection_params']

    window_size, inhibition_time, sigma, min_dist = (
        parameters['window_size'],
        parameters['inhibition_time'],
        parameters['sigma'],
        parameters['min_dist']
    )

    tOffset_thresholds = cfg['tOffset_thresholds']
    fps = cfg['fps']

    assert window_size - inhibition_time > 0, "prediction window size cannot be smaller than the inhibition time"

    _execute_test_block(window_size, inhibition_time, sigma, min_dist, fps, predictions, tOffset_thresholds,
                        gt_action_starts, csv_filename, save_test_folder)


def grid_search_parameters(filename_predictions: str,
                           filename_gt_action_starts: str,
                           save_test_folder: Optional[str] = None,
                           csv_filename: Optional[str] = None) -> None:
    """Test a set of configurations and save each result in a single csv file and in a pickle file.

    Args:
        filename_predictions (str): Path to the predictions file.
        filename_gt_action_starts (str): Path to the ground truth action starts file.
        save_test_folder (Optional[str], optional): Path to the folder to save pickle and csv files. Defaults to None.
        csv_filename (Optional[str], optional): Path to the csv file where the results should or are already saved. Defaults to None.
    """
    with open(filename_predictions, 'rb') as f:
        predictions = pickle.load(f)

    with open(filename_gt_action_starts, 'rb') as f:
        gt_action_starts = pickle.load(f)

    # ENIGMA
    window_size_list = [5, 10, 15, 20]
    inhibition_time_list = [0, 1, 2, 3, 4, 5]
    sigma_list = [1, 2, 3]
    min_dist_list = [10]

    # THUMOS
    # windows_size_list = [4, 8, 12, 16]
    # delta_list = [0, 1, 2, 3, 4]
    # sigma_list = [1, 2, 3]
    # min_dist_list = [8]

    tOffset_thresholds = np.linspace(1.0, 10.0, 10)

    fps = cfg['fps']
    count = 0

    global_time_start = time.time()
    combinations = list(itertools.product(window_size_list, inhibition_time_list, sigma_list, min_dist_list))
    num_combinations = len(combinations)

    for window_size, inhibition_time, sigma, min_dist in combinations:
        # prediction window size cannot be smaller than the inhibition time
        if window_size - inhibition_time <= 0:
            continue

        _execute_test_block(window_size, inhibition_time, sigma, min_dist, fps, predictions, tOffset_thresholds,
                            gt_action_starts, csv_filename, save_test_folder)

        count += 1
        print(f"{count}/{num_combinations}")

    print(f"Experiments took {time.time() - global_time_start} seconds")


def performance_over_latency(filename_test_pickle: str, filename_gt_action_starts: str) -> None:
    """Evaluate performance over varying latency thresholds.

    Args:
        filename_test_pickle (str): Path to the test pickle file.
        filename_gt_action_starts (str): Path to the ground truth action starts file.
    """
    tOffset_thresholds = np.linspace(1.0, 10.0, 10)
    latency_thresholds = np.linspace(1.0, 15.0, 15)

    mp_mAP_list = []

    with open(filename_test_pickle, 'rb') as f:
        test_pickle = pickle.load(f)

    with open(filename_gt_action_starts, 'rb') as f:
        gt_action_starts = pickle.load(f)

    pred_df = test_pickle['pred_df']

    global_time_start = time.time()
    for latency_threshold in latency_thresholds:
        ap_classes, _ = get_p_ap_and_latency_classes(gt_action_starts,
                                                     pred_df,
                                                     tOffset_thresholds,
                                                     latency_threshold=True,
                                                     latency_threshold_chunk=latency_threshold,
                                                     show=False)

        p_mAP = np.mean(ap_classes, axis=0)
        mp_mAP = np.mean(p_mAP)

        mp_mAP_list.append(mp_mAP)
        print(f"With a latency threshold of {latency_threshold} chunks, the mp_mAP is {mp_mAP}")

    print(f"Experiments took {time.time() - global_time_start} seconds")


detection_config_type = Dict[str, Union[int, float]]


def _execute_test(filename_test: str,
                  predictions: Dict[str, dict],
                  tOffset_thresholds: np.ndarray,
                  gt_action_starts: pd.DataFrame,
                  detection_config: detection_config_type,
                  csv_filename: Optional[str] = None,
                  save_test_folder: Optional[str] = None) -> None:
    """Execute the test with the given configuration.

    Args:
        filename_test (str): The test filename.
        predictions (Dict[str, dict]): The predictions data.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds.
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        detection_config (detection_config_type): The detection configuration.
        csv_filename (Optional[str], optional): Path to the csv file where results should be saved. Defaults to None.
        save_test_folder (Optional[str], optional): Path to the folder to save pickle and csv files. Defaults to None.
    """
    csv_exist = os.path.exists(csv_filename) if csv_filename is not None else False
    existing_df = None

    if csv_exist:
        existing_df = pd.read_csv(csv_filename)
        if filename_test in existing_df['filename_test'].values:
            mp_mAP = existing_df[existing_df['filename_test'] == filename_test]['mp_mAP'].values[0]
            mean_class_latency = \
            existing_df[existing_df['filename_test'] == filename_test]['mean_class_latency'].values[0]
            mean_flat_latency = existing_df[existing_df['filename_test'] == filename_test]['mean_flat_latency'].values[
                0]

            print(
                f"Test {filename_test} already executed. mp_mAP: {mp_mAP}, mean_class_latency: {mean_class_latency}, mean_flat_latency: {mean_flat_latency}")
            return

    pred_df = pd.DataFrame()

    class_names = cfg['class_names']
    exclude_classes = cfg['exclude_classes']

    for video_name, pred_scores in predictions['perframe_pred_scores'].items():
        for idx_class in range(len(class_names)):
            class_name = class_names[idx_class]
            if class_name in exclude_classes:
                continue

            class_pred_data = get_class_predictions(pred_scores, idx_class, class_name, video_name, **detection_config)
            pred_df = pd.concat([pred_df, class_pred_data], ignore_index=True)

    if save_test_folder is not None:
        path_save = os.path.join(save_test_folder, filename_test)
        _save_gt_pred(path_save, detection_config, pred_df)

    results = get_formatted_performances(tOffset_thresholds, gt_action_starts, pred_df, detection_config)

    filename_test_data = {'filename_test': filename_test}
    results = dict(filename_test_data, **results)
    df_results = pd.DataFrame([results])

    if save_test_folder is not None:
        if csv_exist and existing_df is not None:
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        df_results.to_csv(csv_filename, index=False)
    else:
        print(results)


def _execute_test_block(window_size: int,
                        inhibition_time: int,
                        sigma: float,
                        min_dist: int,
                        fps: int,
                        predictions: Dict[str, dict],
                        tOffset_thresholds: np.ndarray,
                        gt_action_starts: pd.DataFrame,
                        csv_filename: Optional[str] = None,
                        save_test_folder: Optional[str] = None) -> None:
    """Execute a test block with the specified parameters.

    Args:
        window_size (int): The window size.
        inhibition_time (int): The inhibition time.
        sigma (float): The standard deviation for the Gaussian kernel.
        min_dist (int): The minimum distance between peaks.
        fps (int): Frames per second.
        predictions (Dict[str, dict]): The predictions data.
        tOffset_thresholds (np.ndarray): Temporal offset thresholds.
        gt_action_starts (pd.DataFrame): Ground truth action starts.
        csv_filename (Optional[str], optional): Path to the csv file where results should be saved. Defaults to None.
        save_test_folder (Optional[str], optional): Path to the folder to save pickle and csv files. Defaults to None.
    """
    time_start = time.time()
    detection_config = {'window_size': window_size,
                        'inhibition_time': inhibition_time,
                        'sigma': sigma,
                        'min_dist': min_dist,
                        'fps': fps}

    filename_test = f'test_ws{window_size}_it{inhibition_time}_s{sigma}_md{min_dist}.pkl'

    _execute_test(filename_test, predictions, tOffset_thresholds,
                  gt_action_starts, detection_config, csv_filename=csv_filename, save_test_folder=save_test_folder)

    print(f"{filename_test} done in {time.time() - time_start} seconds")


def _save_gt_pred(path_save: str,
                  detection_config: detection_config_type,
                  pred_df: pd.DataFrame) -> None:
    """Save ground truth and predictions to a pickle file.

    Args:
        path_save (str): The path to save the pickle file.
        detection_config (detection_config_type): The detection configuration.
        pred_df (pd.DataFrame): The predictions data frame.
    """
    pkl.dump({
        'cfg': detection_config,
        'pred_df': pred_df,
    }, open(path_save, 'wb'))
