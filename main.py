import argparse

from config.config_loader import load_cfg
from src.run_tests import single_test_parameters, grid_search_parameters, performance_over_latency


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="config/enigma_config.yaml",
                        help="Path to the configuration file")
    parser.add_argument('--mode', type=str, default='single_test',
                        help='Mode of operation: single_test, grid_search, or performance_over_latency')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    load_cfg(args.config_path)

    # save_test_folder = "data/model_experiments/ENIGMA"
    # csv_filename = "data/model_experiments/ENIGMA/enigma_results.csv"
    # filename_predictions = "data/predictions/enigma_predictions.pkl"
    # filename_gt_action_starts = "data/gt_action_starts/enigma_gt_action_starts.pkl"
    # filename_test_pickle = "data/model_experiments/ENIGMA/test_ws10_it4_s3_md10.pkl"

    save_test_folder = "data/model_experiments/THUMOS"
    csv_filename = "data/model_experiments/THUMOS/thumos_results.csv"
    filename_predictions = "data/predictions/thumos_predictions.pkl"
    filename_gt_action_starts = "data/gt_action_starts/thumos_gt_action_starts.pkl"
    filename_test_pickle = "data/model_experiments/THUMOS/test_ws12_it0_s1_md8.pkl"


    if args.mode == 'single_test':
        single_test_parameters(filename_predictions, filename_gt_action_starts, save_test_folder=None,
                               csv_filename=None)
    elif args.mode == 'grid_search':
        # edit the default values of the parameters to be tested
        grid_search_parameters(filename_test_pickle, filename_gt_action_starts, save_test_folder=None,
                               csv_filename=None)
    elif args.mode == 'performance_over_latency':
        # edit the default values of the parameters to be tested
        performance_over_latency(filename_test_pickle, filename_gt_action_starts)
    else:
        print("Invalid mode specified. Use 'single_test', 'grid_search', or 'performance_over_latency'.")
