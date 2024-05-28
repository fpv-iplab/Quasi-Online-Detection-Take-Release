import os
from yacs.config import CfgNode as CN


class ValidatedConfig(CN):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a validated configuration node.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def validate(self) -> None:
        """
        Validates the configuration parameters to ensure they meet the required constraints.

        Raises:
            AssertionError: If any of the configuration parameters do not meet the specified constraints.
        """
        assert isinstance(self.fps, int) and self.fps > 0, "The number of chunks per second must be a positive integer."
        assert isinstance(self.tOffset_thresholds, list), "tOffset_thresholds must be a numpy array."
        assert all(isinstance(i, float) for i in self.tOffset_thresholds), "tOffset_thresholds must contain floats."
        assert isinstance(self.class_names, list) and len(self.class_names) > 0, "class_names must be a non-empty list."
        assert all(isinstance(i, str) for i in self.class_names), "class_names must contain strings."
        assert isinstance(self.exclude_classes, list), "exclude_classes must be a list."
        assert all(isinstance(i, str) for i in self.exclude_classes), "exclude_classes must contain strings."

        assert isinstance(self.detection_params.window_size,
                          int) and self.detection_params.window_size > 0, "window_size must be a positive integer."
        assert isinstance(self.detection_params.inhibition_time,
                          int) and self.detection_params.inhibition_time >= 0, "inhibition_time must be a non-negative integer."
        assert isinstance(self.detection_params.sigma,
                          int) and self.detection_params.sigma > 0, "sigma must be a positive integer."
        assert isinstance(self.detection_params.min_dist,
                          int) and self.detection_params.min_dist > 0, "min_dist must be a positive integer."

    def merge_from_file(self, cfg_filename: str) -> None:
        """
        Merges the configuration from a file and validates it.

        Args:
            cfg_filename (str): Path to the configuration file.
        """
        super().merge_from_file(cfg_filename)
        self.validate()

    def merge_from_other_cfg(self, cfg_other: CN) -> None:
        """
        Merges the configuration from another configuration node and validates it.

        Args:
            cfg_other (CN): Another configuration node.
        """
        super().merge_from_other_cfg(cfg_other)
        self.validate()

    def merge_from_list(self, cfg_list: list) -> None:
        """
        Merges the configuration from a list and validates it.

        Args:
            cfg_list (list): List of configuration settings.
        """
        super().merge_from_list(cfg_list)
        self.validate()


def get_cfg_defaults() -> ValidatedConfig:
    """
    Returns the default configuration settings.

    Returns:
        ValidatedConfig: The default configuration object.
    """
    config = ValidatedConfig()

    # Class names of the dataset (order matters)
    config.class_names = None
    # Class names to exclude from the dataset
    config.exclude_classes = None

    # Represents the number of chunks per second
    config.fps = 5

    # temporal offset thresholds (in seconds)
    config.tOffset_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Parameters for the quasi-online action detection module
    config.detection_params = ValidatedConfig()
    config.detection_params.window_size = None
    config.detection_params.inhibition_time = None
    config.detection_params.sigma = None
    config.detection_params.min_dist = None

    return config


def load_cfg(config_path: str) -> None:
    """
    Loads and validates the configuration from a file.

    Args:
        config_path (str): Path to the configuration file.
    """
    _validate_config_path(config_path)
    cfg.merge_from_file(config_path)
    cfg.freeze()


def _validate_config_path(config_path: str) -> None:
    """
    Validates the given configuration file path.

    Args:
        config_path (str): Path to the configuration file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the path is not a file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file '{config_path}' does not exist.")
    if not os.path.isfile(config_path):
        raise ValueError(f"The path '{config_path}' is not a file.")


# Create a global configuration object
cfg = get_cfg_defaults()
