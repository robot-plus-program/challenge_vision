from typing import List, Tuple

class AlgorithmConfig(object):
    def __init__(self):
        self.config_file = None
        self.confidence_threshold = None
        self.vocabulary = None
        self.custom_vocabulary = None
        self.pred_all_class = None
        self.opts = None
        self.input = None
        self.output = None
        self.input_size = None
        self.acceptable_size_min = None
        self.acceptable_size_max = None

    def set(self,
            config_file: str,
            confidence_threshold: float,
            pred_all_class: bool,
            opts: List,
            input_size: Tuple[int, int],
            acceptable_size_min: float = None,
            acceptable_size_max: float = None,
            vocabulary: str = None,
            custom_vocabulary: str = None,
            input: list = None,
            output: str = None):
        self.config_file = config_file
        self.confidence_threshold = confidence_threshold
        self.vocabulary = vocabulary
        self.custom_vocabulary = custom_vocabulary
        self.pred_all_class = pred_all_class
        self.opts = opts
        self.input = input
        self.output = output
        self.input_size = input_size
        self.acceptable_size_min = acceptable_size_min
        self.acceptable_size_max = acceptable_size_max
