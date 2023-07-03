from pathlib import Path
import os

class ReadLoggingMetrics():
    def __init__(self) -> None:
        self.path = Path('logging/REDD/ML_GE_64_0624_1022/191750_0624_1022')
        
    def load_trace(self):
        trace_path = self.path.joinpath('REDD_multilabel_TSNet_TSHA0624_1023_2f1030ea504f877d5a73')
        trace_path.joinpath('logs/fit_0624-1023_csvlog')
        