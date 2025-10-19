import numpy as np


class BaseProgressCoordinate:
    
    def __init__(self):
        pass
    
    def calculate(self, data):
        raise NotImplementedError("Subclasses must implement calculate method")
    
    def _validate_data_shape(self, data, expected_ndim=3):
        if len(data.shape) != expected_ndim:
            raise ValueError(f"Data must have {expected_ndim} dimensions, got {len(data.shape)}")
