import torch

class NGNN(object):
    """
    d
    """
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device