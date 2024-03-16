import torch
import torch.nn as nn
from copy import deepcopy
from model.model import BendrEncoder
from model.model import Flatten


def load_model(device='cpu'):
    """Loading BendrEncoder model
    Args:
        device (str): The device to be used.
    Returns:
        BendrEncoder (nn.Module): The model
    """

    # Initialize the model
    encoder = BendrEncoder()

    # Load the pretrained model
    encoder.load_state_dict(deepcopy(torch.load("encoder.pt", map_location=device)))
    encoder = encoder.to(device)

    return encoder

def create_binary_model(device = "cpu"):
    """ Binary model """
    return create_classification_model(n_classes=2, device = device)

def create_classification_model(n_classes = 5, device = "cpu"):
    """ Classificiation model"""
    return nn.Sequential(
        load_model(device),
        Flatten(),
        nn.Linear(in_features = 3 * 512 * 4, out_features = 512 * 4, bias=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.ReLU(),
        nn.BatchNorm1d(512 * 4),
        nn.Linear(512 * 4, n_classes, bias=True) # two outfeatures
    ).to(device)

