from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import torch
import torchaudio.transforms as T
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import subprocess
import pandas as pd
import librosa
import numpy as np
import hashlib
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import AUROC
from torchmetrics.classification import MultilabelConfusionMatrix
from torchmetrics import ClasswiseWrapper
import logging
import random
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
import pickle
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from multiprocessing import Pool
from torchvision.ops import sigmoid_focal_loss
import tempfile
from etils import epath
from typing import Any, List, Optional, Tuple, Union
import dataclasses
import pathlib
import soundfile as sf
import csv
import glob 
import argparse
import time
import shutil


# Set the device
train_on_gpu=torch.cuda.is_available()
device = torch.device("cuda:1" if train_on_gpu else "cpu")

# Global vars
SEED = 78

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
SAMPLE_RATE_AST = 32000
SAMPLE_RATE = 48000
SIGNAL_LENGTH = 3 # seconds
SPEC_SHAPE = (298, 128) # width x height
FMIN = 20
FMAX = 15000
MAX_AUDIO_FILES = 10000



