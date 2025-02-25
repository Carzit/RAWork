import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import utils

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

