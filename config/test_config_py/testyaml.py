
import yaml
import random
import numpy as np
from easydict import EasyDict as edict

with open('default_record.yaml') as f:
    exp_config = edict(yaml.load(f))

print(exp_config)
