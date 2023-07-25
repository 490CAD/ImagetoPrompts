# init
import os
import sys
import torch
import open_clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from clip_interrogator import clip_interrogator
from typing import List
import pandas as pd
sys.path.append('sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

import inspect
import importlib
from blip.models import blip
#Load CLIP model
clip = models.CLIPModel()
model = SentenceTransformer(modules=[clip])
#Encode text descriptions
text_emb = model.encode(['Two dogs in the snow', 'A cat on a table', 'A picture of London at night'])
#Encode an image:
img_emb = model.encode(Image.open('stable-diffusion-image-to-prompts/images/92e911621.png'))
pd.DataFrame(np.array(np.append([img_emb],text_emb,axis=0))).to_csv("data_output/output.csv")