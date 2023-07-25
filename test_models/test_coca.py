import os
import glob
import math
import random
import sys
import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


from PIL import Image
import torch
import open_clip
from transformers import AutoProcessor, BlipForConditionalGeneration


processor = AutoProcessor.from_pretrained("lib/blip-pretrained-model/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("lib/blip-pretrained-model/blip-image-captioning-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sys.path.append("lib/sentence-transformers-222/sentence-transformers")
from sentence_transformers import SentenceTransformer, models
st_model = SentenceTransformer("lib/sentence-transformers-222/all-MiniLM-L6-v2")

submissions1 = []

def make_batches(l, batch_size=16):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]

# image_path = "lib/stable-diffusion-image-to-prompts/images"
image_path = "lib/gustavosta_stable_diffusion_prompts_sd2_v2/train_images"
images = os.listdir(image_path)
image_ids = [i.split('.')[0] for i in images]
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(image_ids, 384),
        np.tile(range(384),len(image_ids))
    )
]
# print(len(imgId_eId))
for batch in make_batches(images, 16):
    images_batch = []
    for i, image in enumerate(batch):
        images_batch.append(Image.open(image_path + "/" + image).convert("RGB"))
    pixel_values = processor(images=images_batch, return_tensors="pt").pixel_values.to(device)
    out = model.generate(pixel_values=pixel_values, max_length=20, min_length=5)
    prompts = processor.batch_decode(out, skip_speical_tokens=True)
    embeddings = st_model.encode(prompts).flatten()
    submissions1.extend(embeddings)
    
class CFG:
    device = "cuda"
    seed = 42
    embedding_length = 384
    model_name = "coca_ViT-L-14"
    model_checkpoint_path = "lib/open-clip-models/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k.bin"


model2 = open_clip.create_model(CFG.model_name)
open_clip.load_checkpoint(model2, CFG.model_checkpoint_path)

transform = open_clip.image_transform(
    model2.visual.image_size,
    is_train = False,
    mean = getattr(model2.visual, 'image_mean', None),
    std = getattr(model2.visual, 'image_std', None),
)

model2.to(device)
prompts2 = []
for image_name in images:
    img = Image.open(image_path + "/" + image_name).convert("RGB")
    # print(image_name)
    img = transform(img).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model2.generate(img.to(device))
    prompts2.append(
        open_clip.decode(
            generated[0]
        ).split("<end_of_text>")[0].replace("<start_of_text>", "").rstrip(" .,")
    )
submissions2 = st_model.encode(prompts2).flatten()
file = open('data_output/coca_outputs.txt', 'w')
for num in range(len(imgId_eId)):
    file.write(imgId_eId[num] + ":" + str(submissions1[num] + submissions2[num]) + "\n")
file.close()