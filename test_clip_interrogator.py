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

sys.path.append('lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

import inspect
import importlib
from blip.models import blip

# replace tokenizer path to prevent downloading
blip_path = inspect.getfile(blip)

fin = open(blip_path, "rt")
data = fin.read()
data = data.replace(
    "BertTokenizer.from_pretrained('bert-base-uncased')", 
    "BertTokenizer.from_pretrained('lib/clip-interrogator-models-x/bert-base-uncased')"
)
fin.close()

fin = open(blip_path, "wt")
fin.write(data)
fin.close()

# reload module
importlib.reload(blip)

clip_interrogator_path = inspect.getfile(clip_interrogator.Interrogator)

fin = open(clip_interrogator_path, "rt")
data = fin.read()
data = data.replace(
    'open_clip.get_tokenizer(clip_model_name)', 
    'open_clip.get_tokenizer(config.clip_model_name.split("/", 2)[0])'
)
fin.close()

fin = open(clip_interrogator_path, "wt")
fin.write(data)
fin.close()

importlib.reload(clip_interrogator)


# Config
class my_config:
    device = "cuda"
    comp_path = Path('lib/stable-diffusion-image-to-prompts/')
    
    model_name = "ViT-H-14/laion2b_s32b_b79k"
    clip_model_name = "ViT-H-14"
    clip_model_path = "lib/clip-interrogator-models-x/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    cache_path = "lib/clip-interrogator-models-x"
    
    blip_model_path = "lib/clip-interrogator-models-x/model_large_caption.pth"
    
    # image_path = comp_path / 'images'
    image_path = 'lib/gustavosta_stable_diffusion_prompts_sd2_v2/train_images'
    embeddings_num = 384
    
    st_model_path = "lib/sentence-transformers-222/all-MiniLM-L6-v2"


def my_interrogate_classic(image: Image, image_features: torch.Tensor, caption: str) -> str:
#     print(image)
#     caption = ci.generate_caption(image)
    
#     image_features = ci.image_to_features(image)

    medium = [ci.mediums.labels[i] for i in cos(image_features, medium_features_array).topk(1).indices][0]
    movement = [ci.movements.labels[i] for i in cos(image_features, movement_features_array).topk(1).indices][0]
    flaves = ", ".join([ci.flavors.labels[i] for i in cos(image_features, flaves_features_array).topk(3).indices])

    if caption.startswith(medium):
        prompt = f"{caption}, {movement}, {flaves}"
    else:
        prompt = f"{caption}, {medium}, {movement}, {flaves}"

    return clip_interrogator._truncate_to_fit(prompt, ci.tokenize)

def my_interrogate_fast(image: Image, image_features: torch.Tensor, caption: str):
#     caption = ci.generate_caption(image)
#     image_features = ci.image_to_features(image)
    
    merged_ans = [merged_labels[i] for i in cos(image_features, merged_array).topk(6).indices]
    return str(clip_interrogator._truncate_to_fit(caption + ", " + ", ".join(merged_ans), ci.tokenize)), merged_ans

def my_interrogate(image: Image) -> str:
    caption = ci.generate_caption(image)
    image_features = ci.image_to_features(image)
    
    fast_prompt, flaves = my_interrogate_fast(image, image_features, caption)
#     flaves = [merged_labels[i] for i in cos(image_features, merged_array).topk(16).indices]
    
    best_prompt, best_sim = caption, ci.similarity(image_features, caption)
    best_prompt = ci.chain(image_features, flaves, best_prompt, best_sim, min_count=2, max_count=4, desc="Flavor chain")
    
    classic_prompt = my_interrogate_classic(image, image_features, caption)
#     candidates = [caption, classic_prompt, fast_prompt, best_prompt]
    candidates = [caption, classic_prompt, best_prompt]
    return candidates[np.argmax(ci.similarities(image_features, candidates))]



def clip_interrogator_init():
    # load part
    # load models
    model_config = clip_interrogator.Config(clip_model_name=my_config.model_name)
    model_config.cache_path = my_config.cache_path

    # load clip model
    clip_model = open_clip.create_model(my_config.clip_model_name, precision='fp16' if model_config.device == "cuda" else 'fp32')
    open_clip.load_checkpoint(clip_model, my_config.clip_model_path)
    clip_model.to(model_config.device).eval()
    model_config.clip_model = clip_model

    clip_preprocess = open_clip.image_transform(
        clip_model.visual.image_size,
        is_train = False,
        mean = getattr(clip_model.visual, 'image_mean', None),
        std = getattr(clip_model.visual, 'image_std', None)
    )
    model_config.clip_preprocess = clip_preprocess

    # load blip model
    configs_path = os.path.join(os.path.dirname(os.path.dirname(blip_path)), 'configs')
    med_config = os.path.join(configs_path, 'med_config.json')

    blip_model = blip.blip_decoder(
        pretrained = my_config.blip_model_path,
        image_size = model_config.blip_image_eval_size, 
        vit = model_config.blip_model_type, 
        med_config = med_config
    )

    blip_model.eval()
    blip_model = blip_model.to(model_config.device)
    model_config.blip_model = blip_model
    # some init
    ci = clip_interrogator.Interrogator(model_config)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    medium_features_array = torch.stack([torch.from_numpy(t) for t in ci.mediums.embeds]).to(ci.device)
    movement_features_array = torch.stack([torch.from_numpy(t) for t in ci.movements.embeds]).to(ci.device)
    flaves_features_array = torch.stack([torch.from_numpy(t) for t in ci.flavors.embeds]).to(ci.device)

    merged_array, merged_labels = [], []
    for i in range(len(medium_features_array)):
        merged_array.append(medium_features_array[i])
        merged_labels.append(ci.mediums.labels[i])
    for i in range(len(movement_features_array)):
        merged_array.append(movement_features_array[i])
        merged_labels.append(ci.movements.labels[i])
    for i in range(len(flaves_features_array)):
        merged_array.append(flaves_features_array[i])
        merged_labels.append(ci.flavors.labels[i])
    
    merged_array = torch.stack(merged_array)

    return ci, cos, medium_features_array, movement_features_array, flaves_features_array, merged_array, merged_labels


ci, cos, medium_features_array, movement_features_array, flaves_features_array, merged_array, merged_labels = clip_interrogator_init()


if __name__ == "__main__":
    # get the images list
    images = os.listdir(my_config.image_path)
    imgIds = [i.split('.')[0] for i in images]

    imgId_eId = []
    for image in imgIds:
        for num in range(my_config.embeddings_num):
            imgId_eId.append(image + "_" + str(num))
    print(imgId_eId[0])

    prompts = []
 
    image_path = str(my_config.image_path) + "/"
    for image in images:
        tmp_path = image_path + image
        img = Image.open(tmp_path).convert("RGB")
        img_ans = my_interrogate(img)
        # caption = ci.generate_caption(img)
        # image_features = ci.image_to_features(img)
        # img_ans = my_interrogate_classic(img, image_features, caption, ci, cos, medium_features_array, movement_features_array, flaves_features_array)
        prompts.append(img_ans)

    prompts_dist = {}
    idx = 0
    for image in images:
        img = image.split('.')[0]
        prompts_dist[img] = prompts[idx]
        idx += 1
    
    file = open('data_output/clip_interrogator_outputs.txt', 'w')
    for key in prompts_dist:
        file.write(str(key) + ":" + str(prompts_dist[key]) + "\n")
    file.close()
