import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import OFATokenizer, OFAModel

from transformers.models.ofa.generate import sequence_generator
sys.path.append('lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

class ImageGen(Dataset):
    def __init__(self, root, batch_size=32):
        self.root = root
        self.im_paths = os.listdir(self.root)
        self.batch_size = batch_size
        self.sz = len(self.im_paths)
        self.genlen = self.sz//self.batch_size + int(self.sz%self.batch_size > 0)
        
    def __getitem__(self, index):
        if index >= self.genlen:
            raise IndexError("Out of bounds")
        
        l, r = index*self.batch_size, min(self.sz, (index+1)*self.batch_size)
        
        f_paths = [os.path.join(self.root, self.im_paths[i]) for i in range(l,r)]
        f_ids = [self.im_paths[i][:-4] for i in range(l,r)]
        
        ims = [Image.open(f_path) for f_path in f_paths]
        ims = [patch_resize_transform(im).cuda().unsqueeze(0) for im in ims]
        ims = torch.cat(ims)
        
        return ims, f_ids
    
    def __len__(self):
        return self.genlen
        


if __name__ == "__main__":
    CKPT_DIR = "lib/OFA-large-caption/"
    # IMAGE_DIR = "lib/stable-diffusion-image-to-prompts/images"

    IMAGE_DIR = "lib/gustavosta_stable_diffusion_prompts_sd2_v2/train_images"
    BATCH_SIZE = 24
    
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])

    tokenizer = OFATokenizer.from_pretrained(CKPT_DIR)
    model = OFAModel.from_pretrained(CKPT_DIR, use_cache=False).cuda()
    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids

    sub_ids = []
    sub_embeds = []


    imgen = ImageGen(IMAGE_DIR, BATCH_SIZE)

    for b in imgen:
        for j in range(len(b[1])):
            sub_ids.extend([f"{b[1][j]}_{i}" for i in range(384)])

        # print(b[1])
        
        img_batch = b[0]
        out = model.generate(inputs.repeat(len(img_batch), 1).cuda(), patch_images=img_batch, num_beams=5, no_repeat_ngram_size=2)
        out_captions = tokenizer.batch_decode(out, skip_special_tokens=True)
        out_captions = [cap + ", fine details, masterpiece" for cap in out_captions]
        st_model = SentenceTransformer('lib/sentence-transformers-222/all-MiniLM-L6-v2')
        embeddings = st_model.encode(out_captions).flatten()
        sub_embeds.extend(embeddings)
        
    sub_embeddings = np.array(sub_embeds)

    file = open('data_output/ofa_outputs.txt', 'w')
    for num in range(len(sub_ids)):
        file.write(str(sub_ids[num]) + ":" + str(sub_embeds[num]) + "\n")
    file.close()