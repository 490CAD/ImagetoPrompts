import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append('lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models


if __name__ == "__main__":
    st_model = SentenceTransformer("lib/sentence-transformers-222/all-MiniLM-L6-v2")

    # df_submission = pd.read_csv( "lib/stable-diffusion-image-to-prompts/prompts.csv")
    df_submission = pd.read_csv("lib/gustavosta_stable_diffusion_prompts_sd2_v2/train.csv")
    dist, right_dist = {}, {}
    for num in range(len(df_submission['Prompt'])):
        image_name = df_submission['image_path'][num].split('/')[-1].split('.')[0]
        dist[image_name] = df_submission['Prompt']
        right_dist[image_name] = torch.Tensor(st_model.encoder(df_submission['Prompt'][num]).flateen()).unsqueeze(0)
        # dist[df_submission['imgId'][num]] = df_submission['prompt'][num]
        # right_dist[df_submission['imgId'][num]] = torch.Tensor(st_model.encode(df_submission['prompt'][num]).flatten()).unsqueeze(0)

    # prompt_length = len(df_submission['prompt'])
    prompt_length = len(df_submission['Prompt'])
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # clip prompt
    clip_interrogator_score = 0.0
    clip_interrogator_file = open('data_output/clip_interrogator_outputs.txt', 'r')
    clip_interrogator_dist = {}
    while True:
        line = clip_interrogator_file.readline()
        if line:
            image_name, prompts = line.split(':')[0], line.split(':')[-1]
            
            clip_interrogator_embeddings = torch.Tensor(st_model.encode(prompts).flatten()).unsqueeze(0)
            right_embeddings = right_dist[image_name]
            # right_embeddings = torch.Tensor(st_model.encode(dist[image_name]).flatten()).unsqueeze(0)

            clip_interrogator_dist[image_name] = clip_interrogator_embeddings

            clip_interrogator_score += cos(clip_interrogator_embeddings, right_embeddings)

        else:
            break
    # ofa embeddings
    ofa_score = 0.0
    cnt, prompts = 0, []
    ofa_dist = {}
    ofa_file = open('data_output/ofa_outputs.txt', "r")
    while True:
        line = ofa_file.readline()
        cnt += 1
        if line:
            prompts.append(float(line.split(':')[-1]))
            if cnt == 384:
                image_name = line.split(':')[0].split('_')[0]

                right_embeddings = right_dist[image_name]
                # right_embeddings = torch.Tensor(st_model.encode(dist[image_name]).flatten()).unsqueeze(0)
                ofa_embeddings = torch.Tensor(prompts).unsqueeze(0)

                ofa_dist[image_name] = ofa_embeddings
                ofa_score += cos(ofa_embeddings, right_embeddings)

                prompts.clear()
                cnt = 0
        else:
            break

    # cal all
    print(clip_interrogator_score / prompt_length)
    print(ofa_score / prompt_length)
    best_score, best_pro = 0.0, 0.0
    
    for ofa_pro in range(101):
        score = 0.0
        for key in ofa_dist:
            embeddings = ofa_pro / 100.0 * ofa_dist[key] + (1.0 - ofa_pro / 100.0) * clip_interrogator_dist[key]
            score += cos(embeddings, right_dist[key])
        if score > best_score:
            best_score = score
            best_pro = ofa_pro / 100.0
    print(best_pro, best_score / prompt_length)