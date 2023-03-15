import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append('lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models


st_model = SentenceTransformer("lib/sentence-transformers-222/all-MiniLM-L6-v2")
cos = torch.nn.CosineSimilarity(dim=1)


def type1_read(output_path, right_dist):
    score = 0.0
    file = open(output_path, "r")
    dist = {}
    while True:
        line = file.readline()
        if line:
            image_name, prompts = line.split(':')[0], line.split(':')[-1]
            embeddings = torch.Tensor(st_model.encode(prompts).flatten()).unsqueeze(0)
            right_embeddings = right_dist[image_name]
            dist[image_name] = embeddings
            score += cos(embeddings, right_embeddings)
        else:
            break
    file.close()
    return dist, score

def type2_read(output_path, right_dist):
    score, dist = 0.0, {}
    cnt, prompts = 0, []
    file = open(output_path, "r")
    while True:
        line = file.readline()
        cnt += 1
        if line:
            prompts.append(float(line.split(':')[-1]))
            if cnt == 384:
                image_name = line.split(':')[0].split('_')[0]

                right_embeddings = right_dist[image_name]
                # right_embeddings = torch.Tensor(st_model.encode(dist[image_name]).flatten()).unsqueeze(0)
                embeddings = torch.Tensor(prompts).unsqueeze(0)

                dist[image_name] = embeddings
                score += cos(embeddings, right_embeddings)

                prompts.clear()
                cnt = 0
        else:
            break
    file.close()
    return dist, score


def cal_one_dimension(right_dist, clip_dist, ofa_dist, pos, pro, score):
    best_score, best_pro = score, pro[pos]
    for i_pro in range(101):
        temp_score =0.0
        for key in right_dist:
            embeddings = [0.0 for i in range(384)]
            for i in range(len(pro)):
                if i == pos:
                    embeddings[i] = i_pro / 100.0 * ofa_dist[key][0][i] + (1.0 - i_pro / 100.0) * clip_dist[key][0][i]
                else:
                    embeddings[i] = pro[i] * ofa_dist[key][0][i] + (1.0 - pro[i]) * clip_dist[key][0][i]
            embeddings = torch.Tensor(embeddings).unsqueeze(0)
            temp_score += cos(embeddings, right_dist[key])
        if temp_score > best_score:
            best_score = temp_score
            best_pro = i_pro / 100.0
    return best_score, best_pro


def merge_two(right_dist, un_dist, use_dist, pre_score):
    best_score, best_pro = pre_score, 1.0
    for pro in range(101):
        score = 0.0
        for key in right_dist:
            embeddings = (pro / 100.0) * use_dist[key] + (1.0 - pro / 100.0) * un_dist[key]
            score += cos(embeddings, right_dist[key])
        if score > best_score:
            best_score = score
            best_pro = pro / 100.0
    return best_score, best_pro


if __name__ == "__main__":
    # df_submission = pd.read_csv( "lib/stable-diffusion-image-to-prompts/prompts.csv")
    df_submission = pd.read_csv("lib/gustavosta_stable_diffusion_prompts_sd2_v2/train.csv")
    dist, right_dist = {}, {}

    prompt_length = len(df_submission['Prompt'])
    print(prompt_length)

    for num in range(prompt_length):
        image_name = df_submission['image_path'][num].split('/')[-1].split('.')[0]
        dist[image_name] = df_submission['Prompt']
        right_dist[image_name] = torch.Tensor(st_model.encode(df_submission['Prompt'][num]).flatten()).unsqueeze(0)
        # break
        # dist[df_submission['imgId'][num]] = df_submission['prompt'][num]
        # right_dist[df_submission['imgId'][num]] = torch.Tensor(st_model.encode(df_submission['prompt'][num]).flatten()).unsqueeze(0)
    print("right done")
    # clip prompt
    clip_interrogator_path = 'data_output/clip_interrogator_outputs.txt'
    clip_interrogator_dist, clip_interrogator_score = type1_read(clip_interrogator_path, right_dist)
    print("clip done")
    # vit-gpt2 embeddings
    vit_gpt2_path = "data_output/vit_gpt2_outputs.txt"
    vit_gpt2_dist, vit_gpt2_score = type2_read(vit_gpt2_path, right_dist)
    print("vit done")
    # ofa embeddings
    ofa_path = "data_output/ofa_outputs.txt"
    ofa_dist, ofa_score = type2_read(ofa_path, right_dist)
    print("ofa done")
    # ofa_vit_tta embeddings
    tta_path = "data_output/vit_gpt2_tta_outputs.txt"
    tta_dist, tta_score = type2_read(tta_path, right_dist)
    print("tta done")
    # coca embeddings
    # coca_path = "data_output/coca_outputs.txt"
    # coca_dist, coca_score = type2_read(coca_path, right_dist)
    # print("CoCa done")

    # cal all
    print(clip_interrogator_score / prompt_length)
    print(ofa_score / prompt_length)
    print(vit_gpt2_score / prompt_length)
    print(tta_score / prompt_length)
    # print(coca_score / prompt_length)
    # test 1 + 2
    afo_dist, two_dist = {}, {}
    score, two_score = 0.0, 0.0
    for key in ofa_dist:
        embeddings = 0.24 * ofa_dist[key] + 0.72 * clip_interrogator_dist[key] + 0.04 * vit_gpt2_dist[key]
        em = 0.32 * ofa_dist[key] + 0.68 * clip_interrogator_dist[key]
        two_dist[key] = em
        afo_dist[key] = embeddings
        score += cos(embeddings, right_dist[key])
        two_score += cos(em, right_dist[key])
    print(score / prompt_length)
    # 0.24 0.72 0.04 : 0.6924
    best_score, best_pro = merge_two(right_dist, tta_dist, two_dist, two_score)
    print(best_score / prompt_length, best_pro)
    # best_score, best_pro = merge_two(right_dist, coca_dist, afo_dist, score)
    # test 1 + 1 + 1
    # best_score, best_pro1, best_pro2, best_pro3 = 0.0, 0.0, 0.0, 0.0

    # for pro1 in range(101):
    #     for pro2 in range(101 - pro1):
    #         score = 0.0
    #         for key in ofa_dist:
    #             n_pro1, n_pro2, n_pro3 = pro1 / 100.0, pro2 / 100.0, 1.0 - pro1 / 100.0 - pro2 / 100.0
    #             embeddings = n_pro1 * ofa_dist[key] + n_pro2 * vit_gpt2_dist[key] +  n_pro3 * clip_interrogator_dist[key]
    #             score += cos(embeddings, right_dist[key])
    #         if score > best_score:
    #             best_score = score
    #             best_pro1 = pro1 / 100.0
    #             best_pro2 = pro2 / 100.0
    #             best_pro3 = 1.0 - best_pro1 - best_pro2
    
    # print(best_pro1, best_pro2, best_pro3)
    # print(best_score / prompt_length)   

    # pro = [0.32 for i in range(384)]

    # for pos in range(384):
    #     best_score, best_pro = cal_one_dimension(right_dist, clip_interrogator_dist, ofa_dist, pos, pro, score)
    #     if best_score > score:
    #         score = best_score
    #         pro[pos] = best_pro 
    #         print(pos + ":", best_score, best_pro)
    # print(pro)
    # print(score)

    # 0.5691 ofa:32
    # 0.6914
    # /root/miniconda3/envs/kaggle/bin/python /share/ImagetoPrompts/cal_cv_model.py