import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

 # load model
class CFG:
    model_path = 'lib/Stable-Diffusion-ViT-Baseline-Train/vit_base_patch16_224.pth'
    model_name = 'vit_base_patch16_224'
    input_size = 224
    batch_size = 64


class DiffusionTestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        return image


def predict(images, model_path, model_name, input_size, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = DiffusionTestDataset(images, transform)    
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    print("data done")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=384
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("model done")

    tta_preds = None
    for _ in range(2):
        preds = []
        for X in tqdm(dataloader, leave=False):
            X = X.to(device)

            with torch.no_grad():
                X_out = model(X)
                preds.append(X_out.cpu().numpy())
        if tta_preds is None:
            tta_preds = np.vstack(preds).flatten()
        else:
            tta_preds += np.vstack(preds).flatten()
    return tta_preds / 2
        
    # preds = []
    # for X in tqdm(dataloader, leave=False):
    #     X = X.to(device)

    #     with torch.no_grad():
    #         X_out = model(X)
    #         preds.append(X_out.cpu().numpy())
    
    # return np.vstack(preds).flatten()


if __name__ == "__main__":
    # image_init
    images = list(Path('lib/gustavosta_stable_diffusion_prompts_sd2_v2/train_images').glob('*'))
    # images = list(Path('lib/stable-diffusion-image-to-prompts/images').glob('*'))
    # image_dir = Path('lib/gustavosta_stable_diffusion_prompts_sd2_v2/train_images') / '*'
    # image_paths = glob.glob(image_dir.as_posix())
    print(len(images))
    print(images[0])
    # predict prompts
    # m_CFG = CFG()
    prompts = predict(images, 'lib/Stable-Diffusion-ViT-Baseline-Train/vit_base_patch16_224.pth','vit_base_patch16_224',224, 64)
    print(len(prompts))
    file = open('data_output/vit_gpt2_tta_outputs.txt', 'w')
    num = 0

    for i in range(len(images)):
        image_name = str(images[i]).split('/')[-1].split('.')[0]
        for j in range(384):
            file.write(image_name + "_" + str(j) + ":" + str(prompts[num]) + "\n")
            num += 1
    file.close()