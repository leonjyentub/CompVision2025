import torch
import os, cv2, random, glob
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models, datasets
from torchvision.ops import nms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import warnings
import configparser
import random
warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')
# Load parameters from config file
root = config.get('DEFAULT', 'root_dir')
IMAGE_ROOT = os.path.join(root, 'dataset1')

class SegData(Dataset):
    def __init__(self, image_dir=IMAGE_ROOT, split='train'):
        self.image_dir = os.path.join(image_dir, f'images_prepped_{split}')
        self.mask_dir = os.path.join(image_dir, f'annotations_prepped_{split}')
        self.items = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f'{self.image_dir}/*.png')]
        #self.items = self.stems(f'dataset1/images_prepped_{split}')
        self.split = split
    def __len__(self):
        return len(self.items)
    def __getitem__(self, ix):
        image = cv2.imread(f'{self.image_dir}/{self.items[ix]}.png', cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224,224))
        mask = cv2.imread(f'{self.mask_dir}/{self.items[ix]}.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        return image, mask
    
    def choose(self): return self[random.randint(0, len(self))-1]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([transform(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks
    
def show_batch(batch):
    ims, masks = batch
    ims = ims.cpu().numpy()
    masks = masks.cpu().numpy()
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        ax[0][i].imshow(ims[i].transpose(1,2,0))
        ax[1][i].imshow(masks[i])
    plt.show()

def show(image, bbs=None, texts=None, ax=None, title=None, sz=10, text_sz=1):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(sz,sz))
    text_sz = text_sz if text_sz else (max(sz) * 3 // 5)
    if title:
        ax.set_title(title)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if bbs is not None:
        for ix, bb in enumerate(bbs):
            x,y,X,Y = bb
            cv2.rectangle(image, (x,y), (X,Y), (0,255,0), text_sz)
            cv2.putText(image, texts[ix], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), text_sz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(img[:,:,::-1]) BGR->RGB剛好是倒過來，可以用-1
    ax.imshow(image)

trn_ds = SegData(image_dir=IMAGE_ROOT, split='train')
val_ds = SegData(image_dir=IMAGE_ROOT, split='test')
show(trn_ds.choose()[0], ax=None, title='train')