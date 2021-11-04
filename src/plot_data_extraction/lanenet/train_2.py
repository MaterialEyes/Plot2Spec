# train for simple cases (solid, gray, less overlap)
# update
# seg cls (plot) -> (axis, plot)
# add text 

from PIL import Image, ImageFont, ImageDraw
import torch
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from src.models.model import LaneNet
from src.models.loss import DiscriminativeLoss
import cv2

import string
import matplotlib.colors as mcolors

class PlotDigitizerDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, debug=False):
        super(PlotDigitizerDataset, self).__init__()
        self.root = root
        self.debug = debug
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = sorted(glob.glob(os.path.join(root, "*.png")))
        self.masks = [img.split("/")[-1].split(".")[0]+".npy" for img in self.imgs]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = os.path.join(self.root, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask, axis = np.load(mask_path, allow_pickle=True)
        
        w,h = img.size
        
        # number of instances in the image
        num_instance = mask.shape[0]
        
        # generate binary segmentation image
        seg_img = np.amax(mask,axis=0)
        seg_img[axis>0] = 2

        
        # generate
        instance_img = mask
        
        sample = {}
        sample["img"] = img
        target = {}
        target["num_instance"] = num_instance
        target["seg_img"] = seg_img
        target["instance_img"] = instance_img
        sample["target"] = target
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample

    def __len__(self):
        return len(self.imgs)
    
    
# custom transform
class ToTensor(object):
    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = transforms.ToTensor()(img)
        target["num_instance"] = torch.as_tensor(target["num_instance"], 
                                                 dtype=torch.int64)
        target["seg_img"] = torch.as_tensor(target["seg_img"], 
                                            dtype=torch.int64)
        target["instance_img"] = torch.as_tensor(target["instance_img"], 
                                                 dtype=torch.int64)
        return {"img": image, "target": target}
    
class GaussianBlur(object):
    def __init__(self, kernel_size = 7, sigma=(0.1,4),p=0.4):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        image = img
        if np.random.rand() < self.p:
            image = transforms.GaussianBlur(self.kernel_size,sigma=self.sigma)(img)
        return {"img": image, "target": target}

class TightCrop(object):
    def __init__(self, bbox=[71,50,582,435]):
        self.bbox = bbox
        
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        x1,y1,x2,y2 = self.bbox
        image = img.crop([x1,y1,x2,y2])
        seg_img = target["seg_img"][y1:y2,x1:x2]
        instance_img = target["instance_img"][:,y1:y2,x1:x2]
        target["seg_img"] = seg_img
        target["instance_img"] = instance_img
        return {"img": image, "target": target}
        
    
class RandomRescale(object):
    def __init__(self, widths = [32*t for t in range(4,12)], aspect_ratios = [1/4, 1/3, 1/2, 1/1, 2, 3, 4]):
        self.widths = widths
        self.aspect_ratios = aspect_ratios
        
    def __call__(self,sample):
        img, target = sample['img'], sample['target']
        width = int(np.random.choice(self.widths))
        aspect_ratio = np.random.choice(self.aspect_ratios)
        height = width*aspect_ratio
        height = min(max(int(height//32*32),128),384)
        image = img.resize((width, height), resample = Image.NEAREST)
        
        seg_img = target["seg_img"]
        instance_img = target["instance_img"]
        target["seg_img"] = cv2.resize(seg_img, (width, height), interpolation = cv2.INTER_NEAREST)
        new_instance_img = []
        for i in range(instance_img.shape[0]):
            new_instance_img.append(cv2.resize(instance_img[i], (width, height), interpolation = cv2.INTER_NEAREST))
        target["instance_img"] = np.array(new_instance_img)
        return {"img": image, "target": target}
    
class AddText(object):
    def __init__(self, max_words = 10, max_chars = 10, p=0.5):
        self.char_list = list(string.printable[:94])
        self.font_type = glob.glob("/usr/share/fonts/truetype/freefont/*.ttf")
        self.font_size = [10,30]
        self.color_list = list(mcolors.BASE_COLORS.keys())
        
        self.max_words = max_words
        self.max_chars = max_chars
        
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        img_draw = ImageDraw.Draw(img)
        w,h = img.size
        num_words = np.random.randint(self.max_words)
        for i in range(num_words):
            font = ImageFont.truetype(np.random.choice(self.font_type), 
                                 np.random.randint(self.font_size[0], self.font_size[1]))
            num_chars = np.random.randint(1, self.max_chars)
            word = "".join(np.random.choice(self.char_list, size=num_chars))
            iw, ih = np.random.randint(w), np.random.randint(h)
            color_fill = mcolors.BASE_COLORS[np.random.choice(self.color_list)]
            color_fill = tuple([int(255*c) for c in color_fill])
            img_draw.text(xy=(iw,ih), 
                      text=word, 
                      fill=color_fill, 
                      font=font)
        return {"img": img, "target": target}
    
    
# custom dataloader collater_n
def collater_n(data):
    imgs = [s['img'] for s in data]
    targets = [s['target'] for s in data]
    return imgs, targets


if __name__ == "__main__":
    device=torch.device("cuda:0")
    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                        delta_dist=1.5,
                                        norm=2,
                                        usegpu=True)
    model = LaneNet(cnn_type="unet", embed_dim=4)
    
    custom_transform = transforms.Compose([
#         TightCrop(),
        RandomRescale(),
        AddText(),
        GaussianBlur(),
        ToTensor(),
    ])
    plot_dataset = PlotDigitizerDataset(root = "/home/weixin/Documents/GitProjects/plot_digitizer/data/train_tmp_2/",transforms=custom_transform)
    
    train_data_loader = torch.utils.data.DataLoader(
            plot_dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=12,
            collate_fn=collater_n)
    
    
    model.load_state_dict( torch.load("./checkpoints/simple_seg_2/plot_digitizer_30.ckpt"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    for epoch in range(0,100):
        for idx, (imgs, targets) in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss = 0
            for i in range(len(imgs)):
                img = imgs[i].unsqueeze(0).to(device)
                instance_img = targets[i]["instance_img"].unsqueeze(0).to(device)
                seg_img = targets[i]["seg_img"].reshape(-1).to(device)
                num_instance = targets[i]["num_instance"].unsqueeze(0).to(device)
                pred = model(img)
                pred_seg = pred[0].permute(0, 2, 3, 1).reshape(-1,3)
                loss_seg = 0#torch.nn.CrossEntropyLoss()(pred_seg, seg_img)
                loss_instance = criterion_disc(pred[1], instance_img, num_instance)
                loss += loss_seg+loss_instance
            loss.backward()
            optimizer.step()
            print("epoch={}, iter={}, loss={}".format(epoch, idx, loss.item()))
        torch.save(model.state_dict(), "./checkpoints/simple_ins_2/plot_digitizer_{}.ckpt".format(epoch+1))