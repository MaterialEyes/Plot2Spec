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

import string
import matplotlib.colors as mcolors
import cv2

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


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

    
class AddText(object):
    def __init__(self, max_words = 10, max_chars = 10, p=0.5):
        self.char_list = list(string.printable[:94])
        self.font_type = glob.glob("/usr/share/fonts/truetype/freefont/*.ttf")
        self.font_size = [5,20]
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

class Rescale(object):
    def __init__(self, widths = [256], aspect_ratios = [1.5]):
        self.widths = widths
        self.aspect_ratios = aspect_ratios
        
    def __call__(self,sample):
        img, target = sample['img'], sample['target']
        width = int(np.random.choice(self.widths))
        aspect_ratio = np.random.choice(self.aspect_ratios)
        height = width*aspect_ratio
        height = min(max(int(height//32*32),64),512)
        image = img.resize((width, height), resample = Image.NEAREST)
        
        seg_img = target["seg_img"]
        instance_img = target["instance_img"]
        target["seg_img"] = cv2.resize(seg_img, (width, height), interpolation = cv2.INTER_NEAREST)
        new_instance_img = []
        for i in range(instance_img.shape[0]):
            new_instance_img.append(cv2.resize(instance_img[i], (width, height), interpolation = cv2.INTER_NEAREST))
        target["instance_img"] = np.array(new_instance_img)
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
    
class PlotDigitizerDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, **kwargs):
        lw_list = sorted(os.listdir(root))
        self.lw = {}
        self.lw_list = lw_list
        for t in lw_list:
            self.lw[t] = sorted(glob.glob(os.path.join(root, t, "*.png")))
        self.transform = transform
        
        self.plot_color_type= kwargs.get("plot_color_type", "gray")
        self.gray_level = kwargs.get("gray_level", [0,30,90,120,150])
        self.is_color_consistent = True
        self.is_display = True
        
        print("initializing the pools ...")
        self.GeneratePool()
        
    def __len__(self):
        return len(self.pools)
    
    def __getitem__(self, idx):
        img, mask = self.pools[idx]
        mask = 255-np.array(mask)
        
        w,h = img.size
        
        # number of instances in the image
        num_instance = mask.shape[0]
        
        # generate binary segmentation image
        seg_img = np.zeros((h,w))
        seg_mask = np.sum(mask,axis=0)
        seg_img[seg_mask > 0] = 1
        
        # generate
        mask[mask > 0] = 1
        instance_img = mask
        
        
        sample = {}
        sample["img"] = img
        target = {}
        target["num_instance"] = num_instance
        target["seg_img"] = seg_img
        target["instance_img"] = instance_img
        sample["target"] = target
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
        
    
    def GeneratePool(self, **kwargs):
        lw = kwargs.get("lw", "1")
        num_imgs = kwargs.get("num_imgs", 5000)
        max_num_plots = kwargs.get("max_num_plots", 10)
        max_iou_thre = kwargs.get("max_iou_thre", 0.1)
        
        lw_lib = self.lw[lw]
        self.pools = []
        
        if self.is_display:
            for i in tqdm(range(num_imgs)):
                num_plots = np.random.randint(1,max_num_plots)
                new_plot, mask_list = self._gen_plot(lw_lib, num_plots, max_iou_thre)
                self.pools.append([new_plot, mask_list])
        else:
            for i in range(num_imgs):
                num_plots = np.random.randint(1,max_num_plots)
                new_plot, mask_list = self._gen_plot(lw_lib, num_plots, max_iou_thre)
                self.pools.append([new_plot, mask_list])
        
    def UpdatePool(self, **kwargs):
        print("updating the pools ...")
        self.GeneratePool(**kwargs)
        
    
    
    def _mask2plot(self, mask_list):
        if self.plot_color_type == "gray":
            color_list = self.gray_level
            if self.is_color_consistent:
                color = np.random.choice(color_list)
                canvas = None
                for mask in mask_list:
                    if canvas is None:
                        canvas = np.ones(mask.shape)*255
                    canvas[mask==0] = color
                canvas = Image.fromarray(np.uint8(canvas)).convert("RGB")
                return canvas
            
            
            
    
    def _gen_plot(self, lw_lib, num_plots, max_iou_thre):
        # shuffle the library
        np.random.shuffle(lw_lib)
        # canvas init
        canvas = np.array(Image.open(lw_lib[0]).convert("L"))
        mask_list = [canvas.copy()]
        img_id = 1
        while len(mask_list)<num_plots and img_id < len(lw_lib):
            new_plot = np.array(Image.open(lw_lib[img_id]).convert("L"))
            overlap = np.sum(new_plot[canvas==0]==0)
            if overlap/np.sum(new_plot==0) < max_iou_thre:
                canvas[new_plot==0] = 0
                mask_list.append(new_plot)
            else:
                pass
            img_id += 1
        new_plot = self._mask2plot(mask_list)
        return new_plot, mask_list
    
    
# custom dataloader collater_n
def collater_n(data):
    imgs = [s['img'] for s in data]
    targets = [s['target'] for s in data]
    return imgs, targets

def createDeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

# if __name__ == "__main__":
#     device=torch.device("cuda:0")
#     criterion_disc = DiscriminativeLoss(delta_var=0.5,
#                                         delta_dist=1.5,
#                                         norm=2,
#                                         usegpu=True)
#     model = createDeepLabv3(outputchannels=16)
    
#     custom_transform = transforms.Compose([
#         Rescale([256],[1.5]),
#         AddText(),
#         GaussianBlur(),
#         ToTensor(),
#     ])
#     plot_dataset = PlotDigitizerDataset(root = "/home/weixin/Documents/GitProjects/plot_digitizer/data/plot_library/",
#                                     transform=custom_transform)
#     train_data_loader = torch.utils.data.DataLoader(
#             plot_dataset, 
#             batch_size=4, 
#             shuffle=True, 
#             num_workers=12,
#             collate_fn=collater_n)
    
#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    
#     for epoch in range(0,30):
#         for idx, (imgs, targets) in enumerate(train_data_loader):
#             optimizer.zero_grad()
#             imgs = torch.stack(imgs).to(device)
#             pred = model(imgs)["out"]
#             for i in range(imgs.shape[0]):
#                 instance_img = targets[i]["instance_img"].unsqueeze(0).to(device)
#                 num_instance = targets[i]["num_instance"].unsqueeze(0).to(device)
#                 loss = criterion_disc(pred[i].unsqueeze(0), instance_img, num_instance)
#                 loss.backward(retain_graph=True)
#             optimizer.step()
#             print("epoch={}, iter={}, loss={}".format(epoch, idx, loss.item()))
#         torch.save(model.state_dict(), "./checkpoints/deeplab_ins/plot_digitizer_{}.ckpt".format(epoch+1))
#         width = 32*np.random.randint(4,12)
#         aspect_ratio = np.random.choice([1/2,2/3,1,3/2,2])
#         custom_transform = transforms.Compose([
#             Rescale([width],[aspect_ratio]),
#             AddText(),
#             GaussianBlur(),
#             ToTensor(),
#         ])
#         train_data_loader.dataset.transform = custom_transform
#         train_data_loader.dataset.UpdatePool(**{"lw":str(np.random.randint(1,4))})


if __name__ == "__main__":
    device=torch.device("cuda:1")
    model = createDeepLabv3(outputchannels=2)
    
    custom_transform = transforms.Compose([
        Rescale([256],[1.5]),
        AddText(),
        GaussianBlur(),
        ToTensor(),
    ])
    plot_dataset = PlotDigitizerDataset(root = "/home/weixin/Documents/GitProjects/plot_digitizer/data/plot_library/",
                                    transform=custom_transform)
    train_data_loader = torch.utils.data.DataLoader(
            plot_dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=12,
            collate_fn=collater_n)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    
    for epoch in range(0,30):
        for idx, (imgs, targets) in enumerate(train_data_loader):
            optimizer.zero_grad()
            imgs = torch.stack(imgs).to(device)
            seg_img = torch.stack([t["seg_img"] for t in targets]).to(device)
            pred = model(imgs)["out"]
            seg_img = seg_img.reshape(-1)
            pred = pred.permute(0, 2, 3, 1).reshape(-1,2)
            loss = torch.nn.CrossEntropyLoss()(pred,seg_img)
            loss.backward()
            optimizer.step()
            print("epoch={}, iter={}, loss={}".format(epoch, idx, loss.item()))
        torch.save(model.state_dict(), "./checkpoints/deeplab_seg_20210324/plot_digitizer_{}.ckpt".format(epoch+1))
        width = [64,128,192,256,320,384,448,512]
        aspect_ratio = np.random.choice([1/2,2/3,1,3/2,2])
        custom_transform = transforms.Compose([
            Rescale([width],[aspect_ratio]),
            AddText(),
            GaussianBlur(),
            ToTensor(),
        ])
        train_data_loader.dataset.transform = custom_transform
        train_data_loader.dataset.UpdatePool(**{"lw":str(np.random.randint(1,4))})
    
    
    
    
    