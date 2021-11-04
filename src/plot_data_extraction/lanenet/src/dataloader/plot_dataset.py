import torch
import glob
from PIL import Image
import numpy as np
from skimage.segmentation import relabel_sequential
from torchvision import transforms
import cv2
import os



# dataset and dataloader
class PlotDigitizerDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_type, mode, transforms):
        super(PlotDigitizerDataset, self).__init__()
        self.root = root
        self.data_type = data_type
        self.transforms = transforms
        self.max_num_plots = 20
        # load all image files
        self.imglist = []
        self.imglist += sorted(glob.glob(os.path.join(root, data_type, mode, "*.png")))
        self.imglist += sorted(glob.glob(os.path.join(root, data_type, mode, "*.jpg")))
    
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imglist[idx]
        mask_path = img_path.replace("leftImg8bit", "gtFine")
        img = Image.open(img_path).convert("RGB")
        mask_img = np.array(Image.open(mask_path))
        w,h = img.size
        
        # generate binary segmentation image
        seg_img = np.zeros_like(mask_img)
        seg_img[mask_img>0] = 1
        seg_img = np.stack([1-seg_img, seg_img])
        
        
        # number of instances in the image
        num_instance = min(len(np.unique(mask_img))-1,self.max_num_plots)
        
        # generate instance image
        instance_img = np.zeros((self.max_num_plots, h, w))
        if self.data_type == "test":
            pass
        else:  
            ins_img = np.zeros_like(mask_img)
            ins_img[mask_img>0] = relabel_sequential(mask_img[mask_img>0])[0]
            for i in range(1, num_instance+1):
                instance_img[i-1, ins_img == i] = 1
        
        sample = {}
        sample["img"] = img
        target = {}
        target["num_instance"] = num_instance
        target["seg_img"] = seg_img
        target["instance_img"] = instance_img
        sample["target"] = target
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        target = sample["target"]
        img = sample["img"]
        
        return img, target["seg_img"], target["instance_img"], target["num_instance"], img_path
    
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
    
class Normalize():
    def __init__(self):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
    def __call__(self, sample):
        img = sample['img']
        sample['img'] = transforms.Normalize(self.mean, self.std)(img)
        return sample

class RandomRescale(object):
    def __init__(self, mode="train"):
        self.size = 512
        self.mode = mode
        
    def __call__(self,sample):
        img, target = sample['img'], sample['target']
        w,h = img.size
        if w>h:
            nw = self.size
            nh = int(nw/w*h)
        else:
            nh = self.size
            nw = int(nh/h*w)
        if self.mode == "train":
            dw = np.random.randint(self.size-nw+1)
            dh = np.random.randint(self.size-nh+1)
        else:
            dw, dh = 0, 0
        seg_img = target["seg_img"]
        instance_img = target["instance_img"]
        
        new_seg_img = cv2.resize(seg_img.transpose(1,2,0), 
                                 (nw, nh), 
                                 interpolation = cv2.INTER_NEAREST).transpose(2,0,1)
        new_instance_img = cv2.resize(instance_img.transpose(1,2,0), 
                                     (nw, nh), 
                                     interpolation = cv2.INTER_NEAREST).transpose(2,0,1)
        new_img = img.resize((nw, nh))
        
        img = Image.new(mode=new_img.mode, size=(self.size, self.size))
        img.paste(new_img, (dw, dh))
        
        seg_img = np.zeros((new_seg_img.shape[0], self.size, self.size))
        seg_img[:, dh:dh+nh, dw:dw+nw] = new_seg_img
        instance_img = np.zeros((new_instance_img.shape[0], self.size, self.size))
        instance_img[:,dh:dh+nh, dw:dw+nw] = new_instance_img
        
        target["seg_img"] = seg_img
        target["instance_img"] = instance_img
        return {"img": img, "target": target} 