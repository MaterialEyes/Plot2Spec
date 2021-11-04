from PIL import Image
import torch
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from src.models.model import LaneNet
from src.models.loss import DiscriminativeLoss

# custom transform
class ToTensor(object):
    def __call__(self, sample):
        img, target = sample['img'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = transforms.GaussianBlur(7,sigma=(0.1,2))(img)
        image = transforms.ToTensor()(img)
        target["num_instance"] = torch.as_tensor(target["num_instance"], 
                                                 dtype=torch.int64)
        target["seg_img"] = torch.as_tensor(target["seg_img"], 
                                            dtype=torch.int64)
        target["instance_img"] = torch.as_tensor(target["instance_img"], 
                                                 dtype=torch.int64)
        return {"img": image, "target": target}
    
class GaussianBlur(object):
    def __init__(self, kernel_size = 7, sigma=(0.1,2)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        img = transforms.GaussianBlur(self.kernel_size,sigma=self.sigma)(img)
        return {"img": image, "target": target}

class TightCrop(object):
    def __init__(self, bbox=[50,50,600,460]):
        self.bbox = bbox
        
    def __call__(self, sample):
        img, target = sample['img'], sample['target']
        x1,y1,x2,y2 = self.bbox
        img = img.crop([x1,y1,x2,y2])
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
        width = np.random.choice(self.widths)
        aspect_ratio = np.random.choice(self.aspect_ratios)
        height = width*aspect_ratio
        height = height//32*32
        img = img.resize((width, height), resample = Image.NEAREST)
        return {"img": image, "target": target}
        
        
    
    
# custom dataloader collater_n
def collater_n(data):
    imgs = [s['img'] for s in data]
    targets = [s['target'] for s in data]
    return imgs, targets

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
        mask = 255 - np.load(mask_path)
        
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
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample

    def __len__(self):
        return len(self.imgs)
    
    
    
if __name__ == "__main__":
    criterion_disc = DiscriminativeLoss(delta_var=0.5,
                                        delta_dist=1.5,
                                        norm=2,
                                        usegpu=True)
    model = LaneNet(cnn_type="unet", embed_dim=16)
    custom_transform = transforms.Compose([ToTensor()])
    plot_dataset = PlotDigitizerDataset(root = "/home/weixin/Documents/GitProjects/plot_digitizer/data/train_s/",transforms=custom_transform)
    train_data_loader = torch.utils.data.DataLoader(
        plot_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=6,
        collate_fn=collater_n)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    for epoch in range(30):
        for idx, (imgs, targets) in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss = 0
            for i in range(len(imgs)):
                img = imgs[i].unsqueeze(0).cuda()
                instance_img = targets[i]["instance_img"].unsqueeze(0).cuda()
                seg_img = targets[i]["seg_img"].reshape(-1).cuda()
                num_instance = targets[i]["num_instance"].unsqueeze(0).cuda()
                pred = model(img)
                pred_seg = pred[0].permute(0, 2, 3, 1).reshape(-1,2)
                loss_seg = torch.nn.CrossEntropyLoss()(pred_seg, seg_img)
                loss_instance = 0#criterion_disc(pred[1], instance_img, num_instance)
                loss += loss_seg+loss_instance
            loss.backward()
            optimizer.step()
            print("epoch={}, iter={}, loss={}".format(epoch, idx, loss.item()))
        torch.save(model.state_dict(), "./unet_checkpoints_instance_only/plot_digitizer_{}.ckpt".format(epoch+1))
    