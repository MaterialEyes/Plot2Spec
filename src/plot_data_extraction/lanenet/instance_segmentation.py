import torch
from torchvision import transforms
from torch.nn.parallel.scatter_gather import gather
from PIL import Image
import numpy as np

from .src.dataloader.plot_dataset import PlotDigitizerDataset, RandomRescale, ToTensor, Normalize
from .src.models.model import LaneNet, PostProcessor, LaneClustering
from .src.utils.parallel import DataParallelModel
from .src.utils.utils import get_lane_area


        

class InstanceSeg():
    def __init__(self, opt):
        self.opt = opt
        self.postprocessor = PostProcessor()
        self.clustering = LaneClustering()
        self.load_model()
        self.load_data()
    
    def run(self, idx):
        model = self.model
        imgs, bin_labels, ins_labels, n_lanes, img_path = self.test_loader[idx]
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        with torch.no_grad():
            if torch.cuda.device_count() <= 1:
                bin_preds, ins_preds = model(imgs)
            else:
                bin_preds, ins_preds = gather(model(imgs), 0, dim=0)
                
            bin_pred = bin_preds[0].data.cpu().numpy()
            bin_img = bin_pred.argmax(0)
            bin_img = self.postprocessor.process(bin_img)
            ins_img = ins_preds[0].data.cpu().numpy()
            lane_embedding_feats, lane_coordinate = get_lane_area(bin_img, 
                                                                  ins_img)
            num_clusters, labels, cluster_centers = self.clustering.cluster(
                lane_embedding_feats, bandwidth=1.5)
            ins_img = np.zeros(bin_img.shape)
            ins_img[lane_coordinate[:,0], lane_coordinate[:,1]] = labels + 1
            img = Image.open(img_path[0]).convert("RGB")
        return bin_img, ins_img, img, img_path[0]
        
    
    def __len__(self):
        return len(self.test_loader)
    
    def load_data(self):
        opt = self.opt
        
        custom_transform = transforms.Compose([RandomRescale("test"), 
                                               ToTensor(), 
                                               Normalize()])
        test_dataset = PlotDigitizerDataset(opt.root,
                                            opt.data_type,
                                            opt.mode,
                                            custom_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  num_workers=opt.num_workers,
                                                  shuffle=False,
                                                  pin_memory=True)
        self.test_loader = list(test_loader)

    
    def load_model(self):
        opt = self.opt
        
        # Load checkpoint
        checkpoint = torch.load(opt.model_file)
        checkpoint_opt = checkpoint['opt']
        
        # Load model location
        model = LaneNet(cnn_type=checkpoint_opt.cnn_type)
        model = DataParallelModel(model)
        
        model.load_state_dict(checkpoint['model'])
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        self.model=model
        