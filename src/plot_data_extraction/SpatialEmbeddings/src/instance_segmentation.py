import torch
from PIL import Image

from .utils.utils import Cluster
from .models import get_model
from .datasets import get_dataset

class InstanceSeg():
    def __init__(self, opt):
        self.opt = opt
        self.cluster = Cluster()
        self.load_model()
        self.load_data()
        
        
    def run(self, idx):
        sample = self.test_loader[idx]
        im = sample['image']
        instances = sample['instance'].squeeze()
        output = self.model(im)
        instance_map, predictions = self.cluster.cluster(output[0], threshold=0.9)
        
        bin_img = torch.sigmoid(output[0,3]).data.cpu().numpy()
        bin_img[bin_img > 0.3] = 1
        bin_img[bin_img < 0.6] = 0
        ins_img = instance_map.data.cpu().numpy()
        img = Image.open(sample['im_name'][0]).convert("RGB")
        return bin_img, ins_img, img, sample['im_name'][0]
        
        
    
    def load_data(self):
        opt = self.opt
        # load dataset
        dataset = get_dataset(opt['dataset']['name'], 
                              opt['dataset']['kwargs'])
        test_loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  num_workers=6, 
                                                  pin_memory=True if opt['cuda'] else False)
        self.test_loader = list(test_loader)
        
        
    def __len__(self):
        return len(self.test_loader)
    
    
    def load_model(self):
        opt = self.opt
        # set device
        if opt["cuda"]:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")  
        # set model
        model = get_model(opt['model']['name'], opt['model']['kwargs'])
        model = torch.nn.DataParallel(model).to(device)
        
        print('Resuming model from {}'.format(opt['checkpoint_path']))
        state = torch.load(opt['checkpoint_path'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        self.model = model