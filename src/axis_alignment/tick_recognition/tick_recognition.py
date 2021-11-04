import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import AttnLabelConverter
from .model import Model
from .dataset import AlignCollate


class PatchDataset(Dataset):
    def __init__(self, patches, opt):
        self.patches = patches
        self.opt = opt
        
    def update(self, patches):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        bbox, patch = self.patches[index]
        if not self.opt.rgb:
            patch = patch.convert("L")
        return (patch, bbox)
        
        


class TickRecognition():
    def __init__(self, opt):
        self.opt = opt
        self.opt.num_gpu = torch.cuda.device_count()
        self.opt.charactor = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        
    def detect(self):
        device = self.device
        opt = self.opt
        model = self.model
        results = []
        results_all = []
        for image_tensors, bbox in self.tick_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            with torch.no_grad():
                preds = model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
                
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for idx in range(len(preds_str)):
                    pred = preds_str[idx]
                    pred_prob = preds_max_prob[idx]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]
                    pred_prob = pred_prob[:pred_EOS]
                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_prob.cumprod(dim=0)[-1]
                    results_all.append([pred, float(confidence_score), bbox[idx]])
                    pred = pred.replace("s","5").replace("o","0")
                    if pred.isdigit():
                        results.append([pred, float(confidence_score), bbox[idx]])
        return results, results_all

        
    
    def load_data(self, img, plot_box, text_bboxes):
        x1,y1,x2,y2 = plot_box
        patches = []
        for bbox in text_bboxes:
            if (bbox[1]+bbox[3])/2 > y2:
                patch = img.crop(bbox)
                patches.append([bbox, patch])
        if hasattr(self, "tick_loader"):
            self.tick_loader.dataset.update(patches)
        else:
            opt = self.opt
            AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            tick_data = PatchDataset(patches, opt)
            tick_loader = torch.utils.data.DataLoader(
                dataset = tick_data, 
                batch_size=100,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_demo, 
                drop_last=False,
                pin_memory=True)
            self.tick_loader = tick_loader
        
    
    
    def load_model(self):
        opt = self.opt
        device = self.device
        
        # init model
        converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, 
              opt.num_fiducial, opt.input_channel, opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, 
              opt.Transformation, opt.FeatureExtraction, 
              opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)
        
        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        
        self.model = model
        self.converter = converter