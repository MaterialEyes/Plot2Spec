import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths

from .lanenet.instance_segmentation import InstanceSeg as LN_InstanceSeg
from .SpatialEmbeddings.src.instance_segmentation import InstanceSeg as SE_InstanceSeg
from .utils import ComputeGrad
from .optical_flow import OpticalFlow

class PlotDigitizer():
    def __init__(self):
        self.result_dict = {
            "visual": {},
            "data": {},
        }
        
        
    def __len__(self):
        return len(self.instance_seg)
    
    
    def find_init_posi(self, threshold = 2.):
        ids = self.result_dict["data"]["start_ids"]
        seg_map = self.result_dict["visual"]["seg_map"]
        img_gray = self.result_dict["visual"]["img_gray"]
        grads = ComputeGrad(img_gray)
        start_posi = []
        for idx in ids:
            p, _ = find_peaks(seg_map[:, idx], 0.)
            max_grad = max(abs(grads[p, idx]))
            if max_grad < threshold:
                start_posi.append(idx)
        self.result_dict["data"]["start_posi"] = start_posi
        
        if len(start_posi) < 20:
            max_grads = []
            for idx in ids:
                p, _ = find_peaks(seg_map[:, idx], 0.)
                max_grad = max(abs(grads[p, idx]))
                max_grads.append(max_grad)
            self.result_dict["data"]["start_posi"] = ids[np.argsort(max_grads)[:20]]
#         self.result_dict["data"]["start_posi"] = self.result_dict["data"]["start_posi"][:20]
        print("num of start position: {}".format(len(self.result_dict["data"]["start_posi"])))
        
    
    
    def linewidth_estimation(self):
        seg_map = self.result_dict["visual"]["seg_map"]
        p_width = []
        num_p = []
        for t in range(seg_map.shape[1]):
            p, _ = find_peaks(seg_map[:, t], 0.)
            num_p.append(len(p))
            results = peak_widths(seg_map[:, t], p, rel_height=1.)
            p_width += list(results[0])
        w = np.median(p_width)
        value, count = np.unique(num_p, return_counts=True)
        try:
            value_cand, count_cand = value[np.argsort(-1*count)[:2]], count[np.argsort(-1*count)[:2]]
            estimated_num_plots = value_cand[0]
            if value_cand[0] < value_cand[1] and count_cand[1]/count_cand[0] > 0.8:
                estimated_num_plots = value_cand[1]
        except:
            estimated_num_plots = value[np.argmax(count)]
        ids = np.where(np.array(num_p) == estimated_num_plots)[0]
        self.result_dict["data"]["start_ids"] = ids
        self.result_dict["data"]["linewidth"] = w
        self.result_dict["data"]["num_plots"] = estimated_num_plots
        print("estimated linewidth: {}".format(w))
        print("estimated num of plots: {}".format(estimated_num_plots))
    
    
    def predict_from_ins_seg(self, idx, denoise=True):
        bin_img, ins_img, img, self.img_name = self.instance_seg.run(idx)
        
        w,h = img.size
        if w > h:
            nw = 512
            nh = int(nw/w*h)
        else:
            nh = 512
            nw = int(nh/h*w)
        img = img.resize((nw,nh))
        seg_map = bin_img[:nh, :nw]
        ins_map = ins_img[:nh, :nw]
        img = np.array(img)
        if denoise:
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
            img = cv2.bilateralFilter(img, 15, 75, 75)
        img_rgb = img/255.
        img_gray = 1.-cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.
        
        targets = ["img_rgb", "img_gray", "seg_map", "ins_map"]
        for target in targets:
            self.result_dict["visual"][target] = eval(target)
            
        self.linewidth_estimation()
        self.optical_flow = OpticalFlow(img_rgb, img_gray, seg_map)
    
    def load_seg(self, seg_type, opt):
        if seg_type == "lanenet":
            self.instance_seg = LN_InstanceSeg(opt)
        elif seg_type == "spatialembedding":
            self.instance_seg = SE_InstanceSeg(opt)
        else:
            raise NotImplementedError