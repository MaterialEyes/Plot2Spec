from utils.utils import Cluster
import test_config
from datasets import get_dataset
from models import get_model

import torch
import json
import os
import shutil
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
import itertools
from tqdm import tqdm
from sklearn.cluster import MeanShift, estimate_bandwidth

class InstanceSeg():
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        self.cluster = Cluster()
        test_args = test_config.get_args()
        self.load_dataset(test_args)
        self.load_model(test_args)
    
    def __len__(self):
        return len(self.dataset)
    
    def run(self, idx):
        
        self.visual_dict = {}
        sample = self.dataset[idx]
        im = sample['image']
        instances = sample['instance'].squeeze()
        output = self.model(im)
        instance_map, predictions = self.cluster.cluster(output[0], threshold=0.9)
        
        raw_data = Image.open(sample['im_name'][0]).convert("RGB")
        
        w,h = raw_data.size
        if w > h:
            nw = 512
            nh = int(512/w*h)
        else:
            nh = 512
            nw = int(512/h*w)
        source_img = raw_data.resize((nw,nh))
        source_img = np.array(source_img)
        
        instance_pred = instance_map.data.cpu().numpy()[0:nh,0:nw]
        instance_gt = instances.data.cpu().numpy()[0:nh,0:nw]
        seg_map = torch.sigmoid(output[0,3]).data.cpu().numpy()[0:nh,0:nw]
        seg_map[seg_map>0.3] = 1
        seg_map[seg_map<0.6] = 0
        
        self.visual_dict["display_targets"] = ["raw_data","source_img", "instance_pred", "instance_gt", "seg_map"]
        for target in self.visual_dict["display_targets"]:
            self.visual_dict[target] = eval(target)
        
        
        
    def load_dataset(self, test_args):
#         print(test_args['dataset'])
        dataset = get_dataset(
            test_args['dataset']['name'], test_args['dataset']['kwargs'])
        dataset_it = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if test_args['cuda'] else False)
        self.dataset = list(dataset_it)
    
    def load_model(self, test_args): 
        # set device
        if test_args["cuda"]:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")  
        # set model
        model = get_model(test_args['model']['name'], test_args['model']['kwargs'])
        model = torch.nn.DataParallel(model).to(device)
        
        print('Resuming model from {}'.format(test_args['checkpoint_path']))
        state = torch.load(test_args['checkpoint_path'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        self.model = model
        
class PlotDigitizer():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.InsSeg_engine = InstanceSeg()
        self.visual_dict = {}
        self._init_visual_dict(["display", "result", "plot_params"])
        
    
    def insmap2lines(self, ins_map, inpaint=True):
        def line_inpaint(line, width):
            value_axis, t_axis = line
            tt = np.linspace(0, width-1, width)
            vv = np.interp(tt, t_axis, value_axis)
            return [vv, tt]
        
        ins_ids = sorted(np.unique(ins_map))
        width = ins_map.shape[1]
        if len(ins_ids) == 1: 
            return None

        lines = []
        for idx, ins_id in enumerate(ins_ids[1:]):
            value_axis, t_axis = [], []
            for t in range(ins_map.shape[1]):
                p = np.where(ins_map[:,t] == ins_id)[0]
                if len(p) > 0:
                    value_axis.append(np.median(p))
                    t_axis.append(t)
            lines.append([np.array(value_axis), np.array(t_axis)])
        if inpaint:
            lines = [line_inpaint(line, width) for line in lines]
        return lines
    
    
    def linewidth_estimation(self):
        seg_map = self.visual_dict["display"]["seg_map"]
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
        self.visual_dict["plot_params"]["start_ids"] = ids
        self.visual_dict["plot_params"]["linewidth"] = w
        self.visual_dict["plot_params"]["num_plots"] = estimated_num_plots
        if self.verbose:
            print("estimated linewidth: {}".format(w))
            print("estimated num of plots: {}".format(estimated_num_plots))
    
    
    
    
    
    
    def flow_prep(self, normed_dist_of_pixel=255., LK_method=True):
        
        self._init_visual_dict(["result"])
        def comp_grad(im):
            gradients_t = np.gradient(im, axis=1, edge_order=1)
            gradients_x = np.gradient(im, axis=0, edge_order=1)
            assert gradients_x.shape == gradients_t.shape
            u = np.zeros_like(gradients_x)
            u[gradients_x!=0] =-1* gradients_t[gradients_x!=0] / gradients_x[gradients_x!=0]
            return u
            
        self.visual_dict["plot_params"]["normed_dist_of_pixel"] = normed_dist_of_pixel
        img_gray = self.visual_dict["display"]["img_gray"]
        seg_map = self.visual_dict["display"]["seg_map"]
        img_gray_gaussian = gaussian_filter1d(img_gray, 2, axis=0)
        gray_grads = comp_grad(img_gray_gaussian)
        seg_map_gaussian = gaussian_filter1d(seg_map, 2, axis=0)
        seg_grads = comp_grad(seg_map_gaussian)
        self.visual_dict["display"]["gray_grads"] = gray_grads
        self.visual_dict["display"]["seg_grads"] = seg_grads
        if LK_method:
            lw = self.visual_dict["plot_params"]["linewidth"]
            grads = gaussian_filter1d(gray_grads, lw/4, axis = 0)
            self.visual_dict["display"]["LK_gray_grads"] = grads
            grads = gaussian_filter1d(seg_grads, lw/4, axis = 0)
            self.visual_dict["display"]["LK_seg_grads"] = grads
        
    
    def rescale(self, mode="all", **kwargs):
        if self.verbose:
            print("rescaling >>>>>> mode: {}".format(mode))
        if mode =="all":
            factor = kwargs.get("factor", 2)
            self.rescale_factor = factor
            targets = ["img_rgb", "img_gray", "seg_map", "instance_pred"]
            for target in targets:
                if target+"_origin" not in self.visual_dict["display"]:
                    img = self.visual_dict["display"][target].copy()
                    self.visual_dict["display"][target+"_origin"] =  self.visual_dict["display"][target].copy()
                else:
                    img = self.visual_dict["display"][target+"_origin"].copy()
                h,w = img.shape[:2]
                interp_method = cv2.INTER_NEAREST if target in ["seg_map", "instance_pred"] else cv2.INTER_AREA
                img_re = cv2.resize(img, (factor*w, h), interpolation=interp_method)
                self.visual_dict["display"][target] = img_re
                
    
    
    
    def line_evaluation(self, lines, mode=["color", "semantic", "smooth", "line_span"]):
        score = {}
        
        if "color" in mode:
            color_score = []
            if hasattr(self, "rescale_factor"):
                img_rgb = self.visual_dict["display"]["img_rgb_origin"]
            else:
                img_rgb = self.visual_dict["display"]["img_rgb"]
            for idx, [value_axis, t_axis] in enumerate(lines):
                color_1 = img_rgb[value_axis.astype(int), t_axis.astype(int)]
                color_2 = img_rgb[np.round(value_axis).astype(int), t_axis.astype(int)]
                color = 0.5*color_1+0.5*color_2
                color_score.append(np.mean(color.std(0)))
            score["color"] = color_score
        if "semantic" in mode:
            seg_score = []
            seg_map = self.visual_dict["display"]["seg_map"]
            for idx, [value_axis, t_axis] in enumerate(lines):
                seg_1 = seg_map[value_axis.astype(int), t_axis.astype(int)]
                seg_2 = seg_map[np.round(value_axis).astype(int), t_axis.astype(int)]
                seg = 0.5*seg_1+0.5*seg_2
                seg_score.append(sum(seg)/seg_map.shape[1])
            score["seg"] = seg_score
        if "smooth" in mode:
            smooth_score = []
            for idx, [value_axis, t_axis] in enumerate(lines):
                diffs = []
                for j in range(1,len(t_axis)-1):
                    diff = abs(value_axis[j+1]+value_axis[j-1]-2*value_axis[j])
                    diffs.append(diff)
                smooth_score.append(np.std(diffs))
            score["smooth"] = smooth_score
            
        if "line_span" in mode:
            if hasattr(self, "rescale_factor"):
                img_rgb = self.visual_dict["display"]["img_rgb_origin"]
            else:
                img_rgb = self.visual_dict["display"]["img_rgb"]
            span_score = 0
            for line_id in range(1, len(lines)):
                span_score += np.mean(lines[line_id][0]-lines[line_id-1][0])/img_rgb.shape[0]
            score["span"] = span_score
                
            
        return score
            
    def compt_dist(self, v1, v2, mode="pixel"):
        if mode == "pixel":
            normed_factor = self.visual_dict["plot_params"]["normed_dist_of_pixel"]
            dist = (float(v1)-float(v2))/normed_factor
            dist = -1*dist**2
            return np.exp(dist)
        elif mode == "color":
            dist = (v1-v2)**2
            dist = -1*np.mean(dist)
            return np.exp(dist)
        else:
            raise NotImplementedError
    
    
    
    def _flow(self, posi_s, posi_e, mode="optical_flow_only", momentum=True, LK_method=True):
        

        
        def comp_prob_1(pred_p,pred_c,cand_p, t, mode="posi"):
            if mode == "posi":
                prob = [self.compt_dist(*t,mode="pixel") for t in list(itertools.product(pred_p, cand_p))]
                prob = np.array(prob).reshape((len(pred_p), len(cand_p)))
                return prob, cand_p
            elif mode == "posi_color":
                pass
            elif mode == "color":
                pass
                
        
        def dfs(seq, visited, score, prob):
            dfs_threshold = self.visual_dict["result"]["dfs_threshold"]
            overlap_threshold = self.visual_dict["result"]["overlap_threshold"]
            keep_order_threshold = self.visual_dict["result"]["keep_order_threshold"]
            if score < self.best_score:
                return None
            if len(seq) == prob.shape[0]:
                return score, seq
            node_id = len(seq)
            indices = [None]+list(np.argsort(-1*prob[node_id]))
            for c in indices:
                if c is None:
                    result = dfs(seq+[c], visited, score*dfs_threshold, prob)
                else:
                    if visited[c]:
                        result = dfs(seq+[c], visited, score*prob[node_id,c]*overlap_threshold, prob)
                        continue
                    prev_seq = np.array(seq)[np.array(seq)!=None]
                    visited[c]=True
                    if len(prev_seq) > 0 and max(prev_seq) > c:
                        result = dfs(seq+[c], visited, score*prob[node_id,c]*keep_order_threshold, prob)
                    else:
                        result = dfs(seq+[c], visited, score*prob[node_id,c], prob)
                    visited[c] = False
                if result is not None:
                    f_score, f_seq = result
                    if f_score > self.best_score:
                        self.best_score = f_score
                        self.best_seq = f_seq
#                 if c is not None:
#                     visited[c] = False
        
        
        def adaptive_dfs(seq, visited, score, prob, dfs_threshold):
            overlap_threshold = self.visual_dict["result"]["overlap_threshold"]
            keep_order_threshold = self.visual_dict["result"]["keep_order_threshold"]
            if score < self.best_score:
                return None
            if len(seq) == prob.shape[0]:
                return score, seq
            node_id = len(seq)
            indices = [None]+list(np.argsort(-1*prob[node_id]))
            for c in indices:
                if c is None:
                    result = adaptive_dfs(seq+[c], visited, score*dfs_threshold[node_id], prob, dfs_threshold)
                else:
                    if visited[c]:
                        result = adaptive_dfs(seq+[c], visited, score*prob[node_id,c]*overlap_threshold, prob, dfs_threshold)
                        continue
                    prev_seq = np.array(seq)[np.array(seq)!=None]
                    visited[c]=True
                    if len(prev_seq) > 0 and max(prev_seq) > c:
                        result = adaptive_dfs(seq+[c], visited, score*prob[node_id,c]*keep_order_threshold, prob, dfs_threshold)
                    else:
                        result = adaptive_dfs(seq+[c], visited, score*prob[node_id,c], prob, dfs_threshold)
                    visited[c] = False
                if result is not None:
                    f_score, f_seq = result
                    if f_score > self.best_score:
                        self.best_score = f_score
                        self.best_seq = f_seq
        
        
        
        def fusion(t, pred_p, pred_c, mode="posi", dfs_type=None, dfs_threshold=None):
            seg_map = self.visual_dict["display"]["seg_map"]
            cand_p, _ = find_peaks(seg_map[:, t], 0.)
            self.best_score, self.best_seq = 0, [None]*len(pred_p)
            
            if len(cand_p) == 0:
                return [], []
            elif len(cand_p) > 20:
                prob, cand_p = comp_prob_1(pred_p,pred_c,cand_p, t, mode=mode)
                prob_max = np.amax(prob, axis=0)
                ids = np.argsort(-1*prob_max)[:20]
                prob, cand_p = prob[:, idx], cand_p[:, idx]
            else:
                prob, cand_p = comp_prob_1(pred_p,pred_c,cand_p, t, mode=mode)
                
            visited = [False]*prob.shape[1]
            if dfs_type == "adaptive":
                adaptive_dfs(seq=[], visited=visited, score=1., prob=prob, dfs_threshold=dfs_threshold)
            else:
                dfs(seq=[], visited=visited, score=1., prob=prob)
            return cand_p, prob
        
        
        if posi_s == posi_e:
            return
        stride = 1 if posi_e > posi_s else -1
        if LK_method:
            seg_grads = self.visual_dict["display"]["LK_seg_grads"]
            gray_grads = self.visual_dict["display"]["LK_gray_grads"]
        else:
            seg_grads = self.visual_dict["display"]["seg_grads"]
            gray_grads = self.visual_dict["display"]["gray_grads"]
        seg_map = self.visual_dict["display"]["seg_map"]
        img_rgb = self.visual_dict["display"]["img_rgb"]
        
        prev_p, _ = find_peaks(seg_map[:, posi_s], 0.)
        pred_c = img_rgb[prev_p, posi_s]
        prev_grads = [None]*len(prev_p)
        
        
#         for t in range(posi_s, posi_e, stride):
        for t in tqdm(range(posi_s, posi_e, stride), desc= "forward" if stride==1 else "backward"):
            # 
            self.visual_dict["result"]["points"][t] = prev_p.copy()
            if mode == "optical_flow_with_seg_grads":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = seg_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = round(float(p) + stride*_grad)
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                prev_grads = _grads.copy()
                prev_p = np.array(np.clip(pred_p,0,seg_map.shape[0]-1), np.int32)
            elif mode == "optical_flow_with_gray_grads":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = round(float(p) + stride*_grad)
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                prev_grads = _grads.copy()
                prev_p = np.array(np.clip(pred_p,0,seg_map.shape[0]-1), np.int32)
            elif mode == "optical_flow_with_seg_gray_grads":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    if prev_grads[idx] is None:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]) < abs(gray_grads[p, t]) else gray_grads[p, t]
                    else:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]-prev_grads[idx]) < abs(gray_grads[p, t]-prev_grads[idx]) else gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = round(float(p) + stride*_grad)
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                prev_grads = _grads.copy()
                prev_p = np.array(np.clip(pred_p,0,seg_map.shape[0]-1), np.int32)
            elif mode == "optical_flow_with_seg_grads_posi_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = seg_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
            elif mode == "optical_flow_with_gray_grads_posi_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
            elif mode == "optical_flow_with_seg_gray_grads_posi_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    if prev_grads[idx] is None:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]) < abs(gray_grads[p, t]) else gray_grads[p, t]
                    else:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]-prev_grads[idx]) < abs(gray_grads[p, t]-prev_grads[idx]) else gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
            elif mode == "optical_flow_with_seg_grads_color_correction":
              # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = seg_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                    up = int(np.clip(pred_p[idx]-self.visual_dict["result"]["neighbor_size"],0,seg_map.shape[0]-1))
                    down = int(np.clip(pred_p[idx]+self.visual_dict["result"]["neighbor_size"],0,seg_map.shape[0]-1))+1
                    color_diff = np.array([[p, self.compt_dist(img_rgb[p,t+stride], pred_c[idx], mode="color")] for p in range(up, down)])
                    pred_p[idx] = color_diff[np.argmax(color_diff[:,1]),0]
                    
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                real_grads = pred_p - prev_p
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                _grad = real_grads.copy()
                prev_p = np.array(pred_p, np.int32)  
                
                
            elif mode == "optical_flow_with_seg_grads_posi_color_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = seg_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                
                
                for idx, p in enumerate(next_p):
                    up = int(np.clip(p-self.visual_dict["result"]["neighbor_size"],0,seg_map.shape[0]-1))
                    down = int(np.clip(p+self.visual_dict["result"]["neighbor_size"],0,seg_map.shape[0]-1))+1
                    color_diff = np.array([[p, self.compt_dist(img_rgb[p,t+stride], pred_c[idx], mode="color")] for p in range(up, down)])
                    next_p[idx] = color_diff[np.argmax(color_diff[:,1]),0]
                
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
            elif mode == "optical_flow_with_seg_gray_grads_posi_color_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    if prev_grads[idx] is None:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]) < abs(gray_grads[p, t]) else gray_grads[p, t]
                    else:
                        _grad = seg_grads[p, t] if abs(seg_grads[p, t]-prev_grads[idx]) < abs(gray_grads[p, t]-prev_grads[idx]) else gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
                
            elif mode == "optical_flow_with_gray_grads_posi_color_correction":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = gray_grads[p, t]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
                
            elif mode == "optical_flow_with_seg_gray_grads_posi_color_correction_gradient_smooth":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    if stride*(t+stride) in self.visual_dict["result"]["real_selected_grads"] and stride*(t+2*stride) in self.visual_dict["result"]["real_selected_grads"]:
                        delta_g_t_prev = self.visual_dict["result"]["real_selected_grads"][stride*(t+stride)] - self.visual_dict["result"]["real_selected_grads"][stride*(t+2*stride)] 
                        est_grad = self.visual_dict["result"]["real_selected_grads"][stride*(t+stride)] + delta_g_t_prev
                    else:
                        est_grad = 0 if prev_grads[idx] is None else prev_grads[idx]
#                     print(est_grad, seg_grads[p, t], gray_grads[p, t])
                    _grad = seg_grads[p, t] if abs(seg_grads[p, t]-est_grad) < abs(gray_grads[p, t]-est_grad) else gray_grads[p, t]
                    
                    _grad = self.visual_dict["result"]["momentum_factor"]*est_grad+(1-self.visual_dict["result"]["momentum_factor"])*_grad
                    
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
                
            elif mode == "optical_flow_with_gray_grads_posi_color_correction_grad_rejection":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = gray_grads[p, t]
                    if abs(_grad) > self.visual_dict["result"]["gradient_rejection"]:
                        _grad = prev_grads[idx]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi")
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
                
            elif mode == "optical_flow_with_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    _grad = gray_grads[p, t]
                    if abs(_grad) > self.visual_dict["result"]["gradient_rejection"]:
                        _grad = prev_grads[idx]
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                normed_factor = self.visual_dict["plot_params"]["normed_dist_of_pixel"]
                dfs_factor = self.visual_dict["result"]["dfs_factor"]
                dfs_th = self.visual_dict["result"]["dfs_threshold"]
                
                if stride*(t-stride) in self.visual_dict["result"]["real_selected_grads"]:
                    prev_real_grad = self.visual_dict["result"]["real_selected_grads"][stride*(t-stride)]
                    normed_dist = dfs_factor*prev_real_grad/normed_factor
                    dfs_threshold =np.exp(-1*normed_dist**2)
                    dfs_threshold = np.clip(dfs_threshold,0, self.visual_dict["result"]["max_dfs_threshold"])
                else:
                    dfs_threshold = np.ones(len(pred_p))*dfs_th
                
                if "adaptive_dfs_threshold" not in self.visual_dict["result"]:
                    self.visual_dict["result"]["adaptive_dfs_threshold"] = {t:dfs_threshold}
                else:
                    self.visual_dict["result"]["adaptive_dfs_threshold"][t] = dfs_threshold
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi",dfs_type="adaptive", dfs_threshold=dfs_threshold)
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
                
            elif mode == "optical_flow_with_seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs":
                # prediction with optical flow estimation
                pred_p = np.zeros(prev_p.shape)
                _grads = []
                for idx, p in enumerate(prev_p):
                    if prev_grads[idx] is None:
                        _grad = seg_grads[p, t] if seg_map[p,t] == 1 and abs(seg_grads[p, t]) < abs(gray_grads[p, t]) else gray_grads[p, t]
                    else:
                        _grad = seg_grads[p, t] if seg_map[p,t] == 1 and abs(seg_grads[p, t]-prev_grads[idx]) < abs(gray_grads[p, t]-prev_grads[idx]) else gray_grads[p, t]
                    
                    if abs(_grad) > self.visual_dict["result"]["gradient_rejection"]:
                        _grad = prev_grads[idx] if prev_grads[idx] is not None else 0
                    if momentum and prev_grads[idx] is not None:
                        momentum_factor = self.visual_dict["result"]["momentum_factor"]
                        _grad = momentum_factor*prev_grads[idx]+(1-momentum_factor)*_grad
                    _grads.append(_grad)
                    pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
                normed_factor = self.visual_dict["plot_params"]["normed_dist_of_pixel"]
                dfs_factor = self.visual_dict["result"]["dfs_factor"]
                dfs_th = self.visual_dict["result"]["dfs_threshold"]
                
                if stride*(t-stride) in self.visual_dict["result"]["real_selected_grads"]:
                    prev_real_grad = self.visual_dict["result"]["real_selected_grads"][stride*(t-stride)]
                    normed_dist = dfs_factor*prev_real_grad/normed_factor
                    dfs_threshold =np.exp(-1*normed_dist**2)
                    dfs_threshold = np.clip(dfs_threshold,0,self.visual_dict["result"]["max_dfs_threshold"])
                else:
                    dfs_threshold = np.ones(len(pred_p))*dfs_th
                
                if "adaptive_dfs_threshold" not in self.visual_dict["result"]:
                    self.visual_dict["result"]["adaptive_dfs_threshold"] = {t:dfs_threshold}
                else:
                    self.visual_dict["result"]["adaptive_dfs_threshold"][t] = dfs_threshold
                cand_p, prob = fusion(t+stride, pred_p, pred_c, mode="posi",dfs_type="adaptive", dfs_threshold=dfs_threshold)
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()
                self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
                
                # updates
                next_p = np.zeros_like(prev_p)
                confidence = np.ones_like(prev_p)
                next_p[np.array(self.best_seq) == None] = pred_p[np.array(self.best_seq) == None]
                confidence[np.array(self.best_seq) == None] = 0
                if len(cand_p) > 0:
                    next_p[np.array(self.best_seq) != None] = cand_p[np.array(self.best_seq)[np.array(self.best_seq) != None].astype(np.int32)]
                    
                next_p = np.clip(next_p, 0, seg_map.shape[0]-1)
                for idx, p in enumerate(next_p):
                    p = int(round(p))
                    for j in range(self.visual_dict["result"]["neighbor_size"]):
                        pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                        pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                        if self.compt_dist(img_rgb[pj,t+stride], pred_c[idx], mode="color") > self.visual_dict["result"]["color_threshold"]:
                            next_p[idx] = pj
                            break
                    
                real_grads = next_p - prev_p
                _grad = real_grads.copy()
                
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
                
                prev_grads = _grads.copy()
                prev_p = np.array(np.round(next_p), np.int32)
                
        self.visual_dict["result"]["points"][t+stride] = prev_p.copy()

    
    
    def _flow_all(self, posi, mode="optical_flow_only", momentum=True, LK_method=True):
        seg_map = self.visual_dict["display"]["seg_map"]
        if self.verbose:
            print("anchor position: {}".format(posi))
        self._flow(posi, seg_map.shape[1]-1, mode=mode, momentum=momentum, LK_method=LK_method)
        self._flow(posi, 0, mode=mode, momentum=momentum, LK_method=LK_method)    
    
    
    def flow(self, posi_id=None, mode=None, momentum=True,  LK_method=True):
        if self.verbose:
            dfs_threshold = self.visual_dict["result"]["dfs_threshold"]
            normed_pixel = self.visual_dict["plot_params"]["normed_dist_of_pixel"]
            normed_color = 255.
            print("pixel dist: {}, color dist: {}".format((-1*np.log(dfs_threshold))**0.5*normed_pixel, (-1*np.log(dfs_threshold))**0.5*normed_color))
        
        self.visual_dict["result"]["mode"] = mode 
        
        
        start_posi = self.visual_dict["plot_params"]["start_posi"]
        assert len(start_posi) > 0
        posi = np.random.choice(start_posi) if posi_id is None else start_posi[posi_id]
        self._flow_all(posi, mode=mode, momentum=momentum, LK_method=LK_method)
        
        
    def find_init_posi(self, threshold = 3.):
        def comp_grad(im):
            gradients_t = np.gradient(im, axis=1, edge_order=1)
            gradients_x = np.gradient(im, axis=0, edge_order=1)
            assert gradients_x.shape == gradients_t.shape
            u = np.zeros_like(gradients_x)
            u[gradients_x!=0] =-1* gradients_t[gradients_x!=0] / gradients_x[gradients_x!=0]
            return u
        
        ids = self.visual_dict["plot_params"]["start_ids"]
        seg_map = self.visual_dict["display"]["seg_map"]
        img_gray = self.visual_dict["display"]["img_gray"]
        grads = comp_grad(img_gray)
        start_posi = []
        for idx in ids:
            p, _ = find_peaks(seg_map[:, idx], 0.)
            max_grad = max(grads[p, idx])
            if max_grad < threshold:
                start_posi.append(idx)
        self.visual_dict["plot_params"]["start_posi"] = start_posi
        if self.verbose:
            print("num of start position: {}".format(len(start_posi)))
    
    
    def points2lines(self, points):
        value_axis, t_axis = [], []
        value_axis = np.array([points[t] for t in sorted(points.keys())])
        lines = [[value_axis[:, idx], np.array(sorted(points.keys()))] for idx in range(value_axis.shape[1])]
        return lines
    
    def detection_rescale(self, lines):
        if hasattr(self, "rescale_factor") and self.rescale_factor!=1:
            new_lines = []
            for idx, [value_axis, t_axis] in enumerate(lines):
                current_length = len(t_axis)
                target_length = current_length//self.rescale_factor
                new_t_axis = np.linspace(0, target_length-1,target_length)
                t_axis = np.linspace(0, target_length-1,current_length)
                new_value_axis = np.interp(new_t_axis, t_axis, value_axis)
                new_lines.append([new_value_axis, new_t_axis])
            return new_lines
        return lines
    
    def refinement(self):
        if hasattr(self, "rescale_factor"):
            img_rgb = self.visual_dict["display"]["img_rgb_origin"]
        else:
            img_rgb = self.visual_dict["display"]["img_rgb"]
        
        plot_points = self.visual_dict["result"]["points"]
        plot_lines = self.points2lines(plot_points)
        plot_lines = self.detection_rescale(plot_lines)
        # smooth
        smooth_lines = []
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            smooth_lines.append([gaussian_filter1d(value_axis, 2), t_axis])
        # color
        color_lines = []
        for line_id in range(len(smooth_lines)):
            value_axis, t_axis = smooth_lines[line_id]
            color_collection = img_rgb[np.round(value_axis).astype(np.int32), t_axis.astype(np.int32)]
            color_avg = np.mean(color_collection, axis=0)
            for idx, p in enumerate(value_axis):
                p = int(round(p))
                for j in range(self.visual_dict["result"]["neighbor_size"]):
                    pj = np.clip(p+j, 0, img_rgb.shape[0]-1)
                    if self.compt_dist(img_rgb[pj,int(t_axis[idx])], color_avg, mode="color") > self.visual_dict["result"]["color_threshold"]:
                        value_axis[idx] = pj
                        break
                    pj = np.clip(p-j, 0, img_rgb.shape[0]-1)
                    if self.compt_dist(img_rgb[pj,int(t_axis[idx])], color_avg, mode="color") > self.visual_dict["result"]["color_threshold"]:
                        value_axis[idx] = pj
                        break
            color_lines.append([value_axis, t_axis])
        self.visual_dict["result"]["refine_lines"] = color_lines
            
        
    def save_results(self, save_path = "./"):
        plot_points = self.visual_dict["result"]["points"]
        plot_lines = self.points2lines(plot_points)
        plot_lines = self.detection_rescale(plot_lines)
        
        conf_points = self.visual_dict["result"]["confidence"]
        conf_lines = self.points2lines(conf_points)
        conf_lines = self.detection_rescale(conf_lines)
        
        self.refinement()
        refine_lines = self.visual_dict["result"]["refine_lines"]
        
        colors = [np.random.rand(3) for i in range(20)]
        
        if hasattr(self, "rescale_factor"):
            img_rgb = self.visual_dict["display"]["img_rgb_origin"]
        else:
            img_rgb = self.visual_dict["display"]["img_rgb"]
            
        canvas = np.ones_like(img_rgb)
        pred_instance = self.visual_dict["display"]["instance_pred"]
        seg_map = self.visual_dict["display"]["seg_map"]
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,3,1)
        plt.imshow(img_rgb)
        plt.title("image rgb", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,2)
        plt.imshow(pred_instance)
        plt.title("instance seg", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,3)
        plt.imshow(seg_map)
        plt.title("semantic seg", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,4)
        plt.imshow(canvas)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        plt.title("pred lines", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,5)
        plt.imshow(canvas)
        for line_id in range(len(conf_lines)):
            value_axis, t_axis = plot_lines[line_id]
            conf_axis, conf_t = conf_lines[line_id]
            for idx, t in enumerate(conf_t):
                plt.plot(t, value_axis[list(t_axis).index(t)], "ro" if conf_axis[idx] == 0 else "go", markersize=3)
        plt.title("conf lines", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,6)
        plt.imshow(canvas)
        for line_id in range(len(refine_lines)):
            value_axis, t_axis = refine_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        plt.title("pred lines(refined)", fontsize=20)
        plt.axis("off")
    
        plt.savefig(os.path.join(save_path, self.img_name))
        plt.close()
    
    
    def show_results(self):
        plot_points = self.visual_dict["result"]["points"]
        plot_lines = self.points2lines(plot_points)
        plot_lines = self.detection_rescale(plot_lines)
        
        conf_points = self.visual_dict["result"]["confidence"]
        conf_lines = self.points2lines(conf_points)
        conf_lines = self.detection_rescale(conf_lines)
        
        self.refinement()
        refine_lines = self.visual_dict["result"]["refine_lines"]
        
        colors = [np.random.rand(3) for i in range(20)]
        if hasattr(self, "rescale_factor"):
            img_rgb = self.visual_dict["display"]["img_rgb_origin"]
        else:
            img_rgb = self.visual_dict["display"]["img_rgb"]
            
        canvas = np.ones_like(img_rgb)
        
        plt.figure(figsize=(20,30))
        plt.subplot(3,2,1)
        plt.imshow(img_rgb)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        plt.subplot(3,2,2)
        plt.imshow(canvas)
        for line_id in range(len(conf_lines)):
            value_axis, t_axis = plot_lines[line_id]
            conf_axis, conf_t = conf_lines[line_id]
            for idx, t in enumerate(conf_t):
                plt.plot(t, value_axis[list(t_axis).index(t)], "ro" if conf_axis[idx] == 0 else "go", markersize=3)
        
        plt.subplot(3,2,3)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, -value_axis)

        plt.subplot(3,2,4)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, gaussian_filter1d(-value_axis, 2))
            
        plt.subplot(3,2,5)
        plt.imshow(img_rgb)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, gaussian_filter1d(value_axis, 2), "o", c=colors[line_id], markersize=3)
            
        plt.subplot(3,2,6)
        plt.imshow(img_rgb)
        for line_id in range(len(refine_lines)):
            value_axis, t_axis = refine_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        
        
    
    def _init_visual_dict(self, targets=["display", "result", "plot_params"]):
        if "display" in targets:
            self.visual_dict["display"] = {}
        if "result" in targets:
            self.visual_dict["result"] = {"momentum_factor": 0.9, 
                                       "dfs_threshold": 0.9,
                                       "neighbor_size": 5,
                                       "color_threshold": 0.8,
                                       "overlap_threshold": 0.9,
                                       "keep_order_threshold": 0.9,
                                       "gradient_rejection": 20,
                                       "dfs_factor": 4,
                                       "max_dfs_threshold": 0.95,
                                       "adaptive_dfs_threshold": {},
                                       "points": {},
                                        "cand_p": {},
                                        "prob": {},
                                        "confidence": {},
                                        "selected_grads": {},
                                        "real_selected_grads": {},
                                        "best_score": {},
                                        "best_seq": {},}
        if "plot_params" in targets:
            self.visual_dict["plot_params"] = {}
            
    
    
    def prediction_from_ins_seg(self, idx, denoise=True):
        if hasattr(self, "rescale_factor"):
            delattr(self, "rescale_factor")
        self._init_visual_dict(["display", "result", "plot_params"])
        self.InsSeg_engine.run(idx)
        self.img_name = self.InsSeg_engine.dataset[idx]["im_name"][0].split("/")[-1]
        
        source_img = self.InsSeg_engine.visual_dict["source_img"]
        if denoise:
            source_img = cv2.fastNlMeansDenoisingColored(source_img,None,10,10,7,21)
            source_img = cv2.bilateralFilter(source_img, 15, 75, 75)
        img_gray = 1.-cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)/255.
        img_rgb = source_img/255.
        
        display_targets = self.InsSeg_engine.visual_dict["display_targets"]
        display_targets += ["img_rgb", "img_gray"]
        
        for target in display_targets:
            try:
                self.visual_dict["display"][target] = self.InsSeg_engine.visual_dict[target]
            except:
                self.visual_dict["display"][target] = eval(target)
        self.linewidth_estimation()
                
                
class PlotAnalysis():
    def __init__(self):
        self.pd_engine = PlotDigitizer(False)
        
    def reference_score(self):
        ins_map = self.pd_engine.visual_dict["display"]["instance_pred"]
        ins_lines = self.pd_engine.insmap2lines(ins_map)
        ins_score = self.pd_engine.line_evaluation(ins_lines, ["color","semantic", "smooth", "line_span"])
        self.r_score = ins_score
#         print("reference scores: ")
#         for mdx, mode in enumerate(["color", "seg", "smooth", "span"]):
#             print("{}: {}".format(mode, np.mean(self.r_score[mode])))
        
    def load_img(self, idx, grad_threshold=3.):
        self.pd_engine.prediction_from_ins_seg(idx)
        self.pd_engine.find_init_posi(grad_threshold)
        self.reference_score()
        
            
    def save_best_results(self, save_path="./", json_path = "/home/weixin/Documents/GitProjects/plot_digitizer_BMVC2021/data/output_axis_alignment"):
        self.grid_search()
        scores = [r[2] for r in self.results]
        max_dfs_threshold, posi_id, score, lines_set = self.results[np.argmin(scores)]
        plot_lines, conf_lines, refine_lines = lines_set
        
        colors = [np.random.rand(3) for i in range(20)]
        
        if hasattr(self, "rescale_factor"):
            img_rgb = self.pd_engine.visual_dict["display"]["img_rgb_origin"]
        else:
            img_rgb = self.pd_engine.visual_dict["display"]["img_rgb"]
            
        canvas = np.ones_like(img_rgb)
        pred_instance = self.pd_engine.visual_dict["display"]["instance_pred"]
        seg_map = self.pd_engine.visual_dict["display"]["seg_map"]
        
        plt.figure(figsize=(20,20))
        plt.subplot(2,3,1)
        plt.imshow(img_rgb)
        plt.title("image rgb", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,2)
        plt.imshow(pred_instance)
        plt.title("instance seg", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,3)
        plt.imshow(seg_map)
        plt.title("semantic seg", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,4)
        plt.imshow(canvas)
        for line_id in range(len(plot_lines)):
            value_axis, t_axis = plot_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        plt.title("pred lines", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,5)
        plt.imshow(canvas)
        for line_id in range(len(conf_lines)):
            value_axis, t_axis = plot_lines[line_id]
            conf_axis, conf_t = conf_lines[line_id]
            for idx, t in enumerate(conf_t):
                plt.plot(t, value_axis[list(t_axis).index(t)], "ro" if conf_axis[idx] == 0 else "go", markersize=3)
        plt.title("conf lines", fontsize=20)
        plt.axis("off")
        
        plt.subplot(2,3,6)
        plt.imshow(canvas)
        for line_id in range(len(refine_lines)):
            value_axis, t_axis = refine_lines[line_id]
            plt.plot(t_axis, value_axis, "o", c=colors[line_id], markersize=3)
        plt.title("pred lines(refined)", fontsize=20)
        plt.axis("off")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, self.pd_engine.img_name))
        plt.close()
        
        # save lines to json
        img_name = self.pd_engine.img_name
        json_name = img_name.replace(".jpg",".json").replace(".png",".json")
        annot_file = os.path.join(json_path, json_name)
        if os.path.exists(annot_file):
            with open(annot_file, "r") as fp:
                annot = json.load(fp)
            annot["plots"] = np.array(refine_lines).tolist()
            annot["plots_conf"] = np.array(conf_lines).tolist()
            with open(annot_file, "w") as fp:
                json.dump(annot, fp, indent=2)
        
        
    
    def grid_search(self):
        start_posi = self.pd_engine.visual_dict["plot_params"]["start_posi"]
        grad_threshold = 0
        posi_ids = start_posi
        while len(posi_ids) > 20:
            grad_threshold += 1
            posi_ids = np.where(abs(np.gradient(start_posi))>grad_threshold)[0]   
        self.results = []
        for idx, max_dfs_threshold in enumerate(np.linspace(0.8,0.95,4)):
            for jdy, posi_id in enumerate(posi_ids):
                self.pd_engine.flow_prep()
                self.pd_engine.visual_dict["result"]["momentum_factor"] = 0.35
                self.pd_engine.visual_dict["result"]["dfs_threshold"] = 0.83
                self.pd_engine.visual_dict["result"]["neighbor_size"] = 5
                self.pd_engine.visual_dict["result"]["color_threshold"] = 0.8
                self.pd_engine.visual_dict["result"]["overlap_threshold"] = 0.
                self.pd_engine.visual_dict["result"]["keep_order_threshold"] = 0.
                self.pd_engine.visual_dict["result"]["gradient_rejection"] = 20
                self.pd_engine.visual_dict["result"]["dfs_factor"] = 4.
                self.pd_engine.visual_dict["result"]["max_dfs_threshold"] = max_dfs_threshold
                self.pd_engine.flow(posi_id=posi_id, 
                               mode= "optical_flow_with_seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs",  
                               momentum=True,
                               LK_method=False)
                points = self.pd_engine.visual_dict["result"]["points"]
                lines = self.pd_engine.points2lines(points)
                lines = self.pd_engine.detection_rescale(lines)
                score = self.pd_engine.line_evaluation(lines, ["color","semantic", "line_span"])
                score = 1/3*np.mean(score["color"])+1/3*(1-np.mean(score["seg"]))+1/3*(1-score["span"])
                conf_points = self.pd_engine.visual_dict["result"]["confidence"]
                conf_lines = self.pd_engine.points2lines(conf_points)
                conf_lines = self.pd_engine.detection_rescale(conf_lines)

                self.pd_engine.refinement()
                refine_lines = self.pd_engine.visual_dict["result"]["refine_lines"]
                
                lines_set = [lines, conf_lines, refine_lines]
                self.results.append([max_dfs_threshold, posi_id, score, lines_set])
                
    
    
    def max_dfs_threshold_analysis(self, posi_id):
        self.max_dfs_threshold_results = []
        for max_dfs_threshold in np.linspace(0.8,0.95,4):
            self.pd_engine.flow_prep()
            self.pd_engine.visual_dict["result"]["momentum_factor"] = 0.35
            self.pd_engine.visual_dict["result"]["dfs_threshold"] = 0.83
            self.pd_engine.visual_dict["result"]["neighbor_size"] = 5
            self.pd_engine.visual_dict["result"]["color_threshold"] = 0.8
            self.pd_engine.visual_dict["result"]["overlap_threshold"] = 0.
            self.pd_engine.visual_dict["result"]["keep_order_threshold"] = 0.
            self.pd_engine.visual_dict["result"]["gradient_rejection"] = 20
            self.pd_engine.visual_dict["result"]["dfs_factor"] = 4.
            self.pd_engine.visual_dict["result"]["max_dfs_threshold"] = max_dfs_threshold
            self.pd_engine.flow(posi_id=posi_id, 
                           mode= "optical_flow_with_seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs",  
                           momentum=True,
                           LK_method=False)
            points = self.pd_engine.visual_dict["result"]["points"]
            lines = self.pd_engine.points2lines(points)
            score = self.pd_engine.line_evaluation(lines, ["color","semantic", "line_span"])
            score = 1/3*np.mean(score["color"])+1/3*(1-np.mean(score["seg"]))+1/3*(1-score["span"])
            self.max_dfs_threshold_results.append([max_dfs_threshold, score, lines])
        
    
    def start_posi_analysis(self, max_dfs_threshold=0.95):
        start_posi = self.pd_engine.visual_dict["plot_params"]["start_posi"]
        posi_ids = np.where(np.gradient(start_posi)>1)[0]
        self.start_posi_results = []
        for posi_id in posi_ids:
            self.pd_engine.flow_prep()
            self.pd_engine.visual_dict["result"]["momentum_factor"] = 0.35
            self.pd_engine.visual_dict["result"]["dfs_threshold"] = 0.83
            self.pd_engine.visual_dict["result"]["neighbor_size"] = 5
            self.pd_engine.visual_dict["result"]["color_threshold"] = 0.8
            self.pd_engine.visual_dict["result"]["overlap_threshold"] = 0.
            self.pd_engine.visual_dict["result"]["keep_order_threshold"] = 0.
            self.pd_engine.visual_dict["result"]["gradient_rejection"] = 20
            self.pd_engine.visual_dict["result"]["dfs_factor"] = 4.
            self.pd_engine.visual_dict["result"]["max_dfs_threshold"] = max_dfs_threshold
            self.pd_engine.flow(posi_id=posi_id, 
                           mode= "optical_flow_with_seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs",  
                           momentum=True,
                           LK_method=False)
            points = self.pd_engine.visual_dict["result"]["points"]
            lines = self.pd_engine.points2lines(points)
            score = self.pd_engine.line_evaluation(lines, ["color","semantic", "line_span"])
            score = 1/3*np.mean(score["color"])+1/3*(1-np.mean(score["seg"]))+1/3*(1-score["span"])
            self.start_posi_results.append([posi_id, score, lines])
        
        
    
#     def grid_search(self, m_factors=[0,0.5,11], dfs_thresholds=[0.7,1,15]):
#         self.pd_engine.visual_dict["result"]["neighbor_size"] = 5
#         self.pd_engine.visual_dict["result"]["color_threshold"] = 0.8
#         self.scores = np.zeros((m_factors[-1], dfs_thresholds[-1], 3))
#         for idx, m_factor in enumerate(np.linspace(*m_factors)):
#             self.pd_engine.visual_dict["result"]["momentum_factor"] = m_factor
#             for jdx, dfs_threshold in enumerate(np.linspace(*dfs_thresholds)):
#                 self.pd_engine.visual_dict["result"]["dfs_threshold"] = dfs_threshold
#                 self.pd_engine.flow(posi_id=None, 
#                                grad_threshold=1., 
#                                mode= "optical_flow_with_seg_gray_grads_posi_color_correction",  
#                                momentum=True,
#                                LK_method=False)
#                 points = self.pd_engine.visual_dict["result"]["points"]
#                 lines = self.pd_engine.points2lines(points)
#                 score = self.pd_engine.line_evaluation(lines, ["color","semantic", "smooth"])
#                 for mdx, mode in enumerate(["color", "seg", "smooth"]):
#                     self.scores[idx, jdx, mdx] = np.mean(score[mode])
                
    
    
#     def momentum_analysis(self):
#         self.momentum_scores = []
#         for m_factor in np.linspace(0,0.9,10):
#             self.pd_engine.visual_dict["result"]["momentum_factor"] = m_factor
#             self.pd_engine.flow(posi_id=0, 
#                            grad_threshold=3., 
#                            mode="optical_flow_with_seg_grads",  
#                            momentum=True,
#                            LK_method=False)
#             points = self.pd_engine.visual_dict["result"]["points"]
#             lines = self.pd_engine.points2lines(points)
#             score = self.pd_engine.line_evaluation(lines, ["color","semantic", "smooth"])
#             self.momentum_scores.append([m_factor, score])
            
#     def dfs_threshold_analysis(self):
#         self.dfs_threshold_scores = []
#         for dfs_threshold in np.linspace(0.8,1,21):
#             self.pd_engine.visual_dict["result"]["dfs_threshold"] = dfs_threshold
#             self.pd_engine.visual_dict["result"]["momentum_factor"] = 0.3
#             self.pd_engine.visual_dict["result"]["neighbor_size"] = 10
#             self.pd_engine.visual_dict["result"]["color_threshold"] = 0.8
#             self.pd_engine.flow(posi_id=0, 
#                            grad_threshold=3., 
#                            mode= "optical_flow_with_seg_gray_grads_posi_color_correction",  
#                            momentum=True,
#                            LK_method=False)
#             points = self.pd_engine.visual_dict["result"]["points"]
#             lines = self.pd_engine.points2lines(points)
#             score = self.pd_engine.line_evaluation(lines, ["color","semantic", "smooth"])
#             self.dfs_threshold_scores.append([dfs_threshold, score])
            