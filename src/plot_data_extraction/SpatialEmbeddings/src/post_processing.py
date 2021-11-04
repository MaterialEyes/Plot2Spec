# from simulator import DATAsimulator
from utils.utils import Cluster
import test_config
from datasets import get_dataset
from models import get_model

import torch
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


class DataPreparation():
    def __init__(self, verbose=True):
        self.verbose = verbose
        test_args = self.init_process()
        self.dataloader = self.load_dataset(test_args)
        if self.verbose:
            print("num of samples in the test set: {}".format(len(self.dataloader)))
        self.visual_dict = {"display_targets": []}
    
    def visualization(self, save=False):
        display_targets = self.visual_dict["display_targets"]
        
        if len(display_targets) == 0:
            print("No targets for visualization!!!")
        elif len(display_targets) == 4:
            plt.figure(figsize=(20,10))
            for idx, target in enumerate(display_targets, 1):
                plt.subplot(1,4,idx)
                plt.imshow(self.visual_dict[target])
                plt.title(target)
#                 plt.axis("off")
            if save:
                plt.savefig("./visualization.png")
    
    def instance_seg(self, idx):
        assert idx >= 0 and idx < len(self.dataloader), "{}<idx<{}".format(-1, len(self.dataloader))
        
        for i, sample in enumerate(self.dataloader):
            if i == idx:
                break
        im = sample['image']
        instances = sample['instance'].squeeze()
        output = self.model(im)
#         print(type(output), output.shape, output.dtype)
        instance_map, predictions = self.cluster.cluster(output[0], threshold=0.9)
        

        raw_data = Image.open(sample['im_name'][0]).convert("RGB")
#         self.visual_dict["raw_data"] = raw_data
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
        
    
    def init_process(self):
        
        torch.backends.cudnn.benchmark = True
        # set device
        device = torch.device("cuda:0")
        test_args = test_config.get_args()
        # set model
        model = get_model(test_args['model']['name'], test_args['model']['kwargs'])
        model = torch.nn.DataParallel(model).to(device)
        # clustering
        cluster = Cluster()
        if self.verbose:
            print('Resuming model from {}'.format(test_args['checkpoint_path']))
        state = torch.load(test_args['checkpoint_path'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        
        self.model = model
        self.cluster = cluster
        return test_args
    
    def load_dataset(self, test_args):
#         data_simulator = DATAsimulator()
#         kwargs = {
#             "dataset_type": "cityscape",
#             "root": "../data/rp_graph/",
#             "save_path": "../data/tmp/",
#             "num_figures": 300,
#             "root_test": "../../plot_digitizer/data/test_samples_crop/",
#             "is_clean": True,
#         }
#         data_simulator._params_init(**kwargs)
#         data_simulator.GenerateTestData()
        
#         print(test_args['dataset']['name'], test_args['dataset']['kwargs'])
        dataset = get_dataset(
            test_args['dataset']['name'], test_args['dataset']['kwargs'])
        dataset_it = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if test_args['cuda'] else False)
        
        return dataset_it


class PostProcess():
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dp_engine = DataPreparation()
        
        
    def line_evaluation(self, lines):
        img_rgb = self.visual_dict["display"]["img_rgb"]
        seg_map = self.visual_dict["display"]["seg_map"]
        scores = []
        for idx, [value_axis, t_axis] in enumerate(lines):
            # semantic score
            sem = seg_map[np.round(value_axis).astype(int), t_axis]/2 + seg_map[value_axis.astype(int), t_axis]/2
            sem = sum(sem)/img_rgb.shape[1]
            # color score
            colors = img_rgb[np.round(value_axis).astype(int), t_axis]/2 + img_rgb[value_axis.astype(int), t_axis]/2
            # smooth score
            grad_diff = [0]*len(t_axis)
            for t_idx in range(1, len(t_axis)-1):
                left_grad = (value_axis[t_idx]-value_axis[t_idx-1])/(t_axis[t_idx]-t_axis[t_idx-1])
                right_grad = (value_axis[t_idx+1]-value_axis[t_idx])/(t_axis[t_idx+1]-t_axis[t_idx])
                grad_diff[t_idx] = (left_grad-right_grad)**2
            scores.append([sem, colors.std(0), grad_diff])
        return scores
        
    def points2lines(self, points):
        value_axis, t_axis = [], []
        value_axis = np.array([points[t] for t in sorted(points.keys())])
        lines = [[value_axis[:, idx], np.array(sorted(points.keys()))] for idx in range(value_axis.shape[1])]
        return lines
        
    
    def instancemap2lines(self, ins_map):
#         ins_map = self.visual_dict["display"]["instance_pred"]
        ins_ids = sorted(np.unique(ins_map))
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

        return lines
        
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
            
    def rescale(self, factor=2):
        targets = ["seg_map", "img_gray", "img_rgb"]
        for target in targets:
            if "{}_0".format(target) in self.visual_dict["display"]:
                img = self.visual_dict["display"]["{}_0".format(target)]
            else:
                img = self.visual_dict["display"][target]
                self.visual_dict["display"]["{}_0".format(target)] = img
            img = Image.fromarray(np.uint8(img*255))
            w,h = img.size
            if target == "seg_map":
                rescale_img = img.resize((w*factor, h), resample = Image.NEAREST)
                rescale_img = np.array(rescale_img)/255.
                self.visual_dict["display"][target] = rescale_img
            else:
                rescale_img = img.resize((w*factor, h), resample = Image.BILINEAR)
                rescale_img = np.array(rescale_img)/255.
                self.visual_dict["display"][target] = rescale_img
    
    
    
    def _flow(self, posi_s, posi_e, dfs_threshold=0.99, mode="optical_flow_only", LK_method=True):
        
        def _comp_prob(pred_p,pred_c,cand_p,cand_c):
            prob = [self.compt_dist(*t,mode="pixel") for t in list(itertools.product(pred_p, cand_p))]
            prob_d = np.array(prob).reshape((len(pred_p), len(cand_p)))
            prob = [self.compt_dist(*t,mode="color") for t in list(itertools.product(pred_c, cand_c))]
            prob_c = np.array(prob).reshape((len(pred_p), len(cand_p)))
            prob = 0.5*prob_d+0.5*prob_c
            return prob
        
        def comp_prob(pred_p,pred_c,cand_p, t):
            img_rgb = self.visual_dict["display"]["img_rgb"]
            lw = self.visual_dict["plot_params"]["linewidth"]
            window = lw//2+1
            
            best_prob = np.zeros((len(pred_p), len(cand_p)))
            best_p = np.zeros((len(pred_p), len(cand_p)))
            
            for idx, p in enumerate(cand_p):
                up = max(0, int(p-window))
                down = min(img_rgb.shape[0], int(p+window+1))
                pp = np.array(range(up,down))
                cand_c = img_rgb[pp, t]
                prob = _comp_prob(pred_p,pred_c,pp,cand_c)
                best_prob[:,idx] = np.amax(prob, axis=1)
                best_p[:,idx] = pp[np.argmax(prob, axis=1)]
            return best_prob, best_p
        
        def dfs(seq, visited, score, prob, dfs_threshold):
            if score < self.best_score:
                return None
            if len(seq) == prob.shape[0]:
                return score, seq
            node_id = len(seq)
            indices = [None]+list(np.argsort(-1*prob[node_id]))
            for c in indices:
                if c is None:
                    result = dfs(seq+[c], visited, score*dfs_threshold, prob, dfs_threshold)
                else:
                    if visited[c]:
                        continue
                    prev_seq = np.array(seq)[np.array(seq)!=None]
                    if len(prev_seq) > 0 and max(prev_seq) > c:
                        continue
                    visited[c]=True
                    result = dfs(seq+[c], visited, score*prob[node_id,c], prob, dfs_threshold)
                    
                if result is not None:
                    f_score, f_seq = result
                    if f_score > self.best_score:
                        self.best_score = f_score
                        self.best_seq = f_seq
                if c is not None:
                    visited[c] = False
        
        def fusion(t, pred_p, pred_c, dfs_threshold=0.995):
            seg_map = self.visual_dict["display"]["seg_map"]
            cand_p, _ = find_peaks(seg_map[:, t], 0.)
            self.best_score, self.best_seq = 0, [None]*len(pred_p)
            
            if len(cand_p) == 0:
                return [], []
            elif len(cand_p) > 20:
                prob, cand_p = comp_prob(pred_p,pred_c,cand_p, t)
                prob_max = np.amax(prob, axis=0)
                ids = np.argsort(-1*prob_max)[:20]
                prob, cand_p = prob[:, idx], cand_p[:, idx]
            else:
                prob, cand_p = comp_prob(pred_p,pred_c,cand_p, t)
                
            visited = [False]*prob.shape[1]
            dfs(seq=[], visited=visited, score=1., prob=prob, dfs_threshold=dfs_threshold)
            return cand_p, prob
        
        
        if posi_s == posi_e:
            return
        stride = 1 if posi_e > posi_s else -1
        if LK_method:
            grads = self.visual_dict["display"]["LK_grads"]
        else:
            grads = self.visual_dict["display"]["grads"]
        seg_map = self.visual_dict["display"]["seg_map"]
        img_rgb = self.visual_dict["display"]["img_rgb"]
        
        prev_p, _ = find_peaks(seg_map[:, posi_s], 0.)
        pred_c = img_rgb[prev_p, posi_s]
        
        for t in tqdm(range(posi_s, posi_e, stride), desc= "forward" if stride==1 else "backward"):
            # 
            self.visual_dict["result"]["points"][t] = prev_p.copy()
            # prediction with optical flow estimation
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                _grad = grads[p, t]
                _grads.append(_grad)
                pred_p[idx] = round(float(p) + stride*_grad)
            self.visual_dict["result"]["selected_grads"][t] = np.array(_grads)
            
            if mode == "optical_flow_only":
                prev_p = np.array(np.clip(pred_p,0,seg_map.shape[0]-1), np.int32)
            elif mode == "optical_flow_and_semantic_seg":
                cand_p, prob = fusion(t+stride, pred_p, pred_c, dfs_threshold=dfs_threshold)
                self.visual_dict["result"]["prob"][stride*t] = prob.copy()
                self.visual_dict["result"]["cand_p"][t+stride] = cand_p.copy()
                self.visual_dict["result"]["best_score"][t+stride] = self.best_score
                self.visual_dict["result"]["best_seq"][t+stride] = self.best_seq.copy()

                # updates
                confidence = []
                real_grads = []
                for idx, p in enumerate(self.best_seq):
                    if p is None:
                        p, c = pred_p[idx], pred_c[idx]
                        p = min(max(p,0),seg_map.shape[0]-1)
                        real_grads.append(p-prev_p[idx])
                        prev_p[idx] = int(p)
                        confidence.append(0)
                    else:
                        real_grads.append(cand_p[idx, p]-prev_p[idx])
                        prev_p[idx] = cand_p[idx, p]
                        confidence.append(1)
                self.visual_dict["result"]["confidence"][t+stride] = confidence
                self.visual_dict["result"]["real_selected_grads"][t*stride] = np.array(real_grads)
                
        
    def _flow_all(self, posi, dfs_threshold=0.99, mode="optical_flow_only", LK_method=True):
        seg_map = self.visual_dict["display"]["seg_map"]
        print("anchor position: {}".format(posi))
#         print("forward pass >>>>>>>>>>>>>>>>>")
        self._flow(posi, seg_map.shape[1]-1, dfs_threshold=dfs_threshold, mode=mode, LK_method=LK_method)
#         print("backward pass >>>>>>>>>>>>>>>>>")
        self._flow(posi, 0, dfs_threshold=dfs_threshold, mode=mode, LK_method=LK_method)
        
    def cancel_rescale(self):
        if not hasattr(self, "rescale_intervals"):
            points = self.visual_dict["result"]["points"]
            lines_from_points = self.points2lines(points)
            self.visual_dict["result"]["lines"] = lines_from_points
            return
        points = self.visual_dict["result"]["points"]
        lines_from_points = self.points2lines(points)
        s = 0
        widths = []
        rescale_factors = []
        for t in self.rescale_intervals:
            if t < 0:
                widths.append(4*(abs(t)-s))
                rescale_factors.append(4)
            else:
                widths.append(abs(t)-s)
                rescale_factors.append(1)
            s = abs(t)
        new_lines = []
        for idx, [value_axis, t_axis] in enumerate(lines_from_points):
            s = 0
            value_axis = [-1]+list(value_axis)+[-1]
            t_axis = [-1]+list(t_axis)+[-1]
            new_value_axis, new_t_axis = [], []
            for j in range(len(widths)):
                factor = rescale_factors[j]
                v = value_axis[s:s+widths[j]]
                x = np.linspace(0,1,widths[j])
                xx = np.linspace(0,1,widths[j]//factor)
                vv = np.interp(xx, x, v)
                new_value_axis += list(vv)
                s = s+widths[j]
            new_t_axis = list(range(len(new_value_axis)))
            new_lines.append([np.array(new_value_axis[1:-1]), np.array(new_t_axis[1:-1])])
        self.visual_dict["result"]["lines"] = new_lines
        self.visual_dict["display"]["img_rgb_rescale"] = self.visual_dict["display"]["img_rgb"].copy()
        self.visual_dict["display"]["img_gray_rescale"] = self.visual_dict["display"]["img_gray"].copy()
        self.visual_dict["display"]["seg_map_rescale"] = self.visual_dict["display"]["seg_map"].copy()
        self.visual_dict["display"]["img_rgb"] = self.visual_dict["display"]["img_rgb_0"].copy()
        self.visual_dict["display"]["img_gray"] = self.visual_dict["display"]["img_gray_0"].copy()
        self.visual_dict["display"]["seg_map"] = self.visual_dict["display"]["seg_map_0"].copy()

            
    
    def flow(self, posi_id=None, dfs_threshold=0.95, grad_threshold=3., mode="optical_flow_only",  LK_method=True):
        if self.verbose:
            normed_pixel = self.visual_dict["plot_params"]["normed_dist_of_pixel"]
            normed_color = 255.
            print("pixel dist: {}, color dist: {}".format((-1*np.log(dfs_threshold))**0.5*normed_pixel, (-1*np.log(dfs_threshold))**0.5*normed_color))
        
        self.visual_dict["result"] = {
            "points": {},
            "cand_p": {},
            "prob": {},
            "confidence": {},
            "selected_grads": {},
            "real_selected_grads": {},
            "best_score": {},
            "best_seq": {},
            "mode": mode,
        }
        
        def find_init_posi(threshold = 3., LK_method=LK_method):
            ids = self.visual_dict["plot_params"]["start_ids"]
            seg_map = self.visual_dict["display"]["seg_map"]
            if LK_method:
                grads = self.visual_dict["display"]["LK_grads"]
            else:
                grads = self.visual_dict["display"]["grads"]
            start_posi = []
            for idx in ids:
                p, _ = find_peaks(seg_map[:, idx], 0.)
                max_grad = max(grads[p, idx])
                if max_grad < threshold:
                    start_posi.append(idx)
            self.visual_dict["plot_params"]["start_posi"] = start_posi
            if self.verbose:
                print("num of start position: {}".format(len(start_posi)))

        if "grad_threshold" not in self.visual_dict["plot_params"] or ("grad_threshold" in self.visual_dict["plot_params"] and self.visual_dict["plot_params"]["grad_threshold"]!=grad_threshold):
            find_init_posi(threshold = grad_threshold, LK_method=LK_method)
        
        start_posi = self.visual_dict["plot_params"]["start_posi"]
        assert len(start_posi) > 0
        posi = np.random.choice(start_posi) if posi_id is None else start_posi[posi_id]
        self._flow_all(posi, dfs_threshold=dfs_threshold, mode=mode, LK_method=LK_method)
        self.cancel_rescale()
        
        
        
            
    def visualization(self):
        seg_map = self.visual_dict["display"]["seg_map"]
        img_rgb = self.visual_dict["display"]["img_rgb"]
        ins_map = self.visual_dict["display"]["instance_pred"]
        colors = [np.random.rand(3) for i in range(20)]
        lines = self.visual_dict["result"]["lines"]
#         time_frames = list(range(0,seg_map.shape[1]))
#         canvas = np.ones_like(img_rgb)
        
        if  self.visual_dict["result"]["mode"] == "optical_flow_only":
            plt.figure(figsize=(20,20))
            plt.subplot(2,2,1)
            plt.imshow(img_rgb)
            plt.subplot(2,2,2)
            plt.imshow(ins_map)
            plt.subplot(2,2,3)
            plt.imshow(seg_map)
            plt.subplot(2,2,4)
            plt.imshow(seg_map)
#             for t in time_frames:
#                 if t in points:
#                     for j, v in enumerate(points[t]):
#                         plt.plot(t, v, "o", c=colors[j], markersize=8)
        elif  self.visual_dict["result"]["mode"] == "optical_flow_and_semantic_seg":
#             conf = self.visual_dict["result"]["confidence"]

            plt.figure(figsize=(20,30))
            plt.subplot(3,2,1)
            plt.imshow(img_rgb)
            plt.subplot(3,2,2)
            plt.imshow(ins_map)
            plt.subplot(3,2,3)
            plt.imshow(seg_map)
            plt.subplot(3,2,4)
            plt.imshow(img_rgb)
            for idx, [value_axis, t_axis] in enumerate(lines):
                plt.plot(t_axis, value_axis, "o", c=colors[idx], markersize=3)
#             for t in time_frames:
#                 if t in points:
#                     for j, v in enumerate(points[t]):
#                         plt.plot(t, v, "o", c=colors[j], markersize=8)
#             plt.subplot(3,2,5)
#             plt.imshow(canvas)
#             for t in time_frames:
#                 if (t in conf) and (t in points):
#                     for j, v in enumerate(points[t]):
#                         plt.plot(t, v, "o", c="r" if conf[t][j]==0 else "g", markersize=8)
#             plt.subplot(3,2,6)
#             plt.imshow(canvas)
#             for t in time_frames:
#                 if (t in conf) and (t in points):
#                     for j, v in enumerate(points[t]):
#                         if conf[t][j] == 1:
#                             plt.plot(t, v, "o", c=colors[j], markersize=8)
    
    
    def instance_based_rescale(self, threshold=30, factor=4, bw_factor=1.5):
        ins_map = self.visual_dict["display"]["instance_pred"]
        lines_from_ins = self.instancemap2lines(ins_map)
        score_ins = self.line_evaluation(lines_from_ins)
        
        peaks = []
        for i in range(len(score_ins)):
            peaks += list(find_peaks(score_ins[i][2], threshold=threshold)[0])
        if len(peaks) == 0:
            return 
        print("start rescaling >>>")
        peaks = np.array(sorted(peaks)).reshape(-1,1)
        bandwidth = 30
#         bandwidth = estimate_bandwidth(peaks, quantile=0.2)
        clustering = MeanShift(bandwidth=bandwidth).fit(peaks)
        intervals = []
        for c in sorted(clustering.cluster_centers_):
            intervals.append(int(c-bw_factor*bandwidth))
            intervals.append(-1*int(c+bw_factor*bandwidth))
        intervals.append(ins_map.shape[1])
        
        new_intervals = []
        for t in intervals:
            if len(new_intervals) > 0 and abs(t) < abs(new_intervals[-1]):
                del new_intervals[-1]
            else:
                new_intervals.append(t)
        if new_intervals[-1] != ins_map.shape[1]:
            new_intervals.append(-1*ins_map.shape[1])
        intervals = new_intervals
        
        if self.verbose:
            print("seg intervals: ", intervals)
        
        seg_patches = []
        gray_patches = []
        rgb_patches = []
        seg_map = self.visual_dict["display"]["seg_map"]
        img_gray = self.visual_dict["display"]["img_gray"]
        img_rgb = self.visual_dict["display"]["img_rgb"]
        
        self.visual_dict["display"]["seg_map_0"] = seg_map
        self.visual_dict["display"]["img_gray_0"] = img_gray
        self.visual_dict["display"]["img_rgb_0"] = img_rgb
        self.rescale_intervals = intervals
        s = 0
        for t in intervals:
            seg_patch = seg_map[:, s:abs(t)]
            gray_patch = img_gray[:, s:abs(t)]
            rgb_patch = img_rgb[:, s:abs(t)]
            if t < 0:
                seg_patch = cv2.resize(seg_patch, (factor*(abs(t)-s),img_rgb.shape[0]), interpolation = cv2.INTER_NEAREST)
                gray_patch = cv2.resize(gray_patch, (factor*(abs(t)-s),img_rgb.shape[0]), interpolation = cv2.INTER_AREA)
                rgb_patch = cv2.resize(rgb_patch, (factor*(abs(t)-s),img_rgb.shape[0]), interpolation = cv2.INTER_AREA)
        
            seg_patches.append(seg_patch)
            gray_patches.append(gray_patch)
            rgb_patches.append(rgb_patch)
            s = abs(t)
        self.visual_dict["display"]["seg_map"] = np.concatenate(seg_patches, axis=1)
        self.visual_dict["display"]["img_gray"] = np.concatenate(gray_patches, axis=1)
        self.visual_dict["display"]["img_rgb"] = np.concatenate(rgb_patches, axis=1)
            
        
    def flow_prep(self, normed_dist_of_pixel=255., LK_method=True):
        if hasattr(self, "rescale_intervals"):
            delattr(self, "rescale_intervals")
        def linewidth_estimation():
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
        def comp_grad(im):
            gradients_t = np.gradient(im, axis=1, edge_order=1)
            gradients_x = np.gradient(im, axis=0, edge_order=1)
            assert gradients_x.shape == gradients_t.shape
            u = np.zeros_like(gradients_x)
            u[gradients_x!=0] =-1* gradients_t[gradients_x!=0] / gradients_x[gradients_x!=0]
            return u
            
        self.instance_based_rescale()
        self.visual_dict["plot_params"] = {
            "normed_dist_of_pixel": normed_dist_of_pixel,
        }
        img_gray = self.visual_dict["display"]["img_gray"]
        img_rgb = self.visual_dict["display"]["img_rgb"]
        grads = comp_grad(img_gray)
        self.visual_dict["display"]["grads"] = grads
        if LK_method:
            linewidth_estimation()
            lw = self.visual_dict["plot_params"]["linewidth"]
            grads = gaussian_filter1d(grads, lw/6, axis = 0)
            self.visual_dict["display"]["LK_grads"] = grads
        
        
    
    def load(self, idx, denoise=True):
        self.dp_engine.instance_seg(idx)
        self.visual_dict = {}
        
        raw_img = self.dp_engine.visual_dict["raw_data"]
        source_img = self.dp_engine.visual_dict["source_img"]
        instance_pred = self.dp_engine.visual_dict["instance_pred"]
        seg_map = self.dp_engine.visual_dict["seg_map"]
        
        if denoise:
            source_img = cv2.fastNlMeansDenoisingColored(source_img,None,10,10,7,21)
            source_img = cv2.bilateralFilter(source_img, 15, 75, 75)
        img_gray = 1.-cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)/255.
        img_rgb = source_img/255.
        
        self.visual_dict["display"] = {}
        self.visual_dict["display_targets"] = ["raw_img","img_gray", "img_rgb","seg_map", "instance_pred"]
        for target in self.visual_dict["display_targets"]:
            self.visual_dict["display"][target] = eval(target)