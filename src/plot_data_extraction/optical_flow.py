import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

from .utils import ComputeGrad, ComputeProb, dfs, ComputeDist, points2lines

class OpticalFlow():
    def __init__(self, img_rgb, img_gray, seg_map):
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.seg_map = seg_map
        self.results = {}
        

    def flow_prep(self, mode, momentum, 
                  momentum_factor=0.3, overlap_threshold = 0., 
                  keep_order_threshold=0., match_threshold=0.95,
                 gradient_threshold = 20,color_neighbor_size=20,
                 color_threshold=0.95):
        self.compute_grad()
        self.mode = mode
        self.momentum = momentum
        self.momentum_factor = momentum_factor
        self.overlap_threshold = overlap_threshold
        self.keep_order_threshold = keep_order_threshold
        self.match_threshold = match_threshold
        self.gradient_threshold = gradient_threshold
        self.color_neighbor_size = color_neighbor_size
        self.color_threshold = color_threshold
        
        
    def visualization(self, display=False, save=False, **kwargs):
        colors = [np.random.rand(3) for _ in range(100)]
        img_rgb = self.img_rgb
        seg_map = self.seg_map
        canvas = np.ones_like(img_rgb)
        plt.figure(figsize=(20,10))
        plt.subplot(2,3,1)
        plt.imshow(img_rgb)
        plt.title("img_rgb")
        plt.subplot(2,3,2)
        plt.imshow(seg_map)
        plt.title("seg_map")
        plt.subplot(2,3,3)
        plt.imshow(canvas)
        plot_ts, plot_lines = self.line_set["plot"]
        for line_id in range(plot_lines.shape[1]):
            plt.plot(plot_ts, plot_lines[:,line_id], "o", c=colors[line_id], markersize=3)
        plt.title("pred_map")
        plt.subplot(2,3,4)
        plt.imshow(canvas)
        conf_ts, conf_lines = self.line_set["conf"]
        for line_id in range(conf_lines.shape[1]):
            for idx, t in enumerate(conf_ts):
                plt.plot(t, plot_lines[plot_ts.tolist().index(t),line_id], 
                         "go" if conf_lines[idx, line_id] == 1 else "ro", 
                         markersize=3)
        plt.title("conf_map")
        plt.subplot(2,3,5)
        plt.imshow(canvas)
        grad_ts, grad_lines = self.line_set["grad"]
        for line_id in range(grad_lines.shape[1]):
            for idx, t in enumerate(grad_ts):
                plt.plot(t, plot_lines[plot_ts.tolist().index(t),line_id], 
                         "o", c = colors[int(round(grad_lines[idx, line_id]))], 
                         markersize=3)
        plt.title("grad_map")
        plt.subplot(2,3,6)
        plt.imshow(canvas)
        thre_ts, thre_lines = self.line_set["thre"]
        for line_id in range(grad_lines.shape[1]):
            for idx, t in enumerate(grad_ts):
                plt.plot(t, plot_lines[plot_ts.tolist().index(t),line_id], 
                         "o", c = colors[int(round(100*thre_lines[idx, line_id]))], 
                         markersize=3)  
        plt.title("dfs_thres")
        
        if save:
            save_path = kwargs.get("save_path", None)
            plt.savefig(save_path,bbox_inches="tight", pad_inches=0.1)
        
        if not display:
            plt.close()
    
    
    def refinement(self, plot_ts, plot_lines):
        img_rgb = self.img_rgb
        # smooth
        plot_lines = gaussian_filter1d(plot_lines, 2, axis=0)
        # color
        for line_id in range(plot_lines.shape[1]):
            line = plot_lines[:, line_id]
            color_collection = img_rgb[np.round(line).astype(int), plot_ts.astype(int)]
            color_avg = np.mean(color_collection, axis=0)
            for idx, p in enumerate(line):
                p = int(round(p))
                for j in range(self.color_neighbor_size):
                    pj = np.clip(p+j, 0, img_rgb.shape[0]-1)
                    color_dist = ComputeDist(img_rgb[pj,int(plot_ts[idx])], color_avg, mode = "color")
                    if color_dist > self.color_threshold:
                        plot_lines[idx, line_id] = pj
                        break
                    pj = np.clip(p-j, 0, img_rgb.shape[0]-1)
                    color_dist = ComputeDist(img_rgb[pj,int(plot_ts[idx])], color_avg, mode = "color")
                    if color_dist > self.color_threshold:
                        plot_lines[idx, line_id] = pj
                        break
        return plot_lines
            
    
    
    
    def generate_lines(self):
        plot_ts, plot_lines = points2lines(self.results["points"])
        try:
            conf_ts, conf_lines = points2lines(self.results["confidence"])
        except:
            conf_ts, conf_lines = None, None
        try:
            grad_ts, grad_lines = points2lines(self.results["actual_grads"])
        except:
            grad_ts, grad_lines = None, None
        try:
            thre_ts, thre_lines = points2lines(self.results["match_threshold"])
        except:
            thre_ts, thre_lines = None, None
        plot_lines = self.refinement(plot_ts, plot_lines)
        
        self.line_set = {
            "plot": [plot_ts, plot_lines],
            "conf": [conf_ts, conf_lines],
            "grad": [grad_ts, grad_lines],
            "thre": [thre_ts, thre_lines]
        }
    
    
    def flow(self, start_posi):
        self._flow(start_posi, self.img_rgb.shape[1]-1)
        self._flow(start_posi, 0)
        self.generate_lines()
        
    
    def flow_with_mode(self, **kwargs):
        
        prev_p = kwargs.get("prev_p", None)
        t = kwargs.get("t", None)
        prev_grads = kwargs.get("prev_grads", None)
        grad_map_seg = kwargs.get("grad_map_seg", None)
        grad_map_gray = kwargs.get("grad_map_gray", None)
        stride = kwargs.get("stride", None)
        seg_map = kwargs.get("seg_map", None)
        overlap_threshold = kwargs.get("overlap_threshold", None)
        keep_order_threshold = kwargs.get("keep_order_threshold", None)
        match_threshold = kwargs.get("match_threshold", None)
        prev_c = kwargs.get("prev_c", None)
        img_rgb = kwargs.get("img_rgb", None)
        color_neighbor_size = kwargs.get("color_neighbor_size", 10)
        color_threshold = kwargs.get("color_threshold", None)

        
        
        if self.mode == "seg_grads":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                _grad = grad_map_seg[p, t]
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                _grads.append(_grad)
                pred_p[idx] = float(p) + stride*_grad
            if "estimated_grads" not in self.results:
                self.results["estimated_grads"] = {}
            self.results["estimated_grads"][t] = np.array(_grads)
            return pred_p
        
        
        
        if self.mode == "gray_grads":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                _grad = grad_map_gray[p, t]
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                _grads.append(_grad)
                pred_p[idx] = float(p) + stride*_grad
            if "estimated_grads" not in self.results:
                self.results["estimated_grads"] = {}
            self.results["estimated_grads"][t] = np.array(_grads)
            return pred_p
        
        
        
        
        if self.mode == "seg_gray_grads":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = float(p) + stride*_grad
            if "estimated_grads" not in self.results:
                self.results["estimated_grads"] = {}
            self.results["estimated_grads"][t] = np.array(_grads)
            return pred_p
        
        
        
        
        if self.mode == "seg_gray_grads_posi_correction":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
            match_threshold = [match_threshold]*len(pred_p)
            pred_p, cand_p, best = self.posi_compensation(t+stride, pred_p, 
                                                          match_threshold,
                                                          overlap_threshold,
                                                          keep_order_threshold)
            if "best" not in self.results:
                self.results["best"] = {}
            self.results["best"][t+stride] = best
            if "cand" not in self.results:
                self.results["cand"] = {}
            self.results["cand"][t+stride] = cand_p
            return pred_p
        
        
        if self.mode == "seg_gray_grads_color_correction":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
            

            
            # color compensation
            pred_p = self.color_compensation(t+stride, pred_p, prev_c,
                                             color_neighbor_size, color_threshold)
            
            

            return pred_p
        
        
        if self.mode == "seg_gray_grads_posi_color_correction":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
            
            # posi compensation
            match_threshold = [match_threshold]*len(pred_p)
            pred_p, cand_p, best = self.posi_compensation(t+stride, pred_p, 
                                                          match_threshold,
                                                          overlap_threshold,
                                                          keep_order_threshold)
            
            # color compensation
            pred_p = self.color_compensation(t+stride, pred_p, prev_c,
                                             color_neighbor_size, color_threshold)
            
            
            if "best" not in self.results:
                self.results["best"] = {}
            self.results["best"][t+stride] = best
            if "cand" not in self.results:
                self.results["cand"] = {}
            self.results["cand"][t+stride] = cand_p
            return pred_p
        
        
        if self.mode == "seg_gray_grads_posi_color_correction_grad_rejection":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # large (abnormal) gradient suppresion
                if abs(_grad) > self.gradient_threshold:
                    _grad = 0
                
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
            
            # posi compensation
            match_threshold = [match_threshold]*len(pred_p)
            pred_p, cand_p, best = self.posi_compensation(t+stride, pred_p, 
                                                          match_threshold,
                                                          overlap_threshold,
                                                          keep_order_threshold)
            
            # color compensation
            pred_p = self.color_compensation(t+stride, pred_p, prev_c,
                                             color_neighbor_size, color_threshold)
            
            
            if "best" not in self.results:
                self.results["best"] = {}
            self.results["best"][t+stride] = best
            if "cand" not in self.results:
                self.results["cand"] = {}
            self.results["cand"][t+stride] = cand_p
            return pred_p
        
        if self.mode == "seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs":
            pred_p = np.zeros(prev_p.shape)
            _grads = []
            for idx, p in enumerate(prev_p):
                # no previous grads
                if prev_grads[idx] is None:
                    _grad = grad_map_seg[p, t] if seg_map[p,t] > 0 else grad_map_gray[p, t]
                else:
                    _grad = grad_map_seg[p, t] if abs(grad_map_seg[p, t]-prev_grads[idx]) < abs(grad_map_gray[p, t]-prev_grads[idx]) else grad_map_gray[p, t]
                    
                # large (abnormal) gradient suppresion
                if abs(_grad) > self.gradient_threshold:
                    _grad = 0
                
                # momentum
                if self.momentum and prev_grads[idx] is not None:
                    _grad = self.momentum_factor*prev_grads[idx]+(1-self.momentum_factor)*_grad
                
                pred_p[idx] = np.clip(float(p) + stride*_grad, 0, seg_map.shape[0]-1)
            
            # posi compensation
            pixel_diff = abs(pred_p.astype(np.float32) - prev_p.astype(np.float32))*4
            match_threshold = np.clip(np.exp(-1*(pixel_diff/200)**2), 0, match_threshold)
            if "match_threshold" not in self.results:
                self.results["match_threshold"] = {}
            self.results["match_threshold"][t+stride] = match_threshold
            pred_p, cand_p, best = self.posi_compensation(t+stride, pred_p, 
                                                          match_threshold,
                                                          overlap_threshold,
                                                          keep_order_threshold)
            
            # color compensation
            pred_p = self.color_compensation(t+stride, pred_p, prev_c,
                                             color_neighbor_size, color_threshold)
            
            
            if "best" not in self.results:
                self.results["best"] = {}
            self.results["best"][t+stride] = best
            if "cand" not in self.results:
                self.results["cand"] = {}
            self.results["cand"][t+stride] = cand_p
            return pred_p
            
            
    def color_compensation(self, t, pred_p, prev_c, color_neighbor_size, color_threshold):
        seg_map = self.seg_map
        img_rgb = self.img_rgb
        next_p = pred_p.copy()
        for idx, p in enumerate(next_p):
            p = int(round(p))
            for j in range(color_neighbor_size):
                pj = np.clip(p+j, 0, seg_map.shape[0]-1)
                color_dist = ComputeDist(img_rgb[pj,t], prev_c[idx], mode = "color")
                if color_dist > color_threshold:
                    next_p[idx] = pj
                    break
                pj = np.clip(p-j, 0, seg_map.shape[0]-1)
                color_dist = ComputeDist(img_rgb[pj,t], prev_c[idx], mode = "color")
                if color_dist > color_threshold:
                    next_p[idx] = pj
                    break
        return next_p
    
    
    
    def posi_compensation(self, t, pred_p, match_threshold, 
                          overlap_threshold, keep_order_threshold):
        seg_map = self.seg_map
        cand_p, _ = find_peaks(seg_map[:, t], 0.)
        best = {"score": 0., "seq": [None]*len(pred_p)}
        
        if len(cand_p) == 0:
            confidence = np.zeros_like(pred_p)
            if "confidence" not in self.results:
                self.results["confidence"] = {}
            self.results["confidence"][t] = confidence
            return pred_p, cand_p, best
        else:
            prob = ComputeProb(pred_p, cand_p)
            prob_max = np.amax(prob, axis=0)
            ids = sorted(np.argsort(-1*prob_max)[:20])
            prob, cand_p = prob[:, ids], cand_p[ids]
            
            visited = [False]*prob.shape[1]
            dfs([], visited, 1., prob, 
                match_threshold, overlap_threshold, 
                keep_order_threshold, best)
            next_p = np.zeros_like(pred_p)
            confidence = np.ones_like(pred_p)
            best_seq = np.array(best["seq"])
            next_p[best_seq == None] = pred_p[best_seq == None]
            confidence[best_seq == None] = 0
            next_p[best_seq != None] = cand_p[best_seq[best_seq != None].astype(np.int32)]
            if "confidence" not in self.results:
                self.results["confidence"] = {}
            self.results["confidence"][t] = confidence
            return next_p, cand_p, best
        
        
        
                
    
    def _flow(self, posi_s, posi_e):
        if posi_s == posi_e:
            return
        stride = 1 if posi_e > posi_s else -1
        grad_map_seg, grad_map_gray = self.grad_map_seg, self.grad_map_gray
        seg_map, img_rgb = self.seg_map, self.img_rgb
        
        prev_p, _ = find_peaks(seg_map[:, posi_s], 0.)
        prev_c = img_rgb[prev_p, posi_s]
        prev_grads = np.array([None]*len(prev_p))
        
#         for t in tqdm(range(posi_s, posi_e, stride), desc= "forward" if stride==1 else "backward"):
        for t in range(posi_s, posi_e, stride):
            if "points" not in self.results:
                self.results["points"] = {}
            self.results["points"][t] = prev_p.copy()
            kwargs = {
                "prev_p": prev_p,
                "prev_c": prev_c,
                "prev_grads": prev_grads,
                "grad_map_seg": grad_map_seg,
                "grad_map_gray": grad_map_gray,
                "seg_map": seg_map,
                "img_rgb": img_rgb,
                "stride": stride,
                "t" : t,
                "overlap_threshold": self.overlap_threshold,
                "keep_order_threshold": self.keep_order_threshold,
                "match_threshold": self.match_threshold,
                "color_neighbor_size": self.color_neighbor_size,
                "color_threshold": self.color_threshold,
            }
            
            
            pred_p = self.flow_with_mode(**kwargs)
            pred_p = np.round(np.clip(pred_p, 0, img_rgb.shape[0]-1)).astype(np.int32)
            prev_grads = pred_p - prev_p
            if "actual_grads" not in self.results:
                self.results["actual_grads"] = {}
            self.results["actual_grads"][t] = prev_grads
            prev_p = pred_p
            
        if "points" not in self.results:
            self.results["points"] = {}
        self.results["points"][t+stride] = prev_p.copy()
            
    
    
    
    def compute_grad(self):
        img_gray_gaussian = gaussian_filter1d(self.img_gray, 2, axis=0)
        seg_map_gaussian = gaussian_filter1d(self.seg_map, 2, axis=0)
        self.grad_map_seg = ComputeGrad(seg_map_gaussian)
        self.grad_map_gray = ComputeGrad(img_gray_gaussian)
        