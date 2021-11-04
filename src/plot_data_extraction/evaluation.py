import os
import numpy as np
import shutil
import itertools
import glob
import json


from .optical_flow import OpticalFlow


class ResultEvaluation():
    def __init__(self):
        pass
    
    def load_json_list(self, gt_json_dir, pred_json_dir):
        self.gt_json_list = sorted(glob.glob(os.path.join(gt_json_dir, "*.json")))
        self.pred_json_list = sorted(glob.glob(os.path.join(pred_json_dir, "*.json")))
        for gt_json, pred_json in zip(self.gt_json_list, self.pred_json_list):
            assert gt_json.split("/")[-1] == pred_json.split("/")[-1]
    
    def align_dist(self, p1, p2):
        p1, p2 = np.array(p1), np.array(p2)
        p1_t_min, p1_t_max = max(0, int(min(p1[:,0]))), int(max(p1[:,0]))
        p2_t_min, p2_t_max = max(0, int(min(p2[:,0]))), int(max(p2[:,0]))
        p_t_min, p_t_max = max(p1_t_min, p2_t_min), min(p1_t_max, p2_t_max)
        p_ts = np.linspace(p_t_min, p_t_max, p_t_max-p_t_min+1)
        p1_real = np.interp(p_ts, p1[:,0], p1[:,1])
        p2_real = np.interp(p_ts, p2[:,0], p2[:,1])
        return np.mean(abs(p1_real-p2_real))
        
    def match(self, dist, visited, pixel_dist_threshold=10):
        dist_ids = np.linspace(0, len(dist)-1, len(dist)).astype(int)
        dist_ids = dist_ids[np.array(visited) == False]
        dist =np.array(dist)[np.array(visited) == False]
        if len(dist) > 0 and min(dist) < pixel_dist_threshold:
            return dist_ids[np.argmin(dist)]
        else:
            return None
        
    def Evaluate(self, pixel_dist_threshold, gt_json_dir, pred_json_dir):
        gt_json_list = sorted(glob.glob(os.path.join(gt_json_dir, "*.json")))
        results = {}
        for gt_json in gt_json_list:
            gt_json_name = gt_json.split("/")[-1]
            pred_json_list = sorted(glob.glob(os.path.join(pred_json_dir, "*"+gt_json_name)))
            result = [self.load_annot(gt_json, pred_json, pixel_dist_threshold) 
                      for pred_json in pred_json_list
                     ]
            result = np.array(result)
            results[gt_json_name]= [result, pred_json_list]
        return results
            
        
    
    def load_annot(self, gt_json, pred_json, pixel_dist_threshold=10):
        with open(gt_json, "r") as fp:
            gt_annot = json.load(fp)
        with open(pred_json, "r") as fp:
            pred_annot = json.load(fp)
            
        points_pred_list = [line_pred["points"] for line_pred in pred_annot["shapes"]]
        visited = [False]*len(points_pred_list)
        for line_gt in gt_annot["shapes"]:
            points_gt = line_gt["points"]
            dist = [self.align_dist(points_gt, points_pred) for points_pred in points_pred_list]
            match_id = self.match(dist, visited, pixel_dist_threshold)
            if match_id is not None:
                visited[match_id] = True
        recall = sum(visited)
        num_gt = len(gt_annot["shapes"])
        num_pred = len(points_pred_list)
        
        return recall, num_gt, num_pred


def SelectIDs(start_ids, num_posi):
    if len(start_ids) <= num_posi:
        select_ids = start_ids
    else:
        id_grads = np.gradient(start_ids)
        select_ids = start_ids[np.argsort(-1*id_grads)[:num_posi]]
    return select_ids


class PlotEvaluator():
    
    def __init__(self, img_rgb, img_gray, seg_map):
        self.img_rgb = img_rgb
        self.img_gray = img_gray
        self.seg_map = seg_map
        
        
    def line_evaluation(self, line_set):
        plot_ts, plot_lines = line_set["plot"]
        conf_ts, conf_lines = line_set["conf"]
        grad_ts, grad_lines = line_set["grad"]
        thre_ts, thre_lines = line_set["thre"]
        
        img_rgb = self.img_rgb
        
        # semantic
        conf_score = np.mean(conf_lines)
        
        # color
        color_score = []
        for line_id in range(plot_lines.shape[1]):
            line = plot_lines[:, line_id]
            color_1 = img_rgb[line.astype(int), plot_ts.astype(int)]
            color_2 = img_rgb[np.round(line).astype(int), plot_ts.astype(int)]
            color = 0.5*color_1+0.5*color_2
            color_score.append(np.mean(color.std(0)))
        color_score = 1-np.mean(color_score)
        
        # grad
        grad_score = 1-np.mean(abs(np.gradient(grad_lines, axis=0)))
        
        # line_span
        span_score = 0
        for line_id in range(1, plot_lines.shape[1]):
            span_score += np.mean(plot_lines[:,line_id]-plot_lines[:,line_id-1])/img_rgb.shape[0]
        return conf_score, color_score, grad_score, span_score
        
        
    
    
    def param_search(self, match_thresholds, start_ids, **kwargs):
        mode = kwargs.get("mode", "seg_gray_grads_posi_color_correction_grad_rejection_adaptive_dfs") 
        momentum = kwargs.get("momentum", True) 
        momentum_factor=kwargs.get("momentum_factor", 0.3)
        overlap_threshold = kwargs.get("overlap_threshold", 0.)
        keep_order_threshold=kwargs.get("keep_order_threshold", 0.)
        gradient_threshold = kwargs.get("gradient_threshold", 20)
        color_neighbor_size=kwargs.get("color_neighbor_size", 10)
        color_threshold=kwargs.get("color_threshold", 0.95)
        save_path = kwargs.get("save_path", "./tmp")
        num_posi = kwargs.get("num_posi", 5)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        
        select_ids = SelectIDs(start_ids, num_posi)
        
        scores = []
        line_sets = []
        param_space = list(itertools.product(select_ids, match_thresholds))
        for start_posi, match_threshold in param_space:
            optical_flow = OpticalFlow(self.img_rgb, self.img_gray, self.seg_map)
            optical_flow.flow_prep(mode, momentum, momentum_factor, 
                                   overlap_threshold, keep_order_threshold,
                                   match_threshold, gradient_threshold,
                                   color_neighbor_size, color_threshold)
            optical_flow.flow(start_posi)
            optical_flow.visualization(display=kwargs.get("display", False), 
                                       save=kwargs.get("save", False), 
                                       **{"save_path": os.path.join(save_path, "startPosi_{}_matchThreshold_{}_analysis.png".format(start_posi, match_threshold))})
            conf_score, color_score, grad_score, span_score = self.line_evaluation(optical_flow.line_set)
            scores.append([conf_score, color_score, grad_score, span_score])
            line_sets.append(optical_flow.line_set)
        return scores, param_space, line_sets
        
 
    