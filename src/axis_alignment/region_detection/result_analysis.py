import glob
import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import itertools

from .refinement import refine
from .detection import PlotDetector


class Analyzer():
    def __init__(self):
        print("build analyzer ...")
        self.is_test = False
    
    
    
    def Run(self, num_len, num_dist):
        results = {}
        # detector
        self.compute_bbox("detector")
        diffs = self.compute_metric("detector")
        diffs = np.array([d for d in diffs if d is not None])
        ref_x_diff, ref_y_diff = np.mean(diffs, axis=0)
        
        results["valid"] = diffs.shape
        results["reference"] = [ref_x_diff, ref_y_diff]
        
        # refine
        len_thres = np.linspace(0.1,0.9,num_len)
        max_dists = np.linspace(5,40,num_dist)
        param_space = list(itertools.product(len_thres, max_dists))
        results["param_space"] = param_space
        for len_threshold, max_dist in param_space:
            kwargs = {
                "len_threshold": len_threshold,
                "angle_threshold": 0.98,
                "max_dist": max_dist,
                "morphology": True,
            }
            self.compute_bbox("refine", **kwargs)
        diffs = self.compute_metric("refine")
        ans = []
        for idx in range(len(diffs)):
            diff = np.array([d for d in diffs[idx][4] if d is not None])
            ans.append(diff)
        ans = np.array(ans)
        results["refine_results"] = ans
        return results
    
    
    
    def compute_metric(self, mode="detector"):
        assert self.is_test
        if not hasattr(self, mode+"_results"):
            return []
        results = getattr(self, mode+"_results")
        diffs = []
        if mode == "detector":
            for annot_id, annot_file in tqdm(enumerate(self.annot_list)):
                with open(annot_file, "r") as fp:
                    annot = json.load(fp)
                points = annot["shapes"][0]["points"]
                gt_x1 = min(np.array(points)[:,0])
                gt_y2 = max(np.array(points)[:,1])
                bbox = results[annot_id]
                if bbox is None:
                    diffs.append(None) 
                else:
                    x1,_,_,y2 = bbox
                    x_diff, y_diff = abs(x1-gt_x1), abs(y2-gt_y2)
                    diffs.append([x_diff, y_diff])
        elif mode == "refine":
            for p1,p2,p3,p4,result in results:
                diff = []
                for annot_id, annot_file in tqdm(enumerate(self.annot_list)):
                    with open(annot_file, "r") as fp:
                        annot = json.load(fp)
                    points = annot["shapes"][0]["points"]
                    gt_x1 = min(np.array(points)[:,0])
                    gt_y2 = max(np.array(points)[:,1])
                    bbox = result[annot_id]
                    if bbox is None:
                        diff.append(None) 
                    else:
                        x1,_,_,y2 = bbox
                        x_diff, y_diff = abs(x1-gt_x1), abs(y2-gt_y2)
                        diff.append([x_diff, y_diff])
                diffs.append([p1,p2,p3,p4, diff])
        else:
            raise NotImplementedError
            
        return diffs
        

            
    
    def compute_bbox(self, mode="detector", **kwargs):
        results = []
        if mode == "detector":
            for img_id, img_path in tqdm(enumerate(self.imglist)):
                detection = self.detector.detection(img_path)
                img = Image.open(img_path).convert("RGB")
                try:
                    x1,y1,x2,y2 = detection[0][0][:4]
                    results.append([x1,y1,x2,y2])
                except:
                    results.append(None)
            self.detector_results = results
        elif mode == "refine":
            if not hasattr(self, "detector_results"):
                self.compute_bbox("detector")
            detector_results = self.detector_results
            len_threshold = kwargs.get("len_threshold", 0.5)
            angle_threshold = kwargs.get("angle_threshold", 0.98)
            max_dist = kwargs.get("max_dist", 30)
            morphology = kwargs.get("morphology", True)
            for img_id, img_path in tqdm(enumerate(self.imglist)):
                detection = detector_results[img_id]
                img = Image.open(img_path).convert("RGB")
                if detection is None:
                    results.append(None)
                else:
                    x1,y1,x2,y2 = detection
                    re_x1, re_y1, re_x2, re_y2 = refine(img, 
                                                [x1,y1,x2,y2],
                                                len_threshold=len_threshold,
                                                angle_threshold = angle_threshold,
                                                max_dist = max_dist,
                                                morphology = morphology)
                    results.append([re_x1, re_y1, re_x2, re_y2])
            if not hasattr(self, "refine_results"):
                self.refine_results = [[len_threshold, angle_threshold, max_dist,
                                       morphology, results]]
            else:
                self.refine_results.append([len_threshold, angle_threshold, max_dist,
                                       morphology, results])
        else:
            raise NotImplementedError
            
            
    
    
    def load_detector(self, config_file, checkpoint_file, device="cuda:0"):
        self.detector = PlotDetector()
        self.detector.load_model(config_file, checkpoint_file, device)
        
        
    def check_data_availablity(self):
        annot_list = self.annot_list
        imglist = []
        for idx, annot_file in tqdm(enumerate(annot_list)):
            img_path = annot_file.replace(".json", ".jpg")
            assert os.path.exists(img_path), "{} not found".format(img_path)
            imglist.append(img_path)
        self.imglist = imglist
    
    def load_data(self, folder, mode="image"):
        if mode == "image":
            imglist = []
            for img_ext in ["jpg", "png"]:
                imglist.extend(glob.glob(os.path.join(folder,"*.{}".format(img_ext))))
            self.imglist = imglist
            self.is_test = False
        elif mode == "annotation":
            self.annot_list = glob.glob(os.path.join(folder,"*.json"))
            self.check_data_availablity()
            self.is_test = True
        print("{} images are available!!!".format(len(self.imglist)))