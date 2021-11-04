import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import json
from distinctipy import distinctipy

from .plot_digitizer import PlotDigitizer
from .SpatialEmbeddings.src.utils import transforms as my_transforms
from .evaluation import PlotEvaluator
from .utils import GenerateTestData, dict2class

class Test():
    def __init__(self, opt):
        plot_digitizer = PlotDigitizer()
        plot_digitizer.load_seg("spatialembedding", opt)
        self.plot_digitizer = plot_digitizer
        
    def GenerateResult(self, img_id, mode, match_threshold, json_dir, save_dir, threshold=2., num_posi=20):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plot_digitizer = self.plot_digitizer
        plot_digitizer.predict_from_ins_seg(img_id)
        plot_digitizer.find_init_posi(threshold=threshold)
        momentum = True
        overlap_threshold = 0.
        keep_order_threshold=0.
        gradient_threshold = 20
        color_neighbor_size=10
        color_threshold=0.95
        plot_digitizer.optical_flow.flow_prep(mode, 
                                      momentum,
                                     overlap_threshold = overlap_threshold,
                                     keep_order_threshold = keep_order_threshold,
                                     match_threshold = match_threshold,
                                     gradient_threshold = gradient_threshold,
                                     color_neighbor_size = color_neighbor_size,
                                     color_threshold = color_threshold)
        json_name = plot_digitizer.img_name.split("/")[-1].replace(".png", ".json")
        with open(os.path.join(json_dir, json_name), "r") as fp:
            annot = json.load(fp)
        
        img_rgb = plot_digitizer.result_dict["visual"]["img_rgb"]
        h,w,_ = img_rgb.shape
        img = Image.open(plot_digitizer.img_name)
        real_w, real_h = img.size
        real_ts = np.linspace(0, real_w-1, real_w)
        new_plot_ts = np.linspace(0, real_w-1, w)
        for start_posi in np.random.choice(plot_digitizer.result_dict["data"]["start_posi"], size=20):
            plot_digitizer.optical_flow.flow(start_posi)
            plot_digitizer.optical_flow.generate_lines()
            plot_ts, plot_lines = plot_digitizer.optical_flow.line_set["plot"]
            annot["shapes"] = []
            for plot_id in range(plot_lines.shape[1]):
                real_line = np.interp(real_ts, new_plot_ts, plot_lines[:,plot_id])
                points = np.array([real_ts, real_line/h*real_h]).transpose(1,0).tolist()
                annot["shapes"].append({
                    'label': 'plot',
                    'points': points,
                    'group_id': None,
                    'shape_type': 'linestrip',
                    'flags': {}
                })
            file_name = "{}_{}_{}".format(str(match_threshold).replace(".",""), start_posi, json_name)
            with open(os.path.join(save_dir, file_name), "w") as fp:
                json.dump(annot, fp, indent=2)
            
            
            
            
            
            
