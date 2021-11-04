import os
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import string
import cv2
import time
from datetime import datetime, date
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import norm
import shutil
import itertools


class dict2class():
    def __init__(self, opt):
        for key in opt.keys():
            setattr(self, key, opt[key])


def LineInpaint(line, width):
    value_axis, t_axis = line
    tt = np.linspace(0, width-1, width)
    vv = np.interp(tt, t_axis, value_axis)
    return vv

def Segmap2Lines(ins_map):
    width = ins_map.shape[1]
    ins_ids = sorted(np.unique(ins_map))
    if len(ins_ids) == 1: 
        return None
    plot_lines = []
    for idx, ins_id in enumerate(ins_ids[1:]):
        value_axis, t_axis = [], []
        for t in range(ins_map.shape[1]):
            p = np.where(ins_map[:,t] == ins_id)[0]
            if len(p) > 0:
                value_axis.append(np.median(p))
                t_axis.append(t)
        line = LineInpaint([value_axis, t_axis], width)
        plot_lines.append(line)
    plot_lines = np.array(plot_lines).transpose(1,0)
    plot_ts = np.linspace(0, width-1, width)
    return plot_ts, plot_lines
    
    


def GenerateTestData(root, save_path):
    imglist = []
    imglist.extend(glob.glob(os.path.join(root, "*.jpg")))
    imglist.extend(glob.glob(os.path.join(root, "*.png")))
    img_dir = os.path.join(save_path, "leftImg8bit", "test", "raman_xanes")
    label_dir = os.path.join(save_path, "gtFine", "test", "raman_xanes")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    for idx, img_path in enumerate(imglist):
        img_name = img_path.split("/")[-1]
        img = Image.open(img_path)
        instance_map = Image.new(mode="I", size=img.size)
        img.save(os.path.join(img_dir, img_name.replace(".jpg", ".png")))
        instance_map.save(os.path.join(label_dir, img_name.replace(".jpg", ".png")))
        

def dfs(seq, visited, score, prob, match_threshold, overlap_threshold, keep_order_threshold, best):
    if score <= best["score"]:
        return None
    if len(seq) == prob.shape[0]:
        if score > best["score"]:
            return score, seq
        else:
            return None
    node_id = len(seq)
    indices = list(np.argsort(-1*prob[node_id]))
    for c_id in indices:
        node_prob = prob[node_id,c_id]
        for c_prev in seq:
            if c_prev is not None and c_prev > c_id:
                node_prob = node_prob*keep_order_threshold
                break
        if visited[c_id]:
            node_prob = node_prob*overlap_threshold
            if node_prob > match_threshold[node_id]:
                ans = dfs(seq+[c_id], visited, score*node_prob, prob, match_threshold, overlap_threshold, keep_order_threshold, best)
                if ans is not None:
                    if ans[0] > best["score"]:
                        best["score"], best["seq"] = ans 
            else:
                ans = dfs(seq+[None], visited, score*match_threshold[node_id], prob, match_threshold, overlap_threshold, keep_order_threshold, best)
                if ans is not None:
                    if ans[0] > best["score"]:
                        best["score"], best["seq"] = ans
        else:
            visited[c_id] = True
            if node_prob > match_threshold[node_id]:
                ans = dfs(seq+[c_id], visited, score*node_prob, prob, match_threshold, overlap_threshold, keep_order_threshold, best)
                if ans is not None:
                    if ans[0] > best["score"]:
                        best["score"], best["seq"] = ans
            else:
                ans = dfs(seq+[None], visited, score*match_threshold[node_id], prob, match_threshold, overlap_threshold, keep_order_threshold, best)
                if ans is not None:
                    if ans[0] > best["score"]:
                        best["score"], best["seq"] = ans
            visited[c_id] = False

# def dfs(seq, visited, score, prob, match_threshold, overlap_threshold, keep_order_threshold, best):
#     if score <= best["score"]:
#         return False
#     if len(seq) == prob.shape[0] and score > best["score"]:
#         best["score"] = float(score)
#         best["seq"] = np.array(seq)
#         return True
#     node_id = len(seq)
#     indices = list(np.argsort(-1*prob[node_id]))
#     for c_id in indices:
#         node_prob = prob[node_id,c_id]
#         for c_prev in seq:
#             if c_prev is not None and c_prev > c_id:
#                 node_prob = node_prob*keep_order_threshold
#                 break
#         if visited[c_id]:
#             node_prob = node_prob*overlap_threshold
#             if node_prob > match_threshold[node_id]:
#                 ans = dfs(seq+[c_id], visited, score*node_prob, prob, match_threshold, overlap_threshold, keep_order_threshold, best)
#                 if not ans:
#                     break
#             else:
#                 ans = dfs(seq+[None], visited, score*match_threshold[node_id], prob, match_threshold, overlap_threshold, keep_order_threshold, best)
#                 if not ans:
#                     break
#         else:
#             visited[c_id] = True
#             if node_prob > match_threshold[node_id]:
#                 ans = dfs(seq+[c_id], visited, score*node_prob, prob, match_threshold, overlap_threshold, keep_order_threshold, best)
#                 if not ans:
#                     break
#             else:
#                 ans = dfs(seq+[None], visited, score*match_threshold[node_id], prob, match_threshold, overlap_threshold, keep_order_threshold, best)
#                 if not ans:
#                     break
#             visited[c_id] = False
            
        

def points2lines(points_dict):
    points_ts = np.array(sorted(points_dict.keys()))
    lines = np.array([points_dict[t] for t in points_ts])
    return points_ts, lines
    
    

def ComputeGrad(img):
    assert len(img.shape) == 2
    gradients_t = np.gradient(img, axis=1, edge_order=1)
    gradients_x = np.gradient(img, axis=0, edge_order=1)
    assert gradients_x.shape == gradients_t.shape
    u = np.zeros_like(gradients_x)
    u[gradients_x!=0] =-1* gradients_t[gradients_x!=0] / gradients_x[gradients_x!=0]
    return u


def ComputeDist(v1, v2, mode="pixel", **kwargs):
    if mode == "pixel":
        norm_factor = kwargs.get("norm_factor", 200)
        dist = (float(v1)-float(v2))/norm_factor
        dist = -1*dist**2
        return np.exp(dist)
    elif mode == "color":
        dist = (v1-v2)**2
        dist = -1*np.mean(dist)
        return np.exp(dist)
    else:
        raise NotImplementedError
        

def ComputeProb(pred_p, cand_p):
    prob = [ComputeDist(*t,mode="pixel") for t in list(itertools.product(pred_p, cand_p))]
    prob = np.array(prob).reshape((len(pred_p), len(cand_p)))
    return prob
    

class RealDatasimulator():
    def __init__(self, root, ratio=0.8):
        self.ratio = ratio
        self.load(root)
    
    def Run(self, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        self.GenerateData(save_path, "train")
        self.GenerateData(save_path, "val")
    
    def GenerateData(self, save_path, mode="train"):
        annot_list = self.annot_list
        class_id = 26
        img_dir = os.path.join(save_path, "leftImg8bit", mode, "real")
        label_dir = os.path.join(save_path, "gtFine", mode, "real")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        if mode == "train":
            ids = range(int(len(annot_list)*self.ratio))
        else:
            ids = range(int(len(annot_list)*self.ratio), len(annot_list))
        
        for idx in ids:
            img_name = annot_list[idx].split("/")[-1][:-5]
            with open(annot_list[idx], "r") as fp:
                annot = json.load(fp)
            img_path = annot_list[idx].replace(".json", ".png")
            assert os.path.exists(img_path)
            img = Image.open(img_path).convert("RGB")
            
            img_np = np.array(img)
            target = cv2.fastNlMeansDenoisingColored(img_np,None,10,10,7,21)
            gray = 255-cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

            instance_map = Image.new(mode="I", size=img.size)
            draw = ImageDraw.Draw(instance_map)
            for i, line in enumerate(annot["shapes"]):
                line = [(t[0], t[1]) for t in line["points"]]
                _, lw = self.width_detection(gray, line)
                lw = int(round(lw))
                draw.line(xy=line, width=lw, fill=class_id*1000+i)
            img.save(os.path.join(img_dir, "%s.png"%img_name))
            instance_map.save(os.path.join(label_dir, "%s.png"%img_name))
        
    
    def width_detection(self, gray, line, neighbor=5):
        lws = []
        for x, y in line:
            x = int(x)
            y = int(y)
            if x < 0 or x > gray.shape[1] or y<0 or y>gray.shape[0]:
                continue
            up = max(0, y-neighbor)
            down = min(gray.shape[0], y+neighbor)
            nei = gray[up:down, x]
            peaks, _ = find_peaks(nei)
            results_half = peak_widths(nei, peaks, rel_height=0.8)
            try:
                lw = max(results_half[0])
                lws.append(lw)
            except:
                pass
        return lws, np.median(lws)
    
    def __len__(self):
        return len(self.annot_list)
    
    def load(self, root=None):
        self.annot_list = glob.glob(os.path.join(root, "*.json"))
        
        
class SimuDatasimulator():
    def __init__(self, num_train = 400, num_val = 100):
        self.num_train = num_train
        self.num_val = num_val
        
    def Run(self, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        self.GenerateData(save_path, "train")
        self.GenerateData(save_path, "val")
        
    
    
    def GenerateData(self, save_path, mode="train"):
        img_dir = os.path.join(save_path, "leftImg8bit", mode,"simu")
        label_dir = os.path.join(save_path, "gtFine", mode, "simu")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        if mode == "train":
            ids = range(self.num_train)
        else:
            ids = range(self.num_val)
        
        for idx in ids:
            img, ins = self._plot()
            jpeg_quality = np.random.choice(self.jpeg_quality)
            img.save(os.path.join(img_dir, "%.6d.png"%idx), quality=jpeg_quality)
            ins.save(os.path.join(label_dir, "%.6d.png"%idx))
    
    def _plot(self, verbose=False):
        num_points_per_plot = np.random.choice(self.num_points_per_plot)
        x = np.linspace(-1,1,num_points_per_plot)
        num_plots = np.random.randint(1, self.max_plots)
        lw = np.random.choice(self.lw)
        dpi = np.random.choice(self.dpis)
#         jpeg_quality = np.random.choice(self.jpeg_quality)
        same_color = True
        default_color = "black" if np.random.rand() > 0.2 else np.random.choice(self.lcolor)
        if np.random.rand() > 0.7:
            same_color = False
        GaussianBlur = False
        AddNoise = False
        if np.random.rand() > 0.5:
            GaussianBlur = True
        if np.random.rand() > 0.5:
            AddNoise = True
        
        
        ys, cs = [], []
        
        y = self.func(x)
        for i in range(num_plots):
            if num_plots < 5:
                y = self.func(x)
            ys.append(y+i+np.random.rand())
            if same_color:
                cs.append(default_color)
            else:
                cs.append(np.random.choice(self.lcolor))
        
        assert len(ys) == len(cs), "{},{}".format(num_plots, len(ys), len(cs))

        
        plot_params = {
            "bottom_tick": True,
            "top_tick": True,
            "left_tick": True,
            "right_tick": True,
            "bottom_spine": True,
            "top_spine": True,
            "left_spine": True,
            "right_spine": True,
            "pad_inches": 0
        }
        for i in range(len(ys)):
            plt.plot(x,ys[i],lw=lw,c=cs[i])
        plt.xlim(-1, 1)
        if np.random.rand() > 0.5:
            plt.tick_params(
                axis='x',
                which='both',
                bottom=False,   
                top=False,
                labelbottom=False
            )
            plot_params["bottom_tick"]=False
            plot_params["top_tick"]=False
        else:
            plt.tick_params(
                axis='x',
                which='both',
                bottom=True,   
                top=True,
                labelbottom=False
            )
            
        if np.random.rand() > 0.5:
            plt.tick_params(
            axis='y',
            which='both',
            left=False,   
            right=False,
            labelleft=False
            )
            plot_params["left_tick"]=False
            plot_params["right_tick"]=False
        else:
            plt.tick_params(
                axis='y',
                which='both',
                left=True,   
                right=True,
                labelleft=False
            )
        ax = plt.gca()
        if np.random.rand() > 0.5:
            ax.spines['top'].set_visible(False)
            plot_params["top_spine"]=False
        if np.random.rand() > 0.5:
            ax.spines['left'].set_visible(False)
            plot_params["left_spine"]=False
        if np.random.rand() > 0.5:
            ax.spines['bottom'].set_visible(False)
            plot_params["bottom_spine"]=False
        if np.random.rand() > 0.5:
            ax.spines['right'].set_visible(False)
            plot_params["right_spine"]=False

        
        pad_inches = np.random.rand()*0.1
        plot_params["pad_inches"]=pad_inches
        plt.savefig("./base.png",dpi=dpi, bbox_inches="tight", 
                    pad_inches=plot_params["pad_inches"])
        graph = Image.open("./base.png").convert("RGB")
#         graph.save("./base.png", quality=jpeg_quality)
#         graph = Image.open("./base.png").convert("RGB")
        if AddNoise:
            graph_img = np.array(graph)
            noise = np.random.normal(0,0.1,graph_img.shape)
            graph = Image.fromarray(np.uint8(graph_img+noise))
        if GaussianBlur:
            blur = cv2.GaussianBlur(np.array(graph), (5,5), 0)
            graph = Image.fromarray(np.uint8(blur))
            
        plt.close()

        # Draw annotation
        width, height = graph.size
        x1 = 0
        y1 = 0
        num_annotation_word = 0
        num_annotation_line = 0
        draw = ImageDraw.Draw(graph)
        if np.random.rand() > 0.8:
            num_annotation_line = np.random.randint(1, self.num_annotation_line)
            for i in range(num_annotation_line):
                xx = np.random.randint(x1+width//10, x1+width-width//10)
                y_up = np.random.randint(y1, y1+height//2)
                y_down = np.random.randint(y1+height//2, y1+height)
                draw.line([xx,y_up,xx,y_down], fill=self.color_dict[default_color], width=lw)
                
        if np.random.rand() > 0.5:
            num_annotation_word = np.random.randint(1, self.num_annotation_word)
            for i in range(num_annotation_word):
                num_word = np.random.randint(1, 20)
                word = np.random.choice(self.char_list, size=num_word)
                word = "".join(word)
                xx = np.random.randint(width)
                yy = np.random.randint(height)
                color = np.random.choice(self.lcolor)
                font = ImageFont.truetype(np.random.choice(self.font_type),
                                         np.random.randint(5,20))
                draw.text(xy=(xx,yy), text=word, fill=self.color_dict[color], font=font)
    
    
    
        
        # generate_instance_mask
        instance_mask = []
        
        # axis
        for i in range(len(ys)):
            plt.plot(x,ys[i],lw=lw,c="white")
        plt.xlim(-1, 1)
        plt.tick_params(
            axis='x',
            which='both',
            bottom=plot_params["bottom_tick"],   
            top=plot_params["top_tick"],
            labelbottom=False
        )
        plt.tick_params(
            axis='y',
            which='both',
            left=plot_params["left_tick"],   
            right=plot_params["right_tick"],
            labelleft=False
        )
        ax = plt.gca()
        ax.spines['top'].set_visible(plot_params["top_spine"])
        ax.spines['bottom'].set_visible(plot_params["bottom_spine"])
        ax.spines['left'].set_visible(plot_params["left_spine"])
        ax.spines['right'].set_visible(plot_params["right_spine"])
        plt.savefig("./base.png",dpi=dpi, bbox_inches="tight", 
                    pad_inches=plot_params["pad_inches"])
        axis_only = Image.open("./base.png").convert("L")
        plt.close()
        instance_mask.append(axis_only)
        
        # each plot
        for plot_id in range(len(ys)):
            for i in range(len(ys)):
                if i == plot_id:
                    plt.plot(x,ys[i],lw=lw,c="black")
                else:
                    plt.plot(x,ys[i],lw=lw,c="white")
            plt.xlim(-1, 1)
            plt.tick_params(
                axis='x',
                which='both',
                bottom=plot_params["bottom_tick"],   
                top=plot_params["top_tick"],
                labelbottom=False
            )
            plt.tick_params(
                axis='y',
                which='both',
                left=plot_params["left_tick"],   
                right=plot_params["right_tick"],
                labelleft=False
            )
            ax = plt.gca()
            ax.spines['top'].set_visible(plot_params["top_spine"])
            ax.spines['bottom'].set_visible(plot_params["bottom_spine"])
            ax.spines['left'].set_visible(plot_params["left_spine"])
            ax.spines['right'].set_visible(plot_params["right_spine"])
            plt.savefig("./base.png",dpi=dpi, bbox_inches="tight", 
                        pad_inches=plot_params["pad_inches"])
            plot_mask = Image.open("./base.png").convert("L")
            plt.close()
            instance_mask.append(plot_mask)
            
        instance_mask_np = np.stack([np.array(instance_mask[0]) - np.array(mask) for mask in instance_mask[1:]])
        
        class_id = 26
        instance_map = np.zeros((instance_mask_np.shape[1], instance_mask_np.shape[2]))
        for i in range(instance_mask_np.shape[0]):
            instance_map[instance_mask_np[i]>0] = class_id*1000+i
        instance_map = Image.fromarray(instance_map).convert("I")

        
        if verbose:
            print("number of points per plot: {}".format(num_points_per_plot))
            print("number of plots: {}".format(num_plots))
            print("same color: {}".format(same_color))
            print("line width: {}".format(lw))
            print("dpi: {}".format(dpi))
            print("add noise: {}".format(AddNoise))
            print("gaussian blur: {}".format(GaussianBlur))
#             print("jpeg quality: {}".format(jpeg_quality))
            print("pad inches: {}".format(pad_inches))
            print("plot params: {}".format(plot_params.items()))
            print("annotation: {} lines, {} words".format(num_annotation_line, num_annotation_word))


           
        return graph, instance_map
        
        
    def __len__(self):
        return self.num_train+self.num_val
    
    def func(self, x):
        y = np.zeros(x.shape)
        N = np.random.randint(1, self.func_order)
        for i in range(N):
            y += np.random.normal(0,0.1)*x**i
        y = (y-min(y))/(max(y)-min(y)+1e-10)
        return y
    
    def _params_init(self, **kwargs):
        self.dpis = kwargs.get("dpis", [50,100])
        self.jpeg_quality = kwargs.get("jpeg_quality", [30,50,70,90])
        self.max_plots = kwargs.get("max_plots", 20)
        self.lw = kwargs.get("lw", [1,2,3,4])
        self.lcolor = kwargs.get("lcolor", ["r","g","b","black"])
        self.func_order = kwargs.get("func_order", 30)
        self.num_points_per_plot = kwargs.get("num_points_per_plot",[10,50,100])
        
        self.num_chars = kwargs.get("num_chars", [10,40])
        self.char_list = list(string.printable[:94])
        self.char_list.remove('$')
        self.num_annotation_line = kwargs.get("num_annotation_line", 4)
        self.num_annotation_word = kwargs.get("num_annotation_line", 10)
        self.font_type = glob.glob("/usr/share/fonts/truetype/freefont/*.ttf")
        self.color_dict = {
            "r": (255,0,0),
            "g": (0,255,0),
            "b": (0,0,255),
            "black": (0,0,0),
        }
        
        
class DATAsimulator():
    def __init__(self, root, save_path, **kwargs):
        self.simu_data_simulator = SimuDatasimulator()
        self.real_data_simulator = RealDatasimulator(root)
        self.type = "simu"
        self.save_path = save_path
        
    def _params_init(self, **kwargs):
        self.simu_data_simulator._params_init(**kwargs)
        
    def Run(self):
        if self.type == "simu":
            self.simu_data_simulator.Run(self.save_path)
            self.type = "real"
        else:
            self.real_data_simulator.Run(self.save_path)
            self.type = "simu"
            
            
            