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


class SimuData():
    def __init__(self):
        self.now = datetime.now()
        self.count = 0
        
    
    def Save(self, **kwargs):
        dataset_type = kwargs.get("dataset_type", "cityscape")
        save_path = kwargs.get("save_path",None)
        mode = kwargs.get("mode", "train")
        num_figures = kwargs.get("num_figures", 1000)
        is_clean = kwargs.get("is_clean", False)
        jpeg_quality = np.random.choice(self.jpeg_quality)
        if mode == "val":
            num_figures = num_figures//4
        
        img_dir = os.path.join(save_path, "leftImg8bit", mode,"{}".format(num_figures))
        label_dir = os.path.join(save_path, "gtFine", mode, "{}".format(num_figures))
        if is_clean and os.path.exists(save_path):
            shutil.rmtree(save_path)
#             !rm -rf $save_path
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
#         print("generating {} simu data ...".format(mode))
        for i in tqdm(range(num_figures), desc ="{} simu data".format(mode)):
            img, ins = self._plot()
            if dataset_type == "cityscape":
                img.save(os.path.join(img_dir, "%.6d.png"%i), quality=jpeg_quality)
                ins.save(os.path.join(label_dir, "%.6d.png"%i))

            else:
                raise NotImplementedError
        
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
        self.font_type = glob.glob("/usr/share/fonts/truetype/*/*.ttf")
        self.color_dict = {
            "r": (255,0,0),
            "g": (0,255,0),
            "b": (0,0,255),
            "black": (0,0,0),
        }
        
    def func(self, x):
        y = np.zeros(x.shape)
        N = np.random.randint(1, self.func_order)
        for i in range(N):
            y += np.random.normal(0,0.1)*x**i
        y = (y-min(y))/(max(y)-min(y)+1e-10)
        return y
    
class RealData():
    def __init__(self):
        self.now = datetime.now()
        self.count = 0
        
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
    
    def _params_init(self, **kwargs):
        self.train_ratio = kwargs.get("train_ratio", 0.8)
        
    def Save(self, **kwargs):
        dataset_type = kwargs.get("dataset_type", "cityscape")
        root = kwargs.get("root", None)
        mode = kwargs.get("mode", "train")
        save_path = kwargs.get("save_path",None)
        is_clean = kwargs.get("is_clean", False)
#         assert mode == "train" and dataset_type == "cityscape"
        
        
        annot_list = glob.glob(os.path.join(root, "*.json"))
        np.random.shuffle(annot_list)
        
        class_id = 26
        total_num = len(annot_list)
        
        
        num_train = int(total_num*self.train_ratio)
        
        
        mode="train"
        img_dir = os.path.join(save_path, "leftImg8bit", mode, "{}".format(total_num))
        label_dir = os.path.join(save_path, "gtFine", mode, "{}".format(total_num))
        if is_clean and os.path.exists(save_path):
            shutil.rmtree(save_path)
#             !rm -rf $save_path
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

#         print("generating {} real data ...".format(mode))
        for idx in tqdm(range(num_train), desc ="{} real data".format(mode)):
            img_name = annot_list[idx].split("/")[-1][:-5]
            with open(annot_list[idx], "r") as fp:
                annot = json.load(fp)
            img_path = annot_list[idx].replace("json", "png")
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
            
        mode="val"
        img_dir = os.path.join(save_path, "leftImg8bit", mode, "{}".format(total_num))
        label_dir = os.path.join(save_path, "gtFine", mode, "{}".format(total_num))
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
#         print("generating {} real data ...".format(mode))
        for idx in tqdm(range(num_train, total_num), desc ="{} real data".format(mode)):
            img_name = annot_list[idx].split("/")[-1][:-5]
            with open(annot_list[idx], "r") as fp:
                annot = json.load(fp)
            img_path = annot_list[idx].replace("json", "png")
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
            
class DATAsimulator():
    def __init__(self):
        self.simu_data_engine = SimuData()
        self.real_data_engine = RealData()
        self.type = "simu"
        
    def _params_init(self, **kwargs):
        self.simu_data_engine._params_init(**kwargs)
        self.real_data_engine._params_init(**kwargs)
        self.dataset_type = kwargs.get("dataset_type", "cityscape")
        self.root = kwargs.get("root", None)
        self.mode = kwargs.get("mode", "train")
        self.save_path = kwargs.get("save_path",None)
        self.num_figures = kwargs.get("num_figures", 1000)
        self.root_test = kwargs.get("root_test", None)
        self.is_clean = kwargs.get("is_clean", False)
        
    def GenerateSimuData(self, mode="train", is_clean=False):
        kwargs = {
            "dataset_type": self.dataset_type,
            "mode": mode,
            "save_path": self.save_path,
            "num_figures": self.num_figures,
            "is_clean": is_clean,
        }
        self.simu_data_engine.Save(**kwargs)
        
    def GenerateRealData(self, is_clean=False):
        kwargs = {
            "dataset_type": self.dataset_type,
            "mode": self.mode,
            "save_path": self.save_path,
            "root": self.root,
            "is_clean": is_clean,
        }
        self.real_data_engine.Save(**kwargs)
        
    def GenerateTestData(self):
        dataset_type = self.dataset_type
        root_test = self.root_test
        save_path = self.save_path
        mode="test"
        
        img_list = glob.glob(os.path.join(root_test, "*.jpg"))
        class_id = 26
        total_num = len(img_list)
        
        img_dir = os.path.join(save_path, "leftImg8bit", mode, "{}".format(total_num))
        label_dir = os.path.join(save_path, "gtFine", mode, "{}".format(total_num))
        if self.is_clean and os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        for idx in tqdm(range(total_num), desc ="test data"):
            img_path = img_list[idx]
            img_name = img_list[idx].split("/")[-1][:-4]
            assert os.path.exists(img_path)
            img = Image.open(img_path).convert("RGB")
            instance_map = Image.new(mode="I", size=img.size)
            img.save(os.path.join(img_dir, "%s.png"%img_name))
            instance_map.save(os.path.join(label_dir, "%s.png"%img_name))
            
    def Run(self):
        if self.type == "simu":
            self.GenerateSimuData("train", True)
            self.GenerateSimuData("val", False)
            self.GenerateTestData()
            self.type = "real"
        else:
            self.GenerateRealData(True)
            self.GenerateTestData()
            self.type = "simu"
        
        