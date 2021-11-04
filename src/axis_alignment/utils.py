import glob
import os
from PIL import Image
from tqdm import tqdm

from .region_detection.region_detection import RegionDetector
from .tick_detection.tick_detection import TickDetector
from .tick_recognition.tick_recognition import TickRecognition


class dict2class():
    def __init__(self, opt):
        for key in opt.keys():
            setattr(self, key, opt[key])
            
class AxisAlignment():
    def __init__(self, opt):
        self.opt = opt
        self.args = dict2class(opt)
        self.load_model()
        
    
    def crop(self, save_path):
        os.makedirs(save_path, exist_ok = True)
        for image_id in tqdm(range(len(self.imglist))):
            image_name = self.imglist[image_id].split("/")[-1]
            img = Image.open(self.imglist[image_id])
            plot_bbox = self.region_detector.detect(self.imglist[image_id])
            try:
                img_crop = img.crop(plot_bbox)
                img_crop.save(os.path.join(save_path, image_name))
            except:
                pass
            
    
    
    def run(self, image_id):
        img = Image.open(self.imglist[image_id])
        
        plot_bbox = self.region_detector.detect(self.imglist[image_id])
        text_bboxes = self.tick_detector.detect(self.imglist[image_id])
        
        self.tick_recognizer.load_data(img, plot_bbox, text_bboxes)
        results, results_all = self.tick_recognizer.detect()
        
        
        return img, plot_bbox, results, results_all
    
    
    def load_data(self, path):
        exts = ["jpg", "png", "jpeg"]
        imglist = []
        for ext in exts:
            imglist += sorted(glob.glob(os.path.join(path, "*."+ext)))
        self.imglist = imglist
    
    def load_model(self):
        args = self.args
        opt = self.opt
        region_detector = RegionDetector(args.config_file, 
                                         args.checkpoint_file)
        tick_detector = TickDetector(**opt)
        tick_detector.load_model(**opt)
        tick_recognizer = TickRecognition(args)
        tick_recognizer.load_model()
        self.tick_recognizer = tick_recognizer
        self.tick_detector = tick_detector
        self.region_detector = region_detector