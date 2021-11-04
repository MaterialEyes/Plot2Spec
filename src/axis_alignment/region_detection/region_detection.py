from PIL import Image

from .detection import PlotDetector
from .refinement import refine

class RegionDetector():
    def __init__(self, config_file, checkpoint_file, device="cuda:0"):
        self.detector = PlotDetector()
        self.detector.load_model(config_file, checkpoint_file, device)
        
    def detect(self, img_path, refinement = True, **kwargs):
        result = self.detector.detection(img_path)
        try:
            bbox = result[0][0][:4]
        except:
            return None
        if not refinement:
            return bbox
        else:
            img = Image.open(img_path).convert("RGB")
            len_threshold = kwargs.get("len_threshold", 0.5)
            angle_threshold = kwargs.get("angle_threshold", 0.98)
            max_dist = kwargs.get("max_dist", 20)
            morphology = kwargs.get("morphology", True)
            refine_bbox = refine(img, bbox, len_threshold,
                                 angle_threshold, max_dist, morphology)
            return refine_bbox