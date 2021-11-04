from mmdet.apis import init_detector, inference_detector


"""
CNN-based detector trained on plot data
"""
class PlotDetector():
    def __init__(self):
        print("build plot detector ...")
        
    def load_model(self, config_file, checkpoint_file, device="cuda:0"):
        self.detector = init_detector(config_file, checkpoint_file, device=device)
        
    def detection(self, img_path):
        result = inference_detector(self.detector, img_path)
        return result