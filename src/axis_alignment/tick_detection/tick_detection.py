import cv2
import numpy as np
import time
import torch
from torch.autograd import Variable


from .craft import CRAFT
from .utils import copyStateDict
from .imgproc import loadImage, resize_aspect_ratio, normalizeMeanVariance, cvt2HeatmapImg
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .refinenet import RefineNet




class TickDetector():
    def __init__(self, **kwargs):
        self.canvas_size = kwargs.get("canvas_size", 1280)
        self.mag_ratio = kwargs.get("mag_ratio", 1.5)
        self.show_time = kwargs.get("show_time", False)
        self.text_threshold = kwargs.get("text_threshold", 0.7)
        self.link_threshold = kwargs.get("link_threshold", 0.7)
        self.low_text = kwargs.get("low_text", 0.4)
        self.poly = kwargs.get("poly", False)
        self.cuda = kwargs.get("cuda", True)
        
    
    def detect(self, image_path):
        image = loadImage(image_path)
        bboxes, polys, score_text = self.test_net(self.net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, self.refine_net)
        
        reformed_bboxes = []
        for bbox in bboxes:
            x1, x2 = min(bbox[:,0]), max(bbox[:,0])
            y1, y2 = min(bbox[:,1]), max(bbox[:,1])
            reformed_bboxes.append([x1, y1, x2, y2])
        
        return np.array(reformed_bboxes)
    
    
    def test_net(self, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)

        if self.show_time : 
            print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text
    
    
    def load_model(self, cuda=True, refine=True, **kwargs):
        trained_model = kwargs.get("trained_model", None)
        refiner_model = kwargs.get("refiner_model", None)
        
        # load net
        net = CRAFT()     # initialize

        # load trained mode
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(trained_model)))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
        else:
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
            
        net.eval()
            
        refine_net = None    
        if refine:
            refine_net = RefineNet()
            if cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
            refine_net.eval()
            self.poly = True
            
        self.net, self.refine_net = net, refine_net
            
        
            