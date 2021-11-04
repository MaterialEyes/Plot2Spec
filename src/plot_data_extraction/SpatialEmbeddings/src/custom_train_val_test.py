import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torchvision

import torch
import train_config, test_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
from simulator import DATAsimulator
from PIL import Image


data_simulator = DATAsimulator()
kwargs = {
    "dataset_type": "cityscape",
    "root": "../data/rp_graph/",
    "save_path": "../data/tmp/",
    "num_figures": 300,
    "root_test": "../../plot_digitizer/data/test_samples_crop/",
    "is_clean": False,
}
data_simulator._params_init(**kwargs)
data_simulator.Run()

torch.backends.cudnn.benchmark = True

# set device
device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")

# load config for training
train_args = train_config.get_args()
test_args = test_config.get_args()

# create path for result saving
if train_args['save']:
    if not os.path.exists(train_args['save_dir']):
        os.makedirs(train_args['save_dir'])
        
if test_args['save']:
    if not os.path.exists(test_args['save_dir']):
        os.makedirs(test_args['save_dir'])
        
# set model
model = get_model(train_args['model']['name'], train_args['model']['kwargs'])
model.init_output(train_args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)


# set criterion
criterion = SpatialEmbLoss(**train_args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=train_args['lr'], weight_decay=1e-4)
def lambda_(epoch):
    return pow((1-((epoch)/train_args['n_epochs'])), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_,)

# clustering
cluster = Cluster()

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')


# resume (initialization)
start_epoch = 0
best_iou = 0
if train_args['resume_path'] is not None and os.path.exists(train_args['resume_path']):
    print('Resuming model from {}'.format(train_args['resume_path']))
    state = torch.load(train_args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou = 0#state['best_iou']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']
    
def train(epoch):

    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']
        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()

        output = model(im)
        loss = criterion(output, instances, class_labels, **train_args['loss_w'])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss_meter.update(loss.item())

    return loss_meter.avg

def val(epoch):

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = model(im)
            loss = criterion(output, instances, class_labels, **
                            train_args['loss_w'], iou=True, iou_meter=iou_meter)
            loss = loss.mean()


            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg

def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(train_args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            train_args['save_dir'], 'best_iou_model_%s.pth'%(data_simulator.type)))
        
def test(epoch):
    
    model.eval()
    
    with torch.no_grad():
        plt.figure(figsize=(20,10*len(dataset_it)))
        for idx, sample in tqdm(enumerate(dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()

            output = model(im)
            instance_map, predictions = cluster.cluster(output[0], threshold=0.9)
            
            if test_args['save']:
                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                
                im_name = base + '.png'
                
                source_img = Image.open(os.path.join(test_args["dataset"]["kwargs"]["root_dir"], 
                                                     "leftImg8bit/test/34/", 
                                                     im_name)).convert("RGB")
                w,h = source_img.size
                if w > h:
                    nw = 512
                    nh = int(512/w*h)
                else:
                    nh = 512
                    nw = int(512/h*w)
                dw = 0
                dh = 0
                img_stack = [np.array(Image.new("L", size=(512,512)))]
                for id, pred in enumerate(predictions):
                    im = torchvision.transforms.ToPILImage()(
                        pred['mask'].unsqueeze(0)).convert("L")
                    img_stack.append(np.array(im))
                res_img = np.argmax(img_stack, axis=0)
                res_img = (res_img)/np.amax(res_img)*255
#                 plt.figure(fig_size=(10,10))
                plt.subplot(len(dataset_it),2,1+2*idx)
                plt.imshow(source_img)
                plt.axis("off")
                plt.subplot(len(dataset_it),2,2+2*idx)
                plt.imshow(res_img[0:nh,0:nw])
                plt.axis("off")
        plt.savefig(os.path.join(test_args['save_dir'] ,"%.4d.png"%(epoch)))
        plt.close()
        
# train dataloader
train_dataset = get_dataset(
    train_args['train_dataset']['name'], train_args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=train_args['train_dataset']['workers'], pin_memory=True if train_args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    train_args['val_dataset']['name'], train_args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=train_args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=train_args['train_dataset']['workers'], pin_memory=True if train_args['cuda'] else False)

# test dataloader
dataset = get_dataset(
    test_args['dataset']['name'], test_args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if test_args['cuda'] else False)


anchor_iou = 0
iou_history = []
for epoch in range(start_epoch, train_args['n_epochs']):
    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch)
    val_loss, val_iou = val(epoch)
    iou_history.append(val_iou)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))
    
    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=train_args['save'], save_dir=train_args['save_dir'])
    
    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)
        
    if train_args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best)
        
    
    if len(iou_history) > 10:
        if np.mean(iou_history) > anchor_iou:
            anchor_iou = np.mean(iou_history)
#             iou_history = []
        else:
            # re-generate training set
            # reload train/val dataloader
            # test on test-set
            anchor_iou = 0
            iou_history = []
            best_iou = 0
            
            data_simulator.Run()
            
            # train dataloader
            train_dataset = get_dataset(
                train_args['train_dataset']['name'], train_args['train_dataset']['kwargs'])
            train_dataset_it = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=train_args['train_dataset']['workers'], pin_memory=True if train_args['cuda'] else False)

            # val dataloader
            val_dataset = get_dataset(
                train_args['val_dataset']['name'], train_args['val_dataset']['kwargs'])
            val_dataset_it = torch.utils.data.DataLoader(
                val_dataset, batch_size=train_args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=train_args['train_dataset']['workers'], pin_memory=True if train_args['cuda'] else False)
            
            test(epoch)
