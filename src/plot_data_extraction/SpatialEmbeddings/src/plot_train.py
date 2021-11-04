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
from PIL import Image

from DATAsimulator import DATAsimulator

class SpatialEmbedding():
    def __init__(self, root, save_path, **kwargs):
        torch.backends.cudnn.benchmark = True
        # init data generator
        self.data_simulator = DATAsimulator(root=root, save_path=save_path)
        self.data_simulator._params_init(**kwargs)
#         self.data_simulator.Run()
        
    
    def train(self, epoch, model, optimizer, criterion, train_dataset_it, train_args):

        # define meters
        loss_meter = AverageMeter()

        # put model into training mode
        model.train()

        for param_group in optimizer.param_groups:
            print('learning rate: {}'.format(param_group['lr']))

#         for i, sample in enumerate(train_dataset_it):
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
    
    def val(self, epoch, model, criterion, val_dataset_it, train_args):

        # define meters
        loss_meter, iou_meter = AverageMeter(), AverageMeter()

        # put model into eval mode
        model.eval()

        with torch.no_grad():

#             for i, sample in enumerate(val_dataset_it):
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
    
    
    
    
    def Train(self, first_time=False):
        
        data_simulator = self.data_simulator




        # load config for training
        train_args = train_config.get_args()

        # create path for result saving
        if train_args['save']:
            if not os.path.exists(train_args['save_dir']):
                os.makedirs(train_args['save_dir'])


        # set model
        device = torch.device("cuda:0" if torch.cuda.device_count()>0 else "cpu")
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
            model.load_state_dict(state['model_state_dict'], strict=True)
            if first_time:
                start_epoch = 0
                best_iou = 0 
                optimizer.load_state_dict(state['optim_state_dict'])
                logger.data = state['logger_data']
                

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


        anchor_iou = 0
        iou_history = []
        for epoch in range(start_epoch, train_args['n_epochs']):
            print('Starting epoch {}'.format(epoch))
            scheduler.step(epoch)

            train_loss = self.train(epoch, model, optimizer, criterion, train_dataset_it, train_args)
            val_loss, val_iou = self.val(epoch, model, criterion, val_dataset_it, train_args)
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
                self.save_checkpoint(epoch, state, is_best, val_iou, train_args, 'checkpoint_%.4d.pth'%epoch)
                # save latest checkpoint
                print("save latest checkpoint ... ")
                self.save_checkpoint(49, state, False, 0, train_args, 'checkpoint.pth')


            if len(iou_history) > 10:
                if np.mean(iou_history) > anchor_iou:
                    anchor_iou = np.mean(iou_history)
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


                    
    def save_checkpoint(self, epoch, state, is_best, val_iou, train_args, name='checkpoint.pth'):
        if (epoch+1)%50 == 0:
            print('=> saving checkpoint')
            file_name = os.path.join(train_args['save_dir'], name)
            torch.save(state, file_name)
        if is_best:
            torch.save(state, os.path.join(
                train_args['save_dir'], 'best_iou_model_%.2f.pth'%(val_iou)))
#             shutil.copyfile(file_name, os.path.join(
#                 train_args['save_dir'], 'best_iou_model_%.2f.pth'%(val_iou)))
   



if __name__ == "__main__":
    root = "/home/weixin/Documents/GitProjects/SpatialEmbeddings/data/real_ins_data/"
    save_path = "/home/weixin/Documents/GitProjects/SpatialEmbeddings/data/tmp/"
    engine = SpatialEmbedding(root, save_path)
    engine.Train(True)