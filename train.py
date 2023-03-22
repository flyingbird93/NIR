# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn
from torchtoolbox.transform import Cutout

from data_loader import VideoDataset_images_with_motion_features
from data_loader import VideoDataset_images_with_motion_features_test
from utils import performance_fit, performance_no_fit
from utils import L1RankLoss
from model import UGC_BVQA_model
from tqdm import tqdm

from torchvision import transforms
import time


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    model = UGC_BVQA_model.resnet50(pretrained=True)

    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()
    elif config.loss_type == 'mse':
        criterion = nn.MSELoss()

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    train_videos_dir = config.train_videos_frame_dir
    test_videos_dir = config.test_videos_frame_dir

    feature_dir = config.feature_dir

    datainfo_train = config.datainfo_train
    datainfo_val = config.datainfo_val

    datainfo_test_list = os.listdir(config.test_videos_dir)

    transformations_train = transforms.Compose([transforms.Resize(config.resize),
                                                transforms.RandomCrop(config.crop_size),
                                                transforms.RandomRotation(45, center=(45, 45)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                Cutout(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    transformations_test = transforms.Compose([transforms.Resize(config.resize),
                                               transforms.CenterCrop(config.crop_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    transformations_test_ten = transforms.Compose([transforms.Resize(config.resize),
                                                   transforms.TenCrop(config.crop_size),
                                                   transforms.Lambda(lambda crops: torch.stack(
                                                       [transforms.ToTensor()(crop) for crop in crops])),
                                                   transforms.Lambda(lambda crops: torch.stack(
                                                       [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in
                                                        crops]))])
        
    trainset = VideoDataset_images_with_motion_features(train_videos_dir, feature_dir+"/train", datainfo_train, transformations_train, 'VEC_train', config.crop_size, 'SlowFast', 'train')
    valset = VideoDataset_images_with_motion_features(train_videos_dir, feature_dir+"/train", datainfo_val, transformations_train, 'VEC_valid', config.crop_size, 'SlowFast', 'train')
    testset = VideoDataset_images_with_motion_features_test(test_videos_dir, feature_dir+"/test", datainfo_test_list, transformations_test, 'VEC_test', config.crop_size, 'SlowFast')

    # dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SROCC min
    best_test = []

    print('Starting training:')


    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()

        train_label = []
        train_output = []
        for i, (video, feature_3D, mos, _) in tqdm(enumerate(train_loader)):
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            labels = mos.to(device).float()

            outputs = model(video, feature_3D)

            train_label += list(labels[:].detach().cpu().numpy())
            train_output += list(outputs[:].detach().cpu().numpy())

            optimizer.zero_grad()

            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i+1) % (config.print_samples//config.train_batch_size) == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples//config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % \
                    (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, \
                        avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        train_PLCC, train_SRCC, train_KRCC, train_RMSE = performance_fit(train_label, train_output)
        print(
            'Epoch {} completed. The result on the train databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                epoch + 1, \
                train_PLCC, train_SRCC, train_KRCC, train_RMSE))

        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            label = np.zeros([len(valid_loader)])
            y_output = np.zeros([len(valid_loader)])
            for i, (video, feature_3D, mos, _) in tqdm(enumerate(valid_loader)):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label[i] = mos.item()
                outputs = model(video, feature_3D)

                y_output[i] = outputs.item()

            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)
            mean_metric = (test_SRCC + test_PLCC) / 2

            print('Epoch {} completed. The result on the val databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, '
                  'and RMSE: {:.4f}'.format(epoch + 1, test_SRCC, test_KRCC, test_PLCC, test_RMSE))

            if mean_metric > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = mean_metric
                # best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)

                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' + \
                    config.database + '_' + config.loss_type + '_NR_v'+ str(config.exp_version) \
                        + '_epoch_%d_SRCC_%f.pth' % (epoch + 1, mean_metric))
                torch.save(model.state_dict(), save_model_name)

            with torch.no_grad():
                test_name = []
                y_output = []
                for i, (video, feature_3D, name) in tqdm(enumerate(test_loader)):
                    video = video.to(device)                # 1, 8, 3, 448, 448
                    feature_3D = feature_3D.to(device)      # 1, 8, 2304
                    # label[i] = mos.item()
                    outputs = model(video, feature_3D)
                    # print(name)

                    test_name += list(name[:])
                    y_output += list(outputs[:].detach().cpu().numpy())

            with open(config.save_out_path, 'w') as f:
                for n in range(len(y_output)):
                    line = test_name[n] + ',' + str(abs(y_output[n])) + '\n'
                    f.write(line)

        # with torch.no_grad():
        #     test_label = []
        #     y_output = []
        #     for i, (video, feature_3D, mos, _) in tqdm(enumerate(test_loader)):
        #         video = video.to(device)
        #         frame_num, bs, nc, c, h, w = video.size()
        #         feature_3D = feature_3D.to(device)
        #         frame_num, bs, dim = feature_3D.size()
        #         feature_3D = feature_3D.expand(nc, bs, dim)
        #
        #         # label[i] = mos.item()
        #         outputs = model(video.view(-1, nc, c, h, w), feature_3D)
        #         outputs_avg = outputs.view(bs, nc, -1).mean(1)
        #
        #         test_label += list(mos[:].detach().cpu().numpy())
        #         y_output += list(outputs_avg[:].detach().cpu().numpy())

                # y_output[i] = outputs.item()
            
            # test_PLCC_1080p, test_SRCC_1080p, test_KRCC_1080p, test_RMSE_1080p = performance_fit(test_label, y_output)
            
            # print('Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            #     epoch + 1, test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p))

    print('Training completed.')
    print('The best training result on the train dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        train_PLCC, train_SRCC, train_KRCC, train_RMSE))
    print('The best test result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format( \
        best_test[0], best_test[1], best_test[2], best_test[3]))

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str, default='VEC')
    parser.add_argument('--model_name', type=str, default='SimpleVQA')
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default = 0)
    parser.add_argument('--results_path', type=str, default = '/home/vr/Work/Video_enhancement/SimpleVQA-main/results/')
    parser.add_argument('--exp_version', type=int, default=1)
    parser.add_argument('--print_samples', type=int, default = 1000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=520)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--epochs', type=int, default=30)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/home/vr/Work/Video_enhancement/SimpleVQA-main/ckpts/SimpleVQA_fold4_epoch_30')
    parser.add_argument('--save_out_path', type=str,
                        default='/home/vr/Work/Video_enhancement/SimpleVQA-main/ckpts/SimpleVQA_fold4_epoch_30/SimpleVQA_fold4_output.txt')
    parser.add_argument('--datainfo_train', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/data_split/train_fusion_split_1.txt')
    parser.add_argument('--datainfo_val', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/data_split/test_split_1.txt')

    parser.add_argument('--multi_gpu', action='store_true',  default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='mse') # default='L1RankLoss'
    parser.add_argument('--feature_dir', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/SlowFeature/')

    parser.add_argument('--train_videos_frame_dir', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/VideoFrame/train/')
    parser.add_argument('--test_videos_frame_dir', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/VideoFrame/test/')
    parser.add_argument('--test_videos_dir', type=str,
                        default='/media/vr/OS/Dell/VideoEnhancement/Data/test/')

    config = parser.parse_args()
    main(config)