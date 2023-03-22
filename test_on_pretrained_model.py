import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn
from torchvision import transforms

from model import UGC_BVQA_model

from utils import performance_fit

from data_loader import VideoDataset_images_with_motion_features_test


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UGC_BVQA_model.resnet50(pretrained=False)

    model = model.to(device)

    # load the trained model
    print('loading the trained model')
    model_name = config.model_name
    model.load_state_dict(torch.load(os.path.join(config.trained_model, model_name)))

    datainfo_list = os.listdir(os.path.join(config.data_path, 'Data/test'))
    videos_dir = os.path.join(config.data_path, 'VideoFrame/test')
    feature_dir = os.path.join(config.data_path, 'SlowFeature/test')

    transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),\
        transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
  
    testset = VideoDataset_images_with_motion_features_test(videos_dir, feature_dir, datainfo_list, \
        transformations_test, config.database, 448, config.feature_type)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        # name = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        videos_name = []
        for i, (video, feature_3D, video_name) in tqdm(enumerate(test_loader)):
            # print(video_name[0])
            videos_name.append(video_name)
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            # label[i] = mos.item()
            outputs = model(video, feature_3D)

            y_output[i] = outputs.item()

        # sort
        # dataframe = pd.DataFrame({0:videos_name[0], 1:y_output})
        # dataframe.sort_values(by=0, ascending=True)
        # print(dataframe[0])
        save_file_path = 'results/output.txt'
        with open(save_file_path, 'w') as f:
            for m in range(len(videos_name)):
                lines = str(videos_name[m][0]) + ',' + str(round(y_output[m], 6)) + '\n'
                f.write(lines)
        # print('The result on the databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(\
        #     val_SRCC, val_KRCC, val_PLCC, val_RMSE))
        print('The result on the databaset has saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--train_database', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--num_workers', type=int, default=8)

    # misc
    parser.add_argument('--trained_model', type=str, default='/home/vr/Work/Video_enhancement/SimpleVQA-main/ckpts')
    parser.add_argument('--data_path', type=str, default='/media/vr/OS/Dell/VideoEnhancement/')
    parser.add_argument('--feature_type', type=str, default='SlowFast')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    parser.add_argument('--model_name', type=str, default='SimpleVQA_VEC_L1RankLoss_NR_v1_epoch_20_SRCC_8.562148.pth')

    config = parser.parse_args()

    main(config)



