import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch
import signatory
import math
import os
from visualFER.load_landmark_data import get_ten_fold_data_list, get_ten_fold_data, FerLandmark
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SigExtract(nn.Module):
    def __init__(self, input_channel_num, truncated_degree=3):
        super(SigExtract, self).__init__()
        self.input_channel_num = input_channel_num
        self.truncated_degree = truncated_degree
        self.signature = signatory.Signature(depth=truncated_degree)  # [batch, stream, channels]
        self.sig_feature_num = math.floor(
            pow(self.input_channel_num, self.truncated_degree + 1) / (self.input_channel_num - 1)) - 2


    def forward(self, x_landmark):
        # print(x_landmark.shape)  # batch, seq_len=16, fea_num=2, points_num=52
        # x_landmark = x_landmark[:, :, :, :]
        # x_landmark += 32
        # print(x_landmark)
        for b in range(x_landmark.shape[0]):
            for v in range(x_landmark.shape[1]):
                if x_landmark[b, v, 0, 45] != 0:
                    x_landmark[b, v, 0, :] = x_landmark[b, v, 0, :] / x_landmark[b, v, 0, 38]  # 38 or 45
                    x_landmark[b, v, 1, :] = x_landmark[b, v, 1, :] / x_landmark[b, v, 1, 38]
        # print(x_landmark.shape)
        # x_landmark = torch.cat((torch.zeros([x_landmark.shape[0], 16, x_landmark.shape[2], x_landmark.shape[3]]), x_landmark), dim=1)
        print(x_landmark.shape)
        print(x_landmark[0, :, 0, :])

        sig_feature_list = []
        for p in range(x_landmark.shape[-1]):
            point_sig_feature = self.signature(x_landmark[:, :, :, p])
            sig_feature_list.append(point_sig_feature)
        x_landmark = torch.stack(sig_feature_list, dim=-1)
        print(x_landmark.shape)  # batch, sig_fea=14 or 30, point_num=64
        return x_landmark


if __name__ == "__main__":
    dataset = "mmi"
    ten_fold_root = r'../Ten_fold_png_data'
    # ten_fold_root = r'../MMI_test_png_data'
    ten_fold_png_root = os.path.join(ten_fold_root, dataset)
    ten_fold_sig_root = r'../Ten_fold_sig_feature' + '/' + dataset
    # ten_fold_sig_root = r'../MMI_test_sig_feature' + '/' + dataset
    print(ten_fold_sig_root)
    fold_landmark_root = ten_fold_png_root + 'landmark'
    data_list, landmark_list, label_list, sig = get_ten_fold_data_list(png_root=ten_fold_png_root,
                                                                       landmark_root=fold_landmark_root)
    sig_net = SigExtract(input_channel_num=2)
    for fold in range(10):
        fold_landmark = torch.from_numpy(landmark_list[fold])
        # print(fold_landmark.shape)
        fold_landmark = fold_landmark.to(dtype=torch.float)
        sig_feature = sig_net(fold_landmark)

        # print(sig_feature.shape)
        np.save(ten_fold_sig_root+'/'+str(fold)+'.npy', sig_feature)



