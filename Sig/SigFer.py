import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch
import torch_geometric.nn as gnn
import math
import os
from visualFER.load_landmark_data import get_ten_fold_data_list, get_ten_fold_data, FerLandmark
from torch_geometric.data import Data as GnnData
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SigFer(nn.Module):
    def __init__(self, classes):
        super(SigFer, self).__init__()
        point_num = 64
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=point_num, out_channels=128, kernel_size=(1,), bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(1,), bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.cnn11 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=(1,), bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=14*4, out_channels=64, kernel_size=(1,), bias=True),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        att_size = 64
        self.ch_att1 = nn.Sequential(
            nn.Linear(14 * point_num, att_size),
            nn.BatchNorm1d(att_size),
            nn.ReLU(),
            nn.Linear(att_size, 14),
            nn.Sigmoid(),
        )
        self.ch_att2 = nn.Sequential(
            nn.Linear(14 * point_num, att_size),
            nn.BatchNorm1d(att_size),
            nn.ReLU(),
            nn.Linear(att_size, 14),
            nn.Sigmoid(),
        )
        self.ch_att3 = nn.Sequential(
            nn.Linear(14 * point_num, att_size),
            nn.BatchNorm1d(att_size),
            nn.ReLU(),
            nn.Linear(att_size, 14),
            nn.Sigmoid(),
        )
        self.ch_att4 = nn.Sequential(
            nn.Linear(14 * point_num, att_size),
            nn.BatchNorm1d(att_size),
            nn.ReLU(),
            nn.Linear(att_size, 14),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=64*256, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(in_features=64,  out_features=classes),
        )

    def forward(self, x_sig):
        # print(x_sig.shape)  # # batch, sig_fea=14, point_num=52
        att_in = torch.reshape(x_sig, (x_sig.shape[0], -1))
        att1, att2, att3, att4 = self.ch_att1(att_in), self.ch_att2(att_in), self.ch_att3(att_in), self.ch_att4(att_in),
        x = self.cnn1(x_sig.permute(0, 2, 1))
        x1, x2, x3, x4 = x * att1.unsqueeze(1), x * att2.unsqueeze(1), x * att3.unsqueeze(1), x * att4.unsqueeze(1),
        x = torch.cat((x1, x2, x3, x4), dim=2)
        x = self.cnn2(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        res = x
        x = self.cnn11(x)
        x = x + res
        x = self.relu(x)
        # print(x.shape)
        # print(x.shape)
        x = self.out(torch.reshape(x, (x.shape[0], -1)))
        return x


class GCAB(nn.Module):
    def __init__(self, point_feature_dim, out_feature_dim):
        super(GCAB, self).__init__()
        self.out_feature_dim = out_feature_dim
        self.mlp = nn.Linear(out_feature_dim, out_feature_dim)
        self.gat0 = gnn.GATConv(in_channels=point_feature_dim, out_channels=out_feature_dim) # , num_layers=layer, hidden_channels=64)
        self.gat1 = gnn.GATConv(in_channels=out_feature_dim, out_channels=out_feature_dim)
        self.gcn1 = gnn.GCNConv(in_channels=1, out_channels=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, graph_data):
        x = graph_data.x
        edge_index = graph_data.edge_index
        # edge_attr = graph_data.edge_attr
        h = self.gat0(x, edge_index)  # , edge_attr)  # [53, 64]
        res = h
        h = self.gat1(h, edge_index)
        h = h + res
        h = self.relu(h)
        mch = self.sigmoid(self.relu(self.mlp(torch.avg_pool1d(h.permute(1, 0), kernel_size=52).permute(1, 0))) +
                           self.relu(self.mlp(torch.max_pool1d(h.permute(1, 0), kernel_size=52).permute(1, 0))))
        h = h * mch  # [53, 64]*[1, 64]
        h_max = torch.max_pool1d(h, kernel_size=self.out_feature_dim)
        mno = self.sigmoid(self.gcn1(h_max, edge_index))
        h = h * mno  # [node_num=16, fea_num=out_feature_dim]
        # h = torch.sum(h, dim=0).unsqueeze(0)
        # print(h.shape)
        return h


class SigGnn(nn.Module):
    def __init__(self, classes):
        super(SigGnn, self).__init__()
        self.gcn0 = gnn.GATConv(in_channels=14, out_channels=64)
        self.gcn1 = gnn.GCNConv(in_channels=64, out_channels=128)
        self.gcab = GCAB(point_feature_dim=14, out_feature_dim=128)
        self.cnn1d1 = nn.Sequential(
            nn.Conv1d(in_channels=52, out_channels=52, kernel_size=(1,)),
            nn.BatchNorm1d(52),
            nn.ReLU(),
        )
        self.cnn1d2 = nn.Sequential(
            nn.Conv1d(in_channels=52, out_channels=52, kernel_size=(1,)),
            nn.BatchNorm1d(52),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(in_features=52  * 128, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=classes),
        )
        self.relu = nn.ReLU()

    def build_full_connected_edge_index(self, start_idx, end_idx):
        points_num = end_idx - start_idx
        start_list, end_list = [], []
        node_index_list = list(range(points_num))
        for p in range(points_num):
            start_list += [p for i in range(points_num)]
            end_list += node_index_list
        start_list = [s + start_idx for s in start_list]
        end_list = [s + start_idx for s in end_list]
        edge_attr = torch.from_numpy(np.ones(len(start_list))).to(dtype=torch.long, device=device)
        edge_index = torch.from_numpy(np.array([start_list, end_list])).to(dtype=torch.long, device=device)
        return edge_index, edge_attr

    def build_independent_edge_index(self, points_num):
        start_list = [i for i in range(points_num)]
        end_list = start_list
        return start_list, end_list

    def forward(self, x):
        # edge_index, edge_attr = self.build_full_connected_edge_index(0, 64)
        list_a, list_b = self.build_independent_edge_index(52)
        edge_index = torch.from_numpy(np.array([list_a, list_b])).to(device=device, dtype=torch.long)
        tensor_list = []
        for item in range(x.shape[0]):
            data_item = x[item, :, :].permute(1, 0)
            # print(data_item.shape, edge_index.shape)
            # print(edge_index)
            graph_data = GnnData(x=data_item, edge_index=edge_index)
            x_g = self.gcn0(graph_data.x, graph_data.edge_index)
            x_g = self.gcn1(x_g, graph_data.edge_index)
            x_g = self.relu(x_g)
            # x_g = self.gcab(graph_data)
            tensor_list.append(x_g)
        x = torch.stack(tensor_list, dim=0)
        x = self.cnn1d1(x)
        # x = self.cnn1d2(x) + x
        # print(x.shape)
        x = self.out(torch.reshape(x, [x.shape[0], -1]))
        return x


if __name__ == "__main__":
    dataset = "ck"
    ten_fold_root = r'../Ten_fold_png_data'
    ten_fold_png_root = os.path.join(ten_fold_root, dataset)
    fold_landmark_root = ten_fold_png_root + 'landmark'
    train_data_list, train_landmark_list, train_label_list, sig_list = \
        get_ten_fold_data_list(png_root=ten_fold_png_root, landmark_root=fold_landmark_root)
    train_shit, test_shit = get_ten_fold_data(test_fold=0, data_list=train_data_list,
                                              landmark_list=train_landmark_list,
                                              label_list=train_label_list, sig_list=sig_list)
    train_set = FerLandmark("train", train_shit, test_shit, sig=True)
    train_loader = data.DataLoader(dataset=train_set, batch_size=16, shuffle=False)
    classes = 7 if dataset == "ck" else 6
    sig_net = SigFer(classes)
    for label, sig in train_loader:
        y = sig_net(sig)
        print(y.shape)




