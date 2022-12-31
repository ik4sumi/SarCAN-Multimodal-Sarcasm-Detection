import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import config
from prettytable import PrettyTable
from dataLoader import SarcasmData
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from Network import MultiHeadSelfAttention,CrossAttentionNetwork,MultiHeadCrossSelfAttention,Classify,\
    ContrastiveAttentionNetwork,MultiHeadSelfAttentionLayers,Classify_FC,Trimodal,Trimodal_CMD,Trimodal_Context
from torch.utils.tensorboard import SummaryWriter
from utils import CMD,DiffLoss

from tqdm import tqdm

config = config.Config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_loss_path = r'/home/srp/CSY/sig_face_project/check/best_loss.pth'
save_model_path = r'/home/srp/CSY/sig_face_project/check/best_loss.pth'
batch_size = 32
origin_lr = 0.0001
epochs = 600

criterion = nn.BCEWithLogitsLoss()

# net = SigResCnn1(ck=False).to(device=device)
#net=MultiHeadSelfAttention(18,283,80,80).to(device=device)
#net=CrossAttentionNetwork(18,96,283,768,80,80).to(device=device) #length_x,length_y,dim_in_x,dim_in_y,dim_k,dim_v
#net=Classify(96,768).to(device=device)
#net=ContrastiveAttentionNetwork(18,96,283,768,80,80).to(device=device)

def reduce_lr(epoch, lr):
    if epoch %1000 == 0:
        return lr/2
    else:
        return lr


def train_and_test(k,  lr, net, epochs, criterion,batch_size):
    best_acc = 0
    origin_lr = lr
    train_set = SarcasmData(config, 'train', k,False)
    test_set = SarcasmData(config, 'test', k,False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(epochs), desc='training|fold:'+str(k+1)):
        if epoch<100:
            lr=0.00001
        else:
            lr = reduce_lr(epoch, origin_lr)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        epoch_loss = 0
        corr = 0
        count = 0
        for data, label in train_loader:
            net.train()
            x = data.to(device=device, dtype=torch.float32)
            y = net(x)
            optimizer.zero_grad()
            loss = criterion(y, label.to(device=device).long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy().tolist()
            y = y.cpu().detach().numpy().tolist()
            out = []
            for n in range(len(y)):
                count += 1
                out.append(y[n].index(max(y[n])))
                if y[n].index(max(y[n])) == label.numpy().tolist()[n]:
                    corr += 1
            # acc_train = (corr/count)*100
        # print('train loss: ', epoch_loss/len(train_loader), '\t|\ttrain acc:%', acc_train, '\n')

        corr = 0
        count = 0
        # print('validating..')
        for data, label in test_loader:
            net.eval()
            with torch.no_grad():
                x = data.to(device=device, dtype=torch.float32)
                y = net(x)
                y = y.cpu().detach().numpy().tolist()
                out = []
                for n in range(len(y)):
                    out.append(y[n].index(max(y[n])))
                    count += 1
                    if y[n].index(max(y[n])) == label.numpy().tolist()[n]:
                        corr += 1
                # print(out)
        acc_test = (corr / count) * 100
        if acc_test > best_acc:
            best_acc = acc_test
    print('validation result: ', 'best acc:%', best_acc, '\n')
    time.sleep(0.1)
    return best_acc

def train_and_test_SI(k,  lr, net, epochs, criterion,batch_size):
    #writer = SummaryWriter('runs/fashion_mnist_experiment_1')  # 创建一个folder存储需要记录的数据
    writer = SummaryWriter('runs')  # 创建一个folder存储需要记录的数据
    best_acc = 0
    best_p=0
    best_r=0
    origin_lr = lr
    train_set = SarcasmData(config, 'train', k,True)
    test_set = SarcasmData(config, 'test', k,True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(epochs), desc='training|fold:'+str(k+1),position=0):
        origin_lr = reduce_lr(epoch, origin_lr)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        epoch_loss = 0
        corr_train = 0
        count_train = 0
        confusion_matrix = torch.zeros(2, 2)
        for data_1,data_2, label in train_loader:
            net.train()
            data_1 = data_1.to(device=device, dtype=torch.float32)
            data_2 = data_2.to(device=device, dtype=torch.float32)
            y = net(data_1,data_2)
            optimizer.zero_grad()
            #label_0用于统计
            label_0=label
            #one-hot编码
            label=torch.eye(2)[label, :]
            loss = criterion(y, label.to(device=device).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy().tolist()
            predict = y.argmax(dim=1)
            y = y.cpu().detach().numpy().tolist()
            out = []
            for t, p in zip(predict.view(-1), label_0.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]  # 列相加
            b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
            a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]  # 行相加
            b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
            for n in range(len(y)):
                count_train += 1
                out.append(y[n].index(max(y[n])))
                if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                    corr_train += 1
        acc_train = (corr_train/count_train)*100
        writer.add_scalar('train_ACC', acc_train, epoch)
        writer.add_scalar('train_Percision', b_p, epoch)
        writer.add_scalar('train_Recall', b_r, epoch)
        writer.add_scalar('train_LR', origin_lr, epoch)
    # print('train loss: ', epoch_loss/len(train_loader), '\t|\ttrain acc:%', acc_train, '\n')

        corr_test = 0
        count_test = 0
        # print('validating..')
        confusion_matrix = torch.zeros(2, 2)
        epoch_loss_test = 0
        for data_1, data_2,label in test_loader:
            net.eval()
            with torch.no_grad():
                data_1 = data_1.to(device=device, dtype=torch.float32)
                data_2=data_2.to(device=device,dtype=torch.float32)
                y = net(data_1,data_2)
                label_0=label
                label = torch.eye(2)[label, :]
                loss_test = criterion(y, label.to(device=device).float())
                epoch_loss_test+=loss_test.cpu().detach().numpy().tolist()
                out = []
                predict = y.argmax(dim=1)
                y = y.cpu().detach().numpy().tolist()
                for t, p in zip(predict.view(-1), label_0.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0] #列相加
                b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
                a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0] #行相加
                b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
                for n in range(len(y)):
                    #print(y[n])
                    #print(label)
                    out.append(y[n].index(max(y[n])))
                    count_test += 1
                    if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                        corr_test += 1
                #print(out)
        acc_test = (corr_test / count_test) * 100
        #print(corr)
        #print(count)
        #print(acc_test)
        #print(confusion_matrix.diag())
        #print(confusion_matrix[0][0])
        #print(confusion_matrix[0][1])
        #print(confusion_matrix[1][0])
        #print(confusion_matrix[1][1])
        #print(b_p)
        writer.add_scalar('Loss_train',epoch_loss,epoch)
        writer.add_scalar('Loss_test', epoch_loss_test, epoch)
        writer.add_scalar('ACC',acc_test,epoch)
        writer.add_scalar('Percision',b_p,epoch)
        writer.add_scalar('Recall',b_r,epoch)
        writer.add_scalar('LR', origin_lr, epoch)
        if acc_test > best_acc:
            best_acc = acc_test
            best_epoch = epoch
            best_0_0 = confusion_matrix[0][0].numpy()
            best_0_1 = confusion_matrix[0][1].numpy()
            best_1_0 = confusion_matrix[1][0].numpy()
            best_1_1 = confusion_matrix[1][1].numpy()
            test_num = best_0_0 + best_0_1 + best_1_1 + best_1_0
            best_weighted_p = b_p * ((best_0_1 + best_1_1) / test_num) + a_p * ((best_0_0 + best_1_0) / test_num)
            best_weighted_r = b_r * ((best_0_1 + best_1_1) / test_num) + a_r * ((best_0_0 + best_1_0) / test_num)
            best_weighted_f = (2 * best_weighted_p * best_weighted_r / (best_weighted_p + best_weighted_r))
        if epoch - best_epoch >= 50:
            pt = PrettyTable()
            pt.add_column("", ["Predict 0", "Predict 1"])
            pt.add_column("True 0", [best_0_0, best_1_0])
            pt.add_column("True 1", [best_0_1, best_1_1])
            print('\n', 'validation result: ', '\n', 'best acc:%', best_acc, '\n', 'Precision:', best_weighted_p, '\n',
                  'Recall:',
                  best_weighted_r, '\n', 'F-score:', best_weighted_f, '\n')
            print(pt)
            print("#" * 20)
            time.sleep(0.1)
            return best_acc
    # writer.add_graph(torch.ones(size=(18,283)), input_to_model=data, verbose=False)
    pt = PrettyTable()
    pt.add_column("", ["Predict 0", "Predict 1"])
    pt.add_column("True 0", [best_0_0, best_1_0])
    pt.add_column("True 1", [best_0_1, best_1_1])
    print('validation result: ', 'best acc:%', best_acc, '\n', 'Precision:', best_weighted_p, '\n', 'Recall:',
          best_weighted_r, '\n', 'F-score:', best_weighted_f, '\n')
    print(pt)
    print("#" * 20)
    time.sleep(0.1)
    # writer.add_graph(net,torch.rand(20, 18, 283))
    return best_acc

def train_and_test_SI_unimodal(k,  lr, net, epochs, criterion,batch_size):
    best_acc = 0
    best_epoch=0
    origin_lr = lr
    train_set = SarcasmData(config, 'train', k,False)
    test_set = SarcasmData(config, 'test', k,False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter('runs')  # 创建一个folder存储需要记录的数据
    for epoch in tqdm(range(epochs), desc='training:',position=0):
        if epoch<500:
            origin_lr=0.0001
        else:
            #origin_lr = reduce_lr(epoch, origin_lr)
            origin_lr=0.0001
        optimizer = torch.optim.Adam(net.parameters(), lr=origin_lr)
        epoch_loss = 0
        corr = 0
        count = 0
        confusion_matrix = torch.zeros(2, 2)
        #V,A,T
        for _,_,data, label,_,_,_,_,_ in train_loader:
            net.train()
            x = data.to(device=device, dtype=torch.float32)
            y= net(x)
            optimizer.zero_grad()
            #label_0用于统计
            label_0=label
            #one-hot编码
            label=torch.eye(2)[label, :]
            loss = criterion(y, label.to(device=device).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy().tolist()
            predict = y.argmax(dim=1)
            y = y.cpu().detach().numpy().tolist()
            out = []
            for t, p in zip(predict.view(-1), label_0.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]  # 列相加
            b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
            a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]  # 行相加
            b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
            for n in range(len(y)):
                count += 1
                out.append(y[n].index(max(y[n])))
                if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                    corr += 1
        acc_train = (corr/count)*100
        writer.add_scalar('train_ACC', acc_train, epoch)
        writer.add_scalar('train_Percision', b_p, epoch)
        writer.add_scalar('train_Recall', b_r, epoch)
        writer.add_scalar('train_LR', origin_lr, epoch)
        # print('train loss: ', epoch_loss/len(train_loader), '\t|\ttrain acc:%', acc_train, '\n')

        corr = 0
        count = 0
        epoch_loss_test=0
        confusion_matrix = torch.zeros(2, 2)
        # print('validating..')
        for _,_,data,label,_,_,_,_,_ in test_loader:
            net.eval()
            with torch.no_grad():
                x = data.to(device=device, dtype=torch.float32)
                y= net(x)
                label_0=label
                label = torch.eye(2)[label, :]
                loss_test = criterion(y, label.to(device=device).float())
                epoch_loss_test+=loss_test.cpu().detach().numpy().tolist()
                predict = y.argmax(dim=1)
                y = y.cpu().detach().numpy().tolist()
                out = []
                for t, p in zip(predict.view(-1), label_0.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0].numpy() #列相加
                b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].numpy()
                a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0].numpy() #行相加
                b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].numpy()
                for n in range(len(y)):
                    out.append(y[n].index(max(y[n])))
                    count += 1
                    if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                        corr += 1
                #print(out)
                #writer.add_histogram(x[0],'output')
        acc_test = (corr / count) * 100
        writer.add_scalar('Loss_train',epoch_loss,epoch)
        writer.add_scalar('Loss_test', epoch_loss_test, epoch)
        writer.add_scalar('ACC',acc_test,epoch)
        writer.add_scalar('Percision',b_p,epoch)
        writer.add_scalar('Recall',b_r,epoch)
        writer.add_scalar('LR', origin_lr, epoch)
        print(acc_test)
        if acc_test >= best_acc:
            best_acc = acc_test
            best_epoch=epoch
            best_0_0=confusion_matrix[0][0].numpy()
            best_0_1=confusion_matrix[0][1].numpy()
            best_1_0=confusion_matrix[1][0].numpy()
            best_1_1=confusion_matrix[1][1].numpy()
            test_num = best_0_0 + best_0_1 + best_1_1 + best_1_0
            best_weighted_p=b_p*((best_0_1+best_1_1)/test_num)+a_p*((best_0_0+best_1_0)/test_num)
            best_weighted_r=b_r*((best_0_1+best_1_1)/test_num)+a_r*((best_0_0+best_1_0)/test_num)
            best_weighted_f=(2*best_weighted_p*best_weighted_r/(best_weighted_p+best_weighted_r))
        if epoch-best_epoch>=20:
            pt = PrettyTable()
            pt.add_column("", ["Predict 0", "Predict 1"])
            pt.add_column("True 0", [best_0_0, best_1_0])
            pt.add_column("True 1", [best_0_1, best_1_1])
            print('\n','validation result: ','\n' ,'best acc:%', best_acc, '\n', 'Precision:', best_weighted_p, '\n', 'Recall:',
                  best_weighted_r, '\n', 'F-score:', best_weighted_f, '\n')
            print(pt)
            print("#" * 20)
            time.sleep(0.1)
            return best_acc,best_weighted_p,best_weighted_r,best_weighted_f
    #writer.add_graph(torch.ones(size=(18,283)), input_to_model=data, verbose=False)
    pt=PrettyTable()
    pt.add_column("", ["Predict 0", "Predict 1"])
    pt.add_column("True 0",[best_0_0,best_1_0])
    pt.add_column("True 1", [best_0_1,best_1_1])
    print('validation result: ', 'best acc:%', best_acc, '\n','Precision:',best_weighted_p,'\n','Recall:',best_weighted_r,'\n','F-score:',best_weighted_f,'\n')
    print(pt)
    print("#" * 20)
    time.sleep(0.1)
    #writer.add_graph(net,torch.rand(20, 18, 283))
    return best_acc


def train_and_test_SI_Trimodal(k,  lr, net, epochs, criterion,batch_size):
    best_acc = 0
    best_epoch=0
    origin_lr = lr
    train_set = SarcasmData(config, 'train', k,False)
    test_set = SarcasmData(config, 'test', k,False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter('runs')  # 创建一个folder存储需要记录的数据
    for epoch in tqdm(range(epochs), desc='training:',position=0):
        if epoch<500:
            origin_lr=0.0001
        else:
            #origin_lr = reduce_lr(epoch, origin_lr)
            origin_lr=0.0001
        optimizer = torch.optim.Adam(net.parameters(), lr=origin_lr)
        epoch_loss = 0
        corr = 0
        count = 0
        confusion_matrix = torch.zeros(2, 2)
        repr,y_true,dist,dist_att,ID_get=[],[],[],[],()
        for V,A,T, label,label_SI,label_SE,label_EI,label_EE,A_MFCC,ID in train_loader:
            net.train()
            V = V.to(device=device, dtype=torch.float32)
            A = A.to(device=device, dtype=torch.float32)
            T = T.to(device=device, dtype=torch.float32)
            A_MFCC = A_MFCC.to(device=device, dtype=torch.float32)
            #T_context = T_context.to(device=device, dtype=torch.float32)
            y,representation,SI,SE,EI,EE,dist_TA,dist_TV= net(T,V,A,A_MFCC)
            #repr.append(representation.cpu())
            #dist.append(dist_TV.cpu())
            #ID_get+=ID
            #y_true.append(label.squeeze().long().cpu())
            optimizer.zero_grad()
            #label_0用于统计
            label_0=label
            #one-hot编码
            label=torch.eye(2)[label, :]
            loss = criterion(y, label.to(device=device).float())
            criterion_extension=nn.CrossEntropyLoss()
            #loss+=0.5*criterion_extension(EE,label_EE.to(device=device).long())
            #loss+=0.5*criterion_extension(EI,label_EI.to(device=device).long())
            #loss+=0.5*criterion_extension(SE,label_SE.to(device=device).long())
            #loss+=0.5*criterion_extension(SI,label_SI.to(device=device).long())
            '''
            cmd=CMD()
            diff=DiffLoss()
            loss_cmd = cmd(net.shared_V,net.shared_A,5)
            loss_cmd += cmd(net.shared_V, net.shared_T, 5)
            loss_cmd += cmd(net.shared_T, net.shared_A, 5)
            loss_diff=diff(net.diff_V,net.diff_T)
            loss_diff+=diff(net.diff_A,net.diff_T)
            loss=loss+loss_cmd/3
            '''
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy().tolist()
            predict = y.argmax(dim=1)
            y = y.cpu().detach().numpy().tolist()
            out = []
            for t, p in zip(predict.view(-1), label_0.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]  # 列相加
            b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
            a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]  # 行相加
            b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
            for n in range(len(y)):
                count += 1
                out.append(y[n].index(max(y[n])))
                if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                    corr += 1
        acc_train = (corr/count)*100
        #reprs=torch.cat(repr)
        #true=torch.cat(y_true)
        #dist_att=torch.cat(dist)
        #print(dist_att[0,0,:])
        writer.add_scalar('train_ACC', acc_train, epoch)
        writer.add_scalar('train_Percision', b_p, epoch)
        writer.add_scalar('train_Recall', b_r, epoch)
        writer.add_scalar('train_LR', origin_lr, epoch)
        # print('train loss: ', epoch_loss/len(train_loader), '\t|\ttrain acc:%', acc_train, '\n')

        corr = 0
        count = 0
        epoch_loss_test=0
        confusion_matrix = torch.zeros(2, 2)
        # print('validating..')
        for V,A,T,label,label_SI,label_SE,label_EI,label_EE,A_MFCC,ID in test_loader:
            net.eval()
            with torch.no_grad():
                V = V.to(device=device, dtype=torch.float32)
                A = A.to(device=device, dtype=torch.float32)
                T = T.to(device=device, dtype=torch.float32)
                A_MFCC = A_MFCC.to(device=device, dtype=torch.float32)
                #T_context = T_context.to(device=device, dtype=torch.float32)
                y,representation,SI,SE,EI,EE,dist_TV,_= net(T,V,A,A_MFCC)
                repr.append(representation.cpu())
                y_true.append(label.squeeze().long().cpu())
                dist.append(dist_TV.cpu())
                ID_get += ID
                label_0=label
                label = torch.eye(2)[label, :]
                loss_test = criterion(y, label.to(device=device).float())
                epoch_loss_test+=loss_test.cpu().detach().numpy().tolist()
                predict = y.argmax(dim=1)
                y = y.cpu().detach().numpy().tolist()
                out = []
                for t, p in zip(predict.view(-1), label_0.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0].numpy() #列相加
                b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].numpy()
                a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0].numpy() #行相加
                b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].numpy()
                for n in range(len(y)):
                    out.append(y[n].index(max(y[n])))
                    count += 1
                    if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                        corr += 1
                #print(out)
                #writer.add_histogram(x[0],'output')
        acc_test = (corr / count) * 100
        reprs=torch.cat(repr)
        true=torch.cat(y_true)
        dist_att=torch.cat(dist)
        writer.add_scalar('Loss_train',epoch_loss,epoch)
        writer.add_scalar('Loss_test', epoch_loss_test, epoch)
        writer.add_scalar('ACC',acc_test,epoch)
        writer.add_scalar('Percision',b_p,epoch)
        writer.add_scalar('Recall',b_r,epoch)
        writer.add_scalar('LR', origin_lr, epoch)
        print(acc_test)
        if acc_test >= best_acc:
            best_acc = acc_test
            best_epoch=epoch
            best_0_0=confusion_matrix[0][0].numpy()
            best_0_1=confusion_matrix[0][1].numpy()
            best_1_0=confusion_matrix[1][0].numpy()
            best_1_1=confusion_matrix[1][1].numpy()
            test_num = best_0_0 + best_0_1 + best_1_1 + best_1_0
            a_f=2*a_p*a_r/(a_p+a_r)
            b_f=2*b_p*b_r/(b_p+b_r)
            best_weighted_p=b_p*((best_0_1+best_1_1)/test_num)+a_p*((best_0_0+best_1_0)/test_num)
            best_weighted_r=b_r*((best_0_1+best_1_1)/test_num)+a_r*((best_0_0+best_1_0)/test_num)
            #best_weighted_f=(2*best_weighted_p*best_weighted_r/(best_weighted_p+best_weighted_r))
            best_weighted_f = b_f * ((best_0_1 + best_1_1) / test_num) + a_f * ((best_0_0 + best_1_0) / test_num)
            tsne={'repr':reprs.cpu().detach().numpy(),'label':true.cpu().detach().numpy()}
            #dist={'dist':dist_att[ID_get.index('2_560')][0].cpu().detach().numpy()}
            path="/home/srp/CSY/MUStARD-master/TSNE/SD"+f'/tsne+{acc_test}.npz'
            #path_dist = "/home/srp/CSY/MUStARD-master/TSNE/SD" + f'/dist+{acc_test}.npz'
            np.savez(path, **tsne)
            #np.savez(path_dist,**dist)
        if epoch-best_epoch>=35:
            pt = PrettyTable()
            pt.add_column("", ["Predict 0", "Predict 1"])
            pt.add_column("True 0", [best_0_0, best_1_0])
            pt.add_column("True 1", [best_0_1, best_1_1])
            print('\n','validation result: ','\n' ,'best acc:%', best_acc, '\n', 'Precision:', best_weighted_p, '\n', 'Recall:',
                  best_weighted_r, '\n', 'F-score:', best_weighted_f, '\n')
            print(pt)
            print("#" * 20)
            time.sleep(0.1)
            return best_acc,best_weighted_p,best_weighted_r,best_weighted_f
    #writer.add_graph(torch.ones(size=(18,283)), input_to_model=data, verbose=False)
    pt=PrettyTable()
    pt.add_column("", ["Predict 0", "Predict 1"])
    pt.add_column("True 0",[best_0_0,best_1_0])
    pt.add_column("True 1", [best_0_1,best_1_1])
    print('validation result: ', 'best acc:%', best_acc, '\n','Precision:',best_weighted_p,'\n','Recall:',best_weighted_r,'\n','F-score:',best_weighted_f,'\n')
    print(pt)
    print("#" * 20)
    time.sleep(0.1)
    #writer.add_graph(net,torch.rand(20, 18, 283))
    return best_acc

def train_and_test_SI_Trimodal_context(k,  lr, net, epochs, criterion,batch_size):
    best_acc = 0
    best_epoch=0
    origin_lr = lr
    train_set = SarcasmData(config, 'train', k,False)
    test_set = SarcasmData(config, 'test', k,False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter('runs')  # 创建一个folder存储需要记录的数据
    for epoch in tqdm(range(epochs), desc='training:',position=0):
        if epoch<500:
            origin_lr=0.0001
        else:
            #origin_lr = reduce_lr(epoch, origin_lr)
            origin_lr=0.0001
        optimizer = torch.optim.Adam(net.parameters(), lr=origin_lr)
        epoch_loss = 0
        corr = 0
        count = 0
        confusion_matrix = torch.zeros(2, 2)
        for V,A,T,T_C ,V_C,A_C,label in train_loader:
            net.train()
            V = V.to(device=device, dtype=torch.float32)
            A = A.to(device=device, dtype=torch.float32)
            T = T.to(device=device, dtype=torch.float32)
            T_C = T_C.to(device=device, dtype=torch.float32)
            V_C = V_C.to(device=device, dtype=torch.float32)
            A_C=A_C.to(device=device,dtype=torch.float32)
            #T_context = T_context.to(device=device, dtype=torch.float32)
            y= net(T,V,A,T_C,V_C,A_C)
            optimizer.zero_grad()
            #label_0用于统计
            label_0=label
            #one-hot编码
            label=torch.eye(2)[label, :]
            loss = criterion(y, label.to(device=device).float())
            '''
            cmd=CMD()
            loss_cmd = cmd(net.T_shared,net.V_shared,5)
            loss_cmd += cmd(net.T_shared, net.A_shared, 5)
            loss_cmd += cmd(net.V_shared, net.A_shared, 5)
            loss_cmd=loss_cmd/3
            loss=loss+loss_cmd
            '''
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().numpy().tolist()
            predict = y.argmax(dim=1)
            y = y.cpu().detach().numpy().tolist()
            out = []
            for t, p in zip(predict.view(-1), label_0.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]  # 列相加
            b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
            a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]  # 行相加
            b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
            for n in range(len(y)):
                count += 1
                out.append(y[n].index(max(y[n])))
                if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                    corr += 1
        acc_train = (corr/count)*100
        writer.add_scalar('train_ACC', acc_train, epoch)
        writer.add_scalar('train_Percision', b_p, epoch)
        writer.add_scalar('train_Recall', b_r, epoch)
        writer.add_scalar('train_LR', origin_lr, epoch)
        # print('train loss: ', epoch_loss/len(train_loader), '\t|\ttrain acc:%', acc_train, '\n')

        corr = 0
        count = 0
        epoch_loss_test=0
        confusion_matrix = torch.zeros(2, 2)
        # print('validating..')
        for V,A,T,T_C,V_C,A_C,label in test_loader:
            net.eval()
            with torch.no_grad():
                V = V.to(device=device, dtype=torch.float32)
                A = A.to(device=device, dtype=torch.float32)
                T = T.to(device=device, dtype=torch.float32)
                T_C = T_C.to(device=device, dtype=torch.float32)
                V_C=V_C.to(device=device,dtype=torch.float32)
                A_C = A_C.to(device=device, dtype=torch.float32)
                y= net(T,V,A,T_C,V_C,A_C)
                label_0=label
                label = torch.eye(2)[label, :]
                loss_test = criterion(y, label.to(device=device).float())
                epoch_loss_test+=loss_test.cpu().detach().numpy().tolist()
                predict = y.argmax(dim=1)
                y = y.cpu().detach().numpy().tolist()
                out = []
                for t, p in zip(predict.view(-1), label_0.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0].numpy() #列相加
                b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].numpy()
                a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0].numpy() #行相加
                b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].numpy()
                for n in range(len(y)):
                    out.append(y[n].index(max(y[n])))
                    count += 1
                    if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                        corr += 1
                #print(out)
                #writer.add_histogram(x[0],'output')
        acc_test = (corr / count) * 100
        writer.add_scalar('Loss_train',epoch_loss,epoch)
        writer.add_scalar('Loss_test', epoch_loss_test, epoch)
        writer.add_scalar('ACC',acc_test,epoch)
        writer.add_scalar('Percision',b_p,epoch)
        writer.add_scalar('Recall',b_r,epoch)
        writer.add_scalar('LR', origin_lr, epoch)
        print(acc_test)
        if acc_test >= best_acc:
            best_acc = acc_test
            best_epoch=epoch
            best_0_0=confusion_matrix[0][0].numpy()
            best_0_1=confusion_matrix[0][1].numpy()
            best_1_0=confusion_matrix[1][0].numpy()
            best_1_1=confusion_matrix[1][1].numpy()
            test_num = best_0_0 + best_0_1 + best_1_1 + best_1_0
            a_f=2*a_p*a_r/(a_p+a_r)
            b_f=2*b_p*b_r/(b_p+b_r)
            best_weighted_p=b_p*((best_0_1+best_1_1)/test_num)+a_p*((best_0_0+best_1_0)/test_num)
            best_weighted_r=b_r*((best_0_1+best_1_1)/test_num)+a_r*((best_0_0+best_1_0)/test_num)
            #best_weighted_f=(2*best_weighted_p*best_weighted_r/(best_weighted_p+best_weighted_r))
            best_weighted_f = b_f * ((best_0_1 + best_1_1) / test_num) + a_f * ((best_0_0 + best_1_0) / test_num)
        if epoch-best_epoch>=20:
            pt = PrettyTable()
            pt.add_column("", ["Predict 0", "Predict 1"])
            pt.add_column("True 0", [best_0_0, best_1_0])
            pt.add_column("True 1", [best_0_1, best_1_1])
            print('\n','validation result: ','\n' ,'best acc:%', best_acc, '\n', 'Precision:', best_weighted_p, '\n', 'Recall:',
                  best_weighted_r, '\n', 'F-score:', best_weighted_f, '\n')
            print(pt)
            print("#" * 20)
            time.sleep(0.1)
            return best_acc
    #writer.add_graph(torch.ones(size=(18,283)), input_to_model=data, verbose=False)
    pt=PrettyTable()
    pt.add_column("", ["Predict 0", "Predict 1"])
    pt.add_column("True 0",[best_0_0,best_1_0])
    pt.add_column("True 1", [best_0_1,best_1_1])
    print('validation result: ', 'best acc:%', best_acc, '\n','Precision:',best_weighted_p,'\n','Recall:',best_weighted_r,'\n','F-score:',best_weighted_f,'\n')
    print(pt)
    print("#" * 20)
    time.sleep(0.1)
    #writer.add_graph(net,torch.rand(20, 18, 283))
    return best_acc



def test(k,  lr, net, epochs, criterion,batch_size):
    train_set = SarcasmData(config, 'train', k,True)
    test_set = SarcasmData(config, 'test', k,True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(epochs), desc='training:'):
        corr = 0
        count = 0
        epoch_loss_test=0
        confusion_matrix = torch.zeros(2, 2)
        for data,label in test_loader:
            net.eval()
            with torch.no_grad():
                x = data.to(device=device, dtype=torch.float32)
                y = net(x)
                label_0=label
                label = torch.eye(2)[label, :]
                loss_test = criterion(y, label.to(device=device).float())
                epoch_loss_test+=loss_test.cpu().detach().numpy().tolist()
                predict = y.argmax(dim=1)
                y = y.cpu().detach().numpy().tolist()
                out = []
                for t, p in zip(predict.view(-1), label_0.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                a_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[0] #列相加
                b_p = (confusion_matrix.diag() / confusion_matrix.sum(0))[1].float()
                a_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[0] #行相加
                b_r = (confusion_matrix.diag() / confusion_matrix.sum(1))[1].float()
                for n in range(len(y)):
                    out.append(y[n].index(max(y[n])))
                    count += 1
                    if y[n].index(max(y[n])) == label_0.numpy().tolist()[n]:
                        corr += 1
                #print(out)
        acc_test = (corr / count) * 100
        if acc_test > best_acc:
            best_acc = acc_test
            best_p=b_p
            best_r=b_r
    #writer.add_graph(torch.ones(size=(18,283)), input_to_model=data, verbose=False)
    print('validation result: ', 'best acc:%', best_acc, '\n','Precision:',best_p,'\n','Recall',best_r,'\n')
    time.sleep(0.1)
    #writer.add_graph(net,torch.rand(20, 18, 283))
    return best_acc

if __name__ == '__main__':
    best_acc = 0
    p_t=0
    r_t=0
    f_t=0
    for i in range(10):
        print("validate times:",i+1)
        #net = CrossAttentionNetwork(18, 96, 283, 768, 150, 150).to(device=device) #18 48
        #net = ContrastiveAttentionNetwork(18, 96, 283, 768, 80, 80).to(device=device)
        #net = MultiHeadSelfAttentionLayers(96, 768, 150, 150).to(device=device)
        #net=Classify(1000,1024).to(device=device)
        #net = Classify_FC(2048).to(device=device)
        net=Trimodal(96,480,1000,768,2048,1024,150,150).to(device=device)
        acc,p,r,f=train_and_test_SI_Trimodal(1, lr=origin_lr, net=net, epochs=epochs, criterion=criterion, batch_size=batch_size)
        if acc>best_acc:
            best_acc=acc
        p_t+=p
        r_t+=r
        f_t+=f
        print("当前第", i + 1,"次，最好验证结果：",best_acc)
    p_t=p_t/5
    r_t=r_t/5
    f_t=f_t/5
    print("average:")
    print("p:",p_t)
    print("r:",r_t)
    print("f:",f_t)
