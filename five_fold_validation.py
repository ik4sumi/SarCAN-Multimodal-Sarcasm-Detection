from train import *
from Network0 import MultiHeadSelfAttention
import config
import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = config.Config
save_loss_path = r'/home/srp/CSY/sig_face_project/check/best_loss.pth'
save_model_path = r'/home/srp/CSY/sig_face_project/check/best_loss.pth'
batch_size = 32
origin_lr = 0.0001
epochs = 600
batch_size = 32
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
#net=MultiHeadSelfAttention(283,80,80).to(device=device)

def five_fold_validation(lr,net,epochs,criterion,batch_size):
    acc_sum = 0
    p_sum=0
    r_sum=0
    f_sum=0
    fold_acc = []
    for k in range(5):
        # net, lr, batch_size, k
        #net = Trimodal(96, 480, 1000, 768, 2048, 1024, 150, 150).to(device=device)
        net = MultiHeadSelfAttentionLayers(1000, 1024, 150, 150).to(device=device)
        best_acc_k,p,r,f = train_and_test_SI_unimodal(k, lr=lr,
                                    net=net, epochs=epochs, criterion=criterion,batch_size = batch_size)
        acc_sum += best_acc_k
        p_sum+=p
        r_sum+=r
        f_sum+=f
        fold_acc.append(best_acc_k)
    #print('----------------------------------\n', '\tFINAL ACCURACY: %' + str(acc_sum/5) + '\t',
          #'\n----------------------------------')
    print("acc:",acc_sum/5)
    print("p:",p_sum/5)
    print("r:",r_sum/5)
    print("f:",f_sum/5)
    return fold_acc, acc_sum/5


if __name__ == '__main__':
    #net = Trimodal(96, 480, 1000, 768, 2048, 1024, 150, 150).to(device=device)
    acc_sum = 0
    p_sum=0
    r_sum=0
    f_sum=0
    fold_acc = []
    for k in range(5):
        # net, lr, batch_size, k
        net = Trimodal(96, 480, 1000, 768, 2048, 1024, 150, 150).to(device=device)
        #net = MultiHeadSelfAttentionLayers(96, 768, 150, 150).to(device=device)
        best_acc_k,p,r,f = train_and_test_SI_Trimodal(k, lr=origin_lr,
                                    net=net, epochs=epochs, criterion=criterion,batch_size = batch_size)
        acc_sum += best_acc_k
        p_sum+=p
        r_sum+=r
        f_sum+=f
        fold_acc.append(best_acc_k)
    #print('----------------------------------\n', '\tFINAL ACCURACY: %' + str(acc_sum/5) + '\t',
          #'\n----------------------------------')
    print("acc:",acc_sum/5)
    print("p:",p_sum/5)
    print("r:",r_sum/5)
    print("f:",f_sum/5)
    #five_fold_validation(lr=origin_lr, net=net, epochs=epochs,criterion=criterion,batch_size=batch_size)