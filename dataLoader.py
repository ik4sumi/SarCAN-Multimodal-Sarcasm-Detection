from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import os
import sys
import re
import json
import jsonlines
import pickle
import config
import h5py
import gc

def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")

class SarcasmData(Dataset):

    DATA_PATH_JSON = "./data/sarcasm_data.json"
    AUDIO_PICKLE = "./data/audio_features.p"
    VIDEO_PICKLE='./data/features/i3d_avg_pool.hdf5'
    CONTEXT_VIDEO_PICKLE = './data/features/i3d_avg_pool_context.hdf5'
    #VIDEO_PICKLE = './data/features/resnet_pool5 .hdf5'
    AUDIO_WAV2VEC2='./data/features/audio_wav2vec.hdf5'
    CONTEXT_AUDIO_WAV2VEC2 = './data/features/audio_wav2vec_context.hdf5'
    INDICES_FILE = "./data/split_indices.p"
    GLOVE_DICT = "./data/glove_full_dict.p"
    BERT_TARGET_EMBEDDINGS = "./data/features/BERT_text_features/bert-output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = "./data/features/BERT_text_features/bert-output-context.jsonl"
    T_PADDING='./data/ALL_PICKLE/T_PADDING_ALL.p'
    V_PADDING = './data/ALL_PICKLE/V_PADDING_RESNET_ALL.p'
    A_PADDING = './data/ALL_PICKLE/A_PADDING_ALL.p'
    A_MFCC_PADDING='./data/ALL_PICKLE/A_MFCC_PADDING.p'
    T_C_PADDING='./data/ALL_PICKLE/T_C_PADDING_ALL.p'
    V_C_PADDING = './data/ALL_PICKLE/V_C_PADDING_ALL.p'
    A_C_PADDING = './data/ALL_PICKLE/A_C_PADDING_ALL.p'
    A_C_PADDING_SHORT = './data/ALL_PICKLE/A_C_PADDING_SHORT.p'
    T_C_PADDING_SHORT = './data/ALL_PICKLE/T_C_PADDING_SHORT.p'
    V_C_PADDING_SHORT = './data/ALL_PICKLE/V_C_PADDING_SHORT.p'
    ALL_PICKLE='./data/ALL_PICKLE/ALL_PICKLE.p'
    UTT_ID = 0
    CONTEXT_ID = 2
    SHOW_ID = 9
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    EXTENSION_JSON='/home/srp/CSY/MUStARD-master/data/extension_label.json'

    def __init__(self, config,usage,k,speakerIndependent=False):  # __init__是初始化该类的一些基础参数
        self.config = config
        self.dataset_json = json.load(open(self.DATA_PATH_JSON))
        self.extension_json=json.load(open(self.EXTENSION_JSON))
        self.fold_dict=pickle_loader("./data/split_indices.p")
        self.usage=usage
        self.k=k
        self.data_input_A_final,self.data_input_V_final,self.data_input_T_final,self.data_output_final,\
        self.data_input_T_context_final,self.data_input_V_context_final,self.data_input_A_context_final,\
        self.data_output_SI_final,self.data_output_SE_final,self.data_output_EI_final,self.data_output_EE_final,\
        self.data_input_A_MFCC,self.data_input_A_MFCC_final,self.data_output_ID=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        self.speakerIndependent=speakerIndependent

        '''
        #根据config进行设置
        if config.use_bert and config.use_target_text:
            self.text_bert_embeddings = []
            with jsonlines.open(self.BERT_TARGET_EMBEDDINGS) as reader:

                # Visit each target utterance
                for obj in reader:

                    CLS_TOKEN_INDEX = 0   #CLS表示整个句子的表征，在这里先不用
                    features = obj['features'][CLS_TOKEN_INDEX]
                    length=len(obj['features'])
                    bert_embedding_target_total=[]
                    for i in range(length-1):
                        features = obj['features'][i+1]
                        bert_embedding_target = []
                        for layer in [0, 1, 2, 3]:
                            bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
                        bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                        bert_embedding_target_total.append(np.copy(bert_embedding_target))
                    self.text_bert_embeddings.append(np.array(bert_embedding_target_total))
        else:
            self.text_bert_embeddings = None

        if config.use_bert and config.use_context_text:
            self.context_bert_embeddings = []
            with jsonlines.open(self.BERT_CONTEXT_EMBEDDINGS) as reader:

                # Visit each target utterance
                for obj in reader:

                    CLS_TOKEN_INDEX = 0   #CLS表示整个句子的表征，在这里先不用
                    features = obj['features'][CLS_TOKEN_INDEX]
                    length=len(obj['features'])
                    bert_embedding_target_total=[]
                    for i in range(length-1):
                        features = obj['features'][i+1]
                        bert_embedding_target = []
                        for layer in [0, 1, 2, 3]:
                            bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
                        bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                        bert_embedding_target_total.append(np.copy(bert_embedding_target))
                    self.context_bert_embeddings.append(np.array(bert_embedding_target_total))
        else:
            self.context_bert_embeddings = None

        #if config.use_context:
            #self.context_bert_embeddings = self.loadContextBert(self.dataset_json)
        #else:
            #self.context_bert_embeddings = None

        if config.use_target_audio:
            #self.audio_features = pickle_loader(self.AUDIO_PICKLE)
            self.audio_features=h5py.File(self.AUDIO_WAV2VEC2)
        else:
            self.audio_features = None

        if config.use_target_video:
            self.video_features=h5py.File(self.VIDEO_PICKLE,'r')
            self.context_video_features= None
        else:
            self.video_features=None
            self.context_video_features= None

        if config.use_context:
            self.video_features_context = h5py.File(self.CONTEXT_VIDEO_PICKLE, 'r')
            self.audio_features_context = h5py.File(self.CONTEXT_AUDIO_WAV2VEC2,'r')
        '''

        self.len=690
        #self.parseData(self.dataset_json,self.audio_features,self.video_features,
                       #self.text_bert_embeddings,self.context_bert_embeddings,self.video_features_context,self.audio_features_context)
        #self.data_input_T = np.array(self.data_input_T)
        #print(self.data_input[0][-1])
        #print(self.data_input_T[3].shape)
        self.parseInfo(self.dataset_json,self.extension_json)

        #根据用途划分
        self.data_input_T=pickle_loader(self.T_PADDING if self.config.use_target_text else None)
        self.data_input_V = pickle_loader(self.V_PADDING if self.config.use_target_video else None)
        self.data_input_A = pickle_loader(self.A_PADDING if self.config.use_target_audio else None)
        self.data_input_A_MFCC = pickle_loader(self.A_MFCC_PADDING if self.config.use_target_audio else None)
        if self.config.use_context:
            self.data_input_T_context=pickle_loader(self.T_C_PADDING)
            self.data_input_V_context=pickle_loader(self.V_C_PADDING)
            self.data_input_A_context=pickle_loader(self.A_C_PADDING)

        if speakerIndependent == False:
            if usage=='train':
                train_dict=self.fold_dict[k][0]
                for i in train_dict:
                    self.data_input_A_final.append(self.data_input_A[i])
                    self.data_input_A_MFCC_final.append(
                        self.data_input_A_MFCC[i] if self.config.use_target_audio else None)
                    self.data_input_V_final.append(self.data_input_V[i])
                    self.data_input_T_final.append(self.data_input_T[i])
                    if self.config.use_context:
                        self.data_input_T_context_final.append(self.data_input_T_context[i])
                        self.data_input_V_context_final.append(self.data_input_V_context[i])
                        self.data_input_A_context_final.append(self.data_input_A_context[i])
                    self.data_output_final.append(self.data_output[i])
                    self.data_output_SI_final.append(self.data_output_SI[i])
                    self.data_output_SE_final.append(self.data_output_SE[i])
                    self.data_output_EI_final.append(self.data_output_EI[i])
                    self.data_output_EE_final.append(self.data_output_EE[i])
                    self.data_output_ID.append(self.data_input[i][0])
                print(len(self.data_output_final))
            #del self.data_input_T,self.data_input_V,self.data_input_A
            #gc.collect()


            if usage=='test':
                train_dict=self.fold_dict[k][1]
                for i in train_dict:
                    self.data_input_A_final.append(self.data_input_A[i])
                    self.data_input_A_MFCC_final.append(self.data_input_A_MFCC[i] if self.config.use_target_audio else None)
                    self.data_input_V_final.append(self.data_input_V[i])
                    self.data_input_T_final.append(self.data_input_T[i])
                    if config.use_context:
                        self.data_input_T_context_final.append(self.data_input_T_context[i])
                        self.data_input_V_context_final.append(self.data_input_V_context[i])
                        self.data_input_A_context_final.append(self.data_input_A_context[i])
                    self.data_output_final.append(self.data_output[i])
                    self.data_output_SI_final.append(self.data_output_SI[i])
                    self.data_output_SE_final.append(self.data_output_SE[i])
                    self.data_output_EI_final.append(self.data_output_EI[i])
                    self.data_output_EE_final.append(self.data_output_EE[i])
                    self.data_output_ID.append(self.data_input[i][0])
                print(len(self.data_output_final))
                #del self.data_input_T, self.data_input_V, self.data_input_A
                #gc.collect()

        if speakerIndependent==True:
            if usage == 'train':
                for idx,ID in enumerate(self.data_input):
                    if ID[-1]!="FRIENDS":
                        self.data_input_A_final.append(self.data_input_A[idx] if self.config.use_target_audio else None)
                        self.data_input_A_MFCC_final.append(self.data_input_A_MFCC[idx] if self.config.use_target_audio else None)
                        self.data_input_V_final.append(self.data_input_V[idx] if self.config.use_target_video else None)
                        self.data_input_T_final.append(self.data_input_T[idx] if self.config.use_target_text else None)
                        if self.config.use_context:
                            self.data_input_T_context_final.append(self.data_input_T_context[idx])
                            self.data_input_V_context_final.append(self.data_input_V_context[idx])
                            self.data_input_A_context_final.append(self.data_input_A_context[idx])
                        self.data_output_final.append(self.data_output[idx])
                        self.data_output_SI_final.append(self.data_output_SI[idx])
                        self.data_output_SE_final.append(self.data_output_SE[idx])
                        self.data_output_EI_final.append(self.data_output_EI[idx])
                        self.data_output_EE_final.append(self.data_output_EE[idx])
                        self.data_output_ID.append(ID[0])
                del self.data_input_T, self.data_input_V, self.data_input_A
                gc.collect()

            if usage == 'test':
                for idx,ID in enumerate(self.data_input):
                    if ID[-1]=="FRIENDS":
                        self.data_input_A_final.append(self.data_input_A[idx] if self.config.use_target_audio else None)
                        self.data_input_A_MFCC_final.append(
                            self.data_input_A_MFCC[idx] if self.config.use_target_audio else None)
                        self.data_input_V_final.append(self.data_input_V[idx] if self.config.use_target_video else None)
                        self.data_input_T_final.append(self.data_input_T[idx] if self.config.use_target_text else None)
                        if self.config.use_context:
                            self.data_input_T_context_final.append(self.data_input_T_context[idx])
                            self.data_input_V_context_final.append(self.data_input_V_context[idx])
                            self.data_input_A_context_final.append(self.data_input_A_context[idx])
                        self.data_output_final.append(self.data_output[idx])
                        self.data_output_SI_final.append(self.data_output_SI[idx])
                        self.data_output_SE_final.append(self.data_output_SE[idx])
                        self.data_output_EI_final.append(self.data_output_EI[idx])
                        self.data_output_EE_final.append(self.data_output_EE[idx])
                        self.data_output_ID.append(ID[0])
                del self.data_input_T, self.data_input_V, self.data_input_A,
                gc.collect()

            print(usage,":",len(self.data_output_final))


    def __len__(self):  # 返回整个数据集的大小
        if self.speakerIndependent==False:
            if self.usage=='train': #552
                idx=0
                train_dict=self.fold_dict[self.k-1][0]
                for i in train_dict:
                    idx+=1

            if self.usage=='test':  #138
                idx=0
                train_dict=self.fold_dict[self.k-1][1]
                for i in train_dict:
                    idx+=1

        if self.speakerIndependent==True:
            if self.usage == 'train': #356
                idx=0
                for i,ID in enumerate(self.data_input):
                    if ID[-1]!="FRIENDS":
                        idx+=1

            if self.usage == 'test': #334
                idx=0
                for i,ID in enumerate(self.data_input):
                    if ID[-1]=="FRIENDS":
                        idx+=1


        return idx

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if self.config.use_target_video and self.config.use_target_audio and self.config.use_bert and self.config.use_target_text\
            and self.config.use_context_text==False:
            return self.data_input_V_final[index],self.data_input_A_final[index],self.data_input_T_final[index],\
                   self.data_output_final[index],self.data_output_SI_final[index],self.data_output_SE_final[index],\
                   self.data_output_EI_final[index],self.data_output_EE_final[index],self.data_input_A_MFCC_final[index],\
                   self.data_output_ID[index]  # 返回该样本

        if self.config.use_target_video and self.config.use_target_audio and self.config.use_bert and self.config.use_target_text\
                and self.config.use_context_text:
            return self.data_input_V_final[index],self.data_input_A_final[index],self.data_input_T_final[index],\
                   self.data_input_T_context_final[index],self.data_input_V_context_final[index],\
                   self.data_input_A_context_final[index],self.data_output_final[index]  # 返回该样本

        elif self.config.use_target_video and self.config.use_target_audio==False and self.config.use_target_text==False:
            return self.data_input_V_final[index],self.data_output_final[index]

        elif self.config.use_target_video==False and self.config.use_target_audio and self.config.use_target_text == False:
            return self.data_input_A_final[index], self.data_output_final[index]

        elif self.config.use_target_video==False and self.config.use_target_audio and self.config.use_target_text:
            return self.data_input_A_final[index],self.data_input_T_final[index],self.data_output_final[index]

        elif self.config.use_target_video and self.config.use_target_audio==False and self.config.use_target_text:
            return self.data_input_V_final[index],self.data_input_T_final[index],self.data_output_final[index]

        elif self.config.use_target_video and self.config.use_target_audio and self.config.use_target_text == False:
            return self.data_input_V_final[index],self.data_input_A_final[index],self.data_output_final[index]

        else:
            return self.data_input_T_final[index], self.data_output_final[index]

    def parseData(self, json, audio_features , video_features_file=None,
                  text_bert_embeddings=None, context_bert_embeddings=None,context_video_features_file=None,
                  context_audio_features=None):
        '''
        Prepares json data into lists
        data_input = [ (utterance:string, speaker:string, context:list_of_strings, context_speakers:list_of_strings, utterance_audio:features ) ]
        data_output = [ sarcasm_tag:int ]
        '''
        #print(video_features_file.keys())
        self.data_input, self.data_output,self.data_input_A,self.data_input_T,self.data_input_V,\
        self.data_input_T_context,self.data_input_V_context,self.data_input_A_context = [], [],[],[],[],[],[],[]
        for idx, ID in enumerate(json.keys()):
            if ID=='2_540':
                print()
            if ID!='2_540':
                '''
                self.data_input.append((json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                        json[ID]["context_speakers"], audio_features[ID] if self.audio_features else None,
                                        video_features_file[ID][()] if self.video_features else None,
                                        context_video_features_file[ID][()] if self.context_video_features else None,
                                        text_bert_embeddings[idx] if self.text_bert_embeddings else None,
                                        context_bert_embeddings[idx] if self.context_bert_embeddings else None,
                                        json[ID]["show"]))
                '''
                self.data_input.append((json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                        json[ID]["context_speakers"],
                                        json[ID]["show"]))
                self.data_output.append(int(json[ID]["sarcasm"]))
                #self.data_input_A.append(audio_features[ID] if self.audio_features else None)
                self.data_input_A.append(np.array(audio_features[ID]if self.audio_features else None).transpose(1,0))
                '''
                if self.video_features:
                    video_features=np.array(video_features_file[ID])
                    video_features_short = video_features[0, :]
                    video_features_short=np.expand_dims(video_features_short,axis=0)
                    #缩放10倍，即取1，11，21，。。。

                    if video_features.shape[0]<=10:
                        video_features_short=video_features_short
                    else:
                        if video_features.shape[0]%10==0:
                            for i in range(video_features.shape[0]//10-1):
                                video_features_file_append=np.expand_dims(video_features[(i+1)*10,:],axis=0)
                                video_features_short=np.concatenate((video_features_short,video_features_file_append),axis=0)
                        else:
                            for i in range(video_features.shape[0]//10):
                                video_features_file_append = np.expand_dims(video_features[(i+1) * 10, :], axis=0)
                                video_features_short=np.concatenate((video_features_short,video_features_file_append),axis=0)
                '''
                #print(video_features_short.shape)
                #original video features
                self.data_input_V.append(np.array(video_features_file[ID]).transpose(1,0) if self.video_features else None)
                #self.data_input_V.append(video_features_short.transpose(1,0) if self.video_features else None)
                self.data_input_T.append(text_bert_embeddings[idx].transpose(1,0) if self.text_bert_embeddings else None) #(690,768)
                self.data_input_T_context.append(
                    context_bert_embeddings[idx].transpose(1, 0) if self.context_bert_embeddings else None)
                self.data_input_V_context.append(
                    np.array(context_video_features_file[ID]).transpose(1, 0) if self.video_features_context else None)
                '''
                if self.audio_features_context:
                    video_features=np.array(context_audio_features[ID])
                    audio_features_short = video_features[0, :]
                    audio_features_short=np.expand_dims(audio_features_short,axis=0)
                    #缩放10倍，即取1，11，21，。。。

                    if video_features.shape[0]<=10:
                        audio_features_short=audio_features_short
                    else:
                        if video_features.shape[0]%10==0:
                            for i in range(video_features.shape[0]//10-1):
                                video_features_file_append=np.expand_dims(video_features[(i+1)*10,:],axis=0)
                                audio_features_short=np.concatenate((audio_features_short,video_features_file_append),axis=0)
                        else:
                            for i in range(video_features.shape[0]//10):
                                video_features_file_append = np.expand_dims(video_features[(i+1) * 10, :], axis=0)
                                audio_features_short=np.concatenate((audio_features_short,video_features_file_append),axis=0)
                '''
                self.data_input_A_context.append(np.array(context_audio_features[ID] if self.audio_features_context else None).transpose(1, 0))
                #self.data_input_A_context.append(audio_features_short.transpose(1, 0) if self.audio_features else None)
        #self.data_input_V.append(feature.transpose(1,0) for feature in self.data_input_V)
        #print([feature.shape[1] for feature in self.data_input_A])
        #print([feature.shape[0] for feature in self.data_input_V])
        #input shape (690,dims,length)
        #output shape (690,length,dims)
        del self.text_bert_embeddings,self.video_features,self.audio_features,\
            self.context_bert_embeddings,self.video_features_context,self.audio_features_context
        gc.collect()
        if self.config.use_target_audio:
            self.padAudio(self.data_input_A) #(690,283,18) 12
            #del self.data_input_A
            #gc.collect()
            #for i in range(len(self.data_input_A)):
                #self.data_input_A[i]=np.mean(self.data_input_A[i],axis=1)
                #self.data_input_A[i]=np.expand_dims(self.data_input_A[i],axis=1)
        if self.config.use_target_video:
            self.padVideo(self.data_input_V) #(690,1024,471) 195
            #for i in range(len(self.data_input_V)):
                #self.data_input_V[i]=np.mean(self.data_input_V[i],axis=1)
                #self.data_input_V[i] = np.expand_dims(self.data_input_V[i], axis=1)
        if self.config.use_target_text:
            self.padVideo(self.data_input_T) #(690,768,96)
            #for i in range(len(self.data_input_T)):
                #self.data_input_T[i]=np.mean(self.data_input_T[i],axis=1)
                #self.data_input_T[i] = np.expand_dims(self.data_input_T[i], axis=1)
        #print(self.data_input_A[0].shape)
        if self.config.use_target_text:
            #self.padVideo(self.data_input_T_context)
            for i in range(len(self.data_input_T_context)):
                self.data_input_T_context[i]=np.mean(self.data_input_T_context[i],axis=1)
                self.data_input_T_context[i]=np.expand_dims(self.data_input_T_context[i],axis=1)
        if self.config.use_context:
            #self.padVideo(self.data_input_V_context)
            for i in range(len(self.data_input_V_context)):
                self.data_input_V_context[i]=np.mean(self.data_input_V_context[i],axis=1)
                self.data_input_V_context[i]=np.expand_dims(self.data_input_V_context[i],axis=1)
            #self.padAudio(self.data_input_A_context)
            for i in range(len(self.data_input_A_context)):
                self.data_input_A_context[i]=np.mean(self.data_input_A_context[i],axis=1)
                self.data_input_A_context[i]=np.expand_dims(self.data_input_A_context[i],axis=1)

        '''''
        with open(self.A_PADDING, 'wb') as f:
            pickle.dump(self.data_input_A, f)
        with open(self.T_PADDING, 'wb') as f:
            pickle.dump(self.data_input_T, f)
        with open(self.V_PADDING, 'wb') as f:
            pickle.dump(self.data_input_V, f)
        

        with open(self.A_C_PADDING_SHORT, 'wb') as f:
            pickle.dump(self.data_input_A_context, f)

        with open(self.T_C_PADDING_SHORT, 'wb') as f:
            pickle.dump(self.data_input_T_context, f)
        with open(self.V_C_PADDING_SHORT, 'wb') as f:
            pickle.dump(self.data_input_V_context, f)
        '''

    def parseInfo(self, json,extension_json):
        '''
        Prepares json data into lists
        data_input = [ (utterance:string, speaker:string, context:list_of_strings, context_speakers:list_of_strings, utterance_audio:features ) ]
        data_output = [ sarcasm_tag:int ]
        '''
        #print(video_features_file.keys())
        self.data_input, self.data_output,self.data_output_SI,self.data_output_SE,self.data_output_EI,self.data_output_EE= [], [],[],[],[],[]
        for idx, ID in enumerate(json.keys()):
            '''
            self.data_input.append((json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                    json[ID]["context_speakers"], audio_features[ID] if self.audio_features else None,
                                    video_features_file[ID][()] if self.video_features else None,
                                    context_video_features_file[ID][()] if self.context_video_features else None,
                                    text_bert_embeddings[idx] if self.text_bert_embeddings else None,
                                    context_bert_embeddings[idx] if self.context_bert_embeddings else None,
                                    json[ID]["show"]))
            '''
            self.data_input.append((ID,json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                    json[ID]["context_speakers"],
                                    json[ID]["show"]))
            self.data_output.append(int(json[ID]["sarcasm"]))
            self.data_output_SI.append(extension_json[ID][0])
            self.data_output_SE.append(extension_json[ID][1])
            self.data_output_EI.append(extension_json[ID][2])
            self.data_output_EE.append(extension_json[ID][3])

            #self.data_input_A.append(audio_features[ID] if self.audio_features else None)


    #### Audio related functions ####

    def getAudioMaxLength(self, data):
        print(feature.shape[1] for feature in data)
        return np.max([feature.shape[1] for feature in data])

    def padAudio(self, data):
        max_length=self.getAudioMaxLength(data)
        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros( (instance.shape[0],(max_length-instance.shape[1])))], axis=1)
                data[ind] = instance
            data[ind] = data[ind][:,:max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)

    def getVideoMaxLength(self, data):
        return np.max([feature.shape[1] for feature in data])

    def padVideo(self, data):
        max_length=self.getVideoMaxLength(data)
        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros( (instance.shape[0],(max_length-instance.shape[1])))], axis=1)
                data[ind] = instance
            data[ind] = data[ind][:,:max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)

    def padSequence(self, data):
        max_length=self.getAudioMaxLength(data)
        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = torch.repeat_interleave(torch.from_numpy(instance),dim=1,repeats=max_length).cpu().numpy()
                data[ind] = instance
            data[ind] = data[ind][:,:max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)

    def __padding(self, feature, MAX_LEN):
        """
        mode:
            zero: padding with 0
            normal: padding with normal distribution
        location: front / back
        """
        feature=feature.transpose()
         #feature of shape: (length,dims)
        length = feature.shape[0]
        if length==0:
            length+=1
            copy=feature
            np.squeeze(copy,0)
            np.squeeze(feature, 0)
            np.expand_dims(feature, axis=0)
        else:
            copy = feature[length - 1,:]

        copy = np.expand_dims(copy, axis=0)
        #copy of shape: (1,dims)
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]
        else:
            for i in range(MAX_LEN-length):
                feature=np.concatenate((feature, copy), axis=0)

        return feature

    def __paddingSequence(self, sequences):
        lens = [s.shape[1] for s in sequences]
        # confirm length using (mean + std)
        # 因为Mustard数据集 长度相差很大，因此取均值+1 std填充。
        #print('mean , 1 * std : ',np.mean(lens) , 1 * np.std(lens))
        final_length = int(np.mean(lens) + 1 * np.std(lens))
        print("final length:",final_length)
        # padding sequences to final_length

        for i, s in enumerate(sequences):
            sequences[i] = self.__padding(s, final_length)

        return np.array(sequences)

if __name__ == "__main__":
    #print(len(pickle_loader("./data/split_indices.p")[0][0]))
    #print(len(pickle_loader("./data/split_indices.p")[0][1]))
    #f = h5py.File('./data/features/i3d_avg_pool.hdf5', 'r')
    #print(f.keys())
    config=config.Config
    dataset=SarcasmData(config,'test',1,True)
    print("A大小",dataset.data_input_A[0].shape)
    train_loader = DataLoader(dataset=dataset,
                               batch_size=32,
                               shuffle=False)
    for i,data in enumerate(train_loader):
        input_V,input_A,input_T,label=data
        print("第",i,"个input","输入",input_V.size(),input_A.size(),input_T.size(),"标签",label.size())
