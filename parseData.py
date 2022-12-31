from torch.utils.data import DataLoader, Dataset
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


class SarcasmData():
    DATA_PATH_JSON = "./data/sarcasm_data.json"
    AUDIO_PICKLE = "./data/audio_features.p"
    #VIDEO_PICKLE = './data/features/i3d_avg_pool_all.hdf5'
    CONTEXT_VIDEO_PICKLE = './data/features/i3d_avg_pool_context.hdf5'
    VIDEO_PICKLE = './data/features/resnet_pool5 .hdf5'
    AUDIO_WAV2VEC2 = './data/features/audio_wav2vec.hdf5'
    CONTEXT_AUDIO_WAV2VEC2 = './data/features/audio_wav2vec_context.hdf5'
    INDICES_FILE = "./data/split_indices.p"
    GLOVE_DICT = "./data/glove_full_dict.p"
    BERT_TARGET_EMBEDDINGS = "./data/features/BERT_text_features/bert-output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = "./data/features/BERT_text_features/bert-output-context.jsonl"
    T_PADDING = './data/ALL_PICKLE/T_PADDING_ALL.p'
    V_PADDING = './data/ALL_PICKLE/V_PADDING_RESNET_ALL.p'
    A_PADDING = './data/ALL_PICKLE/A_PADDING_ALL.p'
    T_C_PADDING = './data/ALL_PICKLE/T_C_PADDING_ALL.p'
    V_C_PADDING = './data/ALL_PICKLE/V_C_PADDING_ALL.p'
    A_C_PADDING = './data/ALL_PICKLE/A_C_PADDING_ALL.p'
    A_C_PADDING_SHORT = './data/ALL_PICKLE/A_C_PADDING_SHORT.p'
    T_C_PADDING_SHORT = './data/ALL_PICKLE/T_C_PADDING_SHORT.p'
    V_C_PADDING_SHORT = './data/ALL_PICKLE/V_C_PADDING_SHORT.p'
    A_MFCC_PADDING = './data/ALL_PICKLE/A_MFCC_PADDING.p'
    ALL_PICKLE = './data/ALL_PICKLE/ALL_PICKLE.p'
    UTT_ID = 0
    CONTEXT_ID = 2
    SHOW_ID = 9
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    EXTENSION_JSON = '/home/srp/CSY/MUStARD-master/data/extension_label.json'

    def __init__(self, config):  # __init__是初始化该类的一些基础参数
        self.config = config
        self.dataset_json = json.load(open(self.DATA_PATH_JSON))
        self.extension_json = json.load(open(self.EXTENSION_JSON))
        self.fold_dict = pickle_loader("./data/split_indices.p")
        self.data_input_A_final, self.data_input_V_final, self.data_input_T_final, self.data_output_final, \
        self.data_input_T_context_final, self.data_input_V_context_final, self.data_input_A_context_final, \
        self.data_output_SI_final, self.data_output_SE_final = [], [], [], [], [], [], [], [], []


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

        self.parseData(self.dataset_json,self.audio_features,self.video_features,self.text_bert_embeddings,
                       self.context_bert_embeddings,self.video_features_context,self.audio_features_context)



    def parseData(self, json, audio_features, video_features_file=None,
                  text_bert_embeddings=None, context_bert_embeddings=None, context_video_features_file=None,
                  context_audio_features=None):
        '''
        Prepares json data into lists
        data_input = [ (utterance:string, speaker:string, context:list_of_strings, context_speakers:list_of_strings, utterance_audio:features ) ]
        data_output = [ sarcasm_tag:int ]
        '''
        # print(video_features_file.keys())
        self.data_input, self.data_output, self.data_input_A, self.data_input_T, self.data_input_V, \
        self.data_input_T_context, self.data_input_V_context, self.data_input_A_context = [], [], [], [], [], [], [], []
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
            self.data_input.append((json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"],
                                    json[ID]["context_speakers"],
                                    json[ID]["show"]))
            self.data_output.append(int(json[ID]["sarcasm"]))
            #self.data_input_A.append(audio_features[ID] if self.audio_features else None)
            self.data_input_A.append(np.array(audio_features[ID] if self.audio_features else None).transpose(1, 0))
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
            # print(video_features_short.shape)
            # original video features
            self.data_input_V.append(
                np.array(video_features_file[ID]).transpose(1, 0) if self.video_features else None)
            # self.data_input_V.append(video_features_short.transpose(1,0) if self.video_features else None)
            self.data_input_T.append(
                text_bert_embeddings[idx].transpose(1, 0) if self.text_bert_embeddings else None)  # (690,768)

            self.data_input_T_context.append(
                context_bert_embeddings[idx].transpose(1, 0) if self.context_bert_embeddings else None)
            self.data_input_V_context.append(
                np.array(context_video_features_file[ID]).transpose(1, 0) if self.video_features_context else None)
            

            self.data_input_A_context.append(
                np.array(context_audio_features[ID] if self.audio_features_context else None).transpose(1, 0))

            # self.data_input_A_context.append(audio_features_short.transpose(1, 0) if self.audio_features else None)
        # self.data_input_V.append(feature.transpose(1,0) for feature in self.data_input_V)
        # print([feature.shape[1] for feature in self.data_input_A])
        # print([feature.shape[0] for feature in self.data_input_V])
        # input shape (690,dims,length)
        # output shape (690,length,dims)
        if self.config.use_target_audio:
            self.padAudio(self.data_input_A)  # (690,283,18) 12
            # del self.data_input_A
            # gc.collect()
            # for i in range(len(self.data_input_A)):
            # self.data_input_A[i]=np.mean(self.data_input_A[i],axis=1)
            # self.data_input_A[i]=np.expand_dims(self.data_input_A[i],axis=1)
        if self.config.use_target_video:
            self.padVideo(self.data_input_V)  # (690,1024,471) 195
            # for i in range(len(self.data_input_V)):
            # self.data_input_V[i]=np.mean(self.data_input_V[i],axis=1)
            # self.data_input_V[i] = np.expand_dims(self.data_input_V[i], axis=1)
        if self.config.use_target_text:
            self.padVideo(self.data_input_T)  # (690,768,96)
            # for i in range(len(self.data_input_T)):
            # self.data_input_T[i]=np.mean(self.data_input_T[i],axis=1)
            # self.data_input_T[i] = np.expand_dims(self.data_input_T[i], axis=1)
        # print(self.data_input_A[0].shape)
        if self.config.use_target_text:
            self.padVideo(self.data_input_T_context)

        if self.config.use_context:
            self.padVideo(self.data_input_V_context)

            self.padAudio(self.data_input_A_context)

        #''''
        #with open(self.A_PADDING, 'wb') as f:
            #pickle.dump(self.data_input_A, f)
        #with open(self.T_PADDING, 'wb') as f:
            #pickle.dump(self.data_input_T, f)
        with open(self.V_PADDING, 'wb') as f:
            pickle.dump(self.data_input_V, f)
        #'''


        #with open(self.A_C_PADDING, 'wb') as f:
            #pickle.dump(self.data_input_A_context, f)

        #with open(self.T_C_PADDING, 'wb') as f:
            #pickle.dump(self.data_input_T_context, f)
        #with open(self.V_C_PADDING, 'wb') as f:
            #pickle.dump(self.data_input_V_context, f)


    def getAudioMaxLength(self, data):
        print(feature.shape[1] for feature in data)
        return np.max([feature.shape[1] for feature in data])

    def padAudio(self, data):
        max_length = self.getAudioMaxLength(data)
        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros((instance.shape[0], (max_length - instance.shape[1])))],
                                          axis=1)
                data[ind] = instance
            data[ind] = data[ind][:, :max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)

    def getVideoMaxLength(self, data):
        return np.max([feature.shape[1] for feature in data])

    def padVideo(self, data):
        max_length = self.getVideoMaxLength(data)
        for ind, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros((instance.shape[0], (max_length - instance.shape[1])))],
                                          axis=1)
                data[ind] = instance
            data[ind] = data[ind][:, :max_length]
            data[ind] = data[ind].transpose()
        return np.array(data)


if __name__ == "__main__":
    config=config.Config
    a=SarcasmData(config)