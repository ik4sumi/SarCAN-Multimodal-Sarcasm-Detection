import h5py
import numpy as np
import torch
import pickle
import pandas as pd
#import _Pickle
import json
import xlrd

xlsx="/home/srp/CSY/MUStARD-master/data/MUStARD_extension.xls"
extension_label='/home/srp/CSY/MUStARD-master/data/extension_label.json'
def parseXLSX(xlsx):
    label_dic={}
    data1=pd.read_excel(xlsx,keep_default_na=False)
    data = data1.loc[:, ['KEY', 'SPEAKER', 'SENTENCE', 'SHOW', 'SARCASM', 'SENTIMENT_IMPLICIT', 'SENTIMENT_EXPLICIT',
                         'EMOTION_IMPLICIT', 'EMOTION_EXPLICIT', 'MINTIME', 'MAXTIME', 'ALLTIME']]
    excel_keys = data['KEY']
    sentiment_implicit_labels = data['SENTIMENT_IMPLICIT']
    sentiment_explicit_labels = data['SENTIMENT_EXPLICIT']
    emotion_implicit_labels = data['EMOTION_IMPLICIT']
    emotion_explicit_labels = data['EMOTION_EXPLICIT']
    sarcasm_labels = data['SARCASM']
    for ID,key in enumerate(excel_keys):
        print(key)
        if key=='':
            print(ID)
            true_key=excel_keys[ID-1]
            label_dic[true_key]=[]

            if type(emotion_implicit_labels[ID - 1])==str:
                emotion_implicit_labels[ID - 1] = str(emotion_implicit_labels[ID - 1])
                emotion_implicit_labels[ID - 1]=emotion_implicit_labels[ID - 1].split(',')[0]
            if type(emotion_explicit_labels[ID - 1])==str:
                emotion_explicit_labels[ID - 1] = str(emotion_explicit_labels[ID - 1])
                emotion_explicit_labels[ID - 1] = emotion_explicit_labels[ID - 1].split(',')[0]


            label_dic[true_key].append(int(sentiment_implicit_labels[ID-1])+1)
            label_dic[true_key].append(int(sentiment_explicit_labels[ID - 1])+1)
            label_dic[true_key].append(int(emotion_implicit_labels[ID-1])-1)
            label_dic[true_key].append(int(emotion_explicit_labels[ID - 1])-1)
            #label_dic[true_key].append(int(emotion_implicit_labels[ID - 1]))
            #label_dic[true_key].append(int(emotion_explicit_labels[ID - 1]))
            print(label_dic[true_key])
        if key=='2_99':
            label_dic['2_99']=[]
            label_dic['2_99'].append(int(sentiment_implicit_labels[3639])+1)
            label_dic['2_99'].append(int(sentiment_explicit_labels[3639])+1)
            label_dic['2_99'].append(int(emotion_implicit_labels[3639])-1)
            label_dic['2_99'].append(int(emotion_explicit_labels[3639])-1)
    print(label_dic)
    #label_dic_json=json.dumps(label_dic)
    with open(extension_label,'w') as f:
        json.dump(label_dic,f)


if __name__ == "__main__":
    parseXLSX(xlsx)