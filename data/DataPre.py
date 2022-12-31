# coding: utf-8
import os
import argparse

import librosa
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
#from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2 as cv
#from transformers import Wav2Vec2PreTrainedModel,Wav2Vec2Model,Wav2Vec2FeatureExtractor
import pickle
import transformers as tf
import torch
import h5py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
final_pkl_file='/home/srp/CSY/MUStARD-master/data/features/audio_wav2vec.hdf5'
class dataPre():
    def __init__(self, working_dir):
        self.working_dir = working_dir

    def FetchFrames(self, input_dir, output_dir):
        """
        fetch frames from raw videos using ffmpeg toolkits
        """
        print("Start Fetch Frames...")
        video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*.mp4')))
        output_dir=os.path.join(self.working_dir,output_dir)
        #video_pathes=glob('E:\大四下\毕业设计\mmsd_raw_data\\utterances_final\*.mp4')
        #print(video_pathes)
        #print(output_dir)
        k=0
        for video_path in tqdm(video_pathes):
            #k+=1
            #if k>=3:
                #break
            video_id= video_path.split('/')[-1]
            video_id = video_id.split('.')[0]
            #video_id=video_id.split('c')[0]
            #print(video_id)
            #clip_id = '%04d' % (int(clip_id))
            #cur_output_dir = os.path.join(output_dir, video_id, clip_id)
            cur_output_dir = os.path.join(output_dir,video_id)
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            #print(video_path)
            video = cv.VideoCapture(video_path)  # 读取视频
            if video.isOpened() == False:
                print("error opening video stream or file!")
            i = 0
            while video.isOpened():
                success, frames = video.read()
                i += 1
                if success:
                    # time_stamp_fromvideo = video.get(cv.CAP_PROP_POS_MSEC)
                    #pic_path = path_save + '/' + files_name[:-4] + '/'  # 图片保存路径
                    # 存储为图像, 保存名为视频名称-第几帧.jpg
                    #print(os.path.join(cur_output_dir,str(i)+'.png'))
                    cv.imwrite(os.path.join(cur_output_dir,str(i)+'.png'), frames)
                    #print('save image:', i)
                    cv.waitKey(0)
                else:
                    break
            video.release()
            print('the current video' + cur_output_dir + ' is done')
            #cmd = 'ffmpeg -i ' + video_path + ' -r 30 ' + cur_output_dir + '/%04d.png -loglevel quiet'
            #os.system(cmd)
   
    def AlignFaces(self, input_dir,output_dir):
        """
        fetch faces from frames using MTCNN
        """
        print("Start Align Faces...")
        mtcnn = MTCNN(image_size=224, margin=0)

        video_pathes = sorted(glob(input_dir+"/*"))
        for video in video_pathes:
            video_id=os.path.split(video)[-1]
            list = sorted(glob(os.path.join(video,'*.jpg')))
            for frames_path in tqdm(list):
                output_path = os.path.join(output_dir,video_id)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                try:
                    #frame=os.path.split(frames_path)[-1].replace(".png",".jpg")
                    #img=cv.imread(frames_path)
                    #cv.imwrite(os.path.join(output_path,frame),img)
                    img = Image.open('F:\FFOutput\\1.jpg')
                    mtcnn(img, save_path=output_path)
                except Exception as e:
                    print(e)
                    continue

    def FetchAudios(self, input_dir, output_dir):
        """
        fetch audios from videos using ffmpeg toolkits
        """
        print("Start Fetch Audios...")
        video_pathes = sorted(glob(os.path.join(self.working_dir, input_dir, '*.mp4')))
        output_dir = os.path.join(self.working_dir, output_dir)
        for video_path in tqdm(video_pathes):
            output_path = video_path.replace(input_dir, output_dir).replace('.mp4', '.wav')
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            # 调用ffmpeg执行音频提取功能
            print(video_path)
            cmd = 'ffmpeg -i ' + video_path + ' -f wav -vn ' + \
                    output_path + ' -loglevel quiet'
            os.system(cmd)

    def faces(self,input_dir, output_dir):
        output_path = os.path.join(self.working_dir, output_dir)
        print(output_path)

        img = Image.open('/home/srp/LJL/CH-SIMS/Processed/video/Frame_1/video_0001-1.png')

        mtcnn = MTCNN(image_size=224, margin=0)
        mtcnn(img, save_path=output_path)

    def video_to_audio(self,video_path, audio_path):
        video_paths = sorted(glob(os.path.join(video_path, '*.mp4')))
        for vp in tqdm(video_paths[:]):
            output_path = vp.replace(video_path, audio_path).replace('.mp4', '.wav')
            print(output_path)
            cmd = 'ffmpeg -i ' + vp + ' -f wav -vn ' + output_path + ' -loglevel quiet'
            os.system(cmd)

    def classify_emotion(self,audio_path):
        feature_extractor=tf.Wav2Vec2Processor
        model=tf.Wav2Vec2Model
        feature_extractor=feature_extractor.from_pretrained("facebook/wav2vec2-large-960h")
        model=model.from_pretrained("facebook/wav2vec2-large-960h")
        model.to(device)
        #feature_extractor = Wav2Vec2Model.from_pretrained("superb/hubert-large-superb-er")
        audio_pathes=sorted(glob(os.path.join(audio_path, '*.wav')))
        audio_features_dic={}
        with h5py.File("/home/srp/CSY/MUStARD-master/data/features/audio_wav2vec_context.hdf5", "w") as f:
            for audio in audio_pathes:
                video_id=os.path.split(audio)[-1]
                video_id=video_id.split('.')[0]
                video_id=video_id.split('_c')[0]
                print(video_id)
                speech, _ = librosa.load(audio, sr=16000, mono=True)
                input_values = feature_extractor(speech, sampling_rate=16000, padding=True, return_tensors="pt").input_values
                input_values=input_values.to(device)
                feature=model(input_values).last_hidden_state
                feature=feature.squeeze(0).cpu().data.numpy().tolist()
                print(len(feature))
                f.create_dataset(video_id,shape=(len(feature),1024),data=feature)


            #with open(final_pkl_file, 'wb') as f:
                #pickle.dump(audio_features_dic, f, pickle.HIGHEST_PROTOCOL)
                #print(f'saved in {final_pkl_file}')

        return
    
#def parse_args():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--/home/srp/LJL/CH-SIMS', type=str,
                       #help='path to CH-SIMS')
    #return parser.parse_args()

if __name__ == "__main__":
    #args = parse_args()

    dp = dataPre('/home/srp/CSY/MUStARD_Data')


    # print(args.data_dir)

    # fetch frames from videos
    dp.FetchFrames('utterances_final', 'utterances_frames')

    # align faces
    #dp.AlignFaces('Processed/video/Frame_1', 'Processed/video/AlignedFaces_1')
    #dp.faces('Processed/video/Frame_1', 'Processed/video/AlignedFaces_1/1.png')
    # fetch audio
    #dp.FetchAudios('Raw_1', 'Processed/audio_1') # run in 3 down
    #dp.video_to_audio('E:\BYSJPROJECT\mmsd_raw_data\\context_final','E:\BYSJPROJECT\mmsd_raw_data\\context_audio')
    #dp.classify_emotion('/home/srp/CSY/MUStARD_Data/context_audio')
    #dp.AlignFaces('F:\\utterances_frames_jpg','F:\\utterances_faces')