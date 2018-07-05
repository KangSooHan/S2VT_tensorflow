#-*- coding: utf-8 -*-
from pytube import YouTube
import pandas as pd
import numpy as np
import skvideo.io
import cv2
#import cv
import os
import pdb

def download_and_process_video(save_path, row):
#    pdb.set_trace()
    video_id = row['VideoID']
    video_path = row['video_path']
    full_path = os.path.join(save_path, video_path)
    if os.path.exists(full_path):
        return

    start = row['Start']
    end = row['End']

    print(video_id)

    if os.path.exists('tmp.mp4'):
        os.system('rm tmp.mp4')

    try:
        youtube = YouTube("https://www.youtube.com/watch?v="+video_id)
    except:
        print("Download failed...")
        return

    youtube.set_filename('tmp')
    
    try:
        video = youtube.get('mp4', '360p')
    except:
        pdb.set_trace()
    video.download('.')

    cap = cv2.VideoCapture('tmp.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.VideoWriter_fourcc(*'XVID')))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(full_path, fourcc, fps, (w,h))

    start_frame = int(fps * start)
    end_frame = int(fps * end)

    frame_count = 0
    while frame_count < end_frame:
        ret, frame = cap.read()
        frame_count += 1
        
        if frame_count >= start_frame:
            out.write(frame)

    cap.release()
    out.release()

def main():
    video_data_path = './data/msvd/video_corpus.csv'
    video_save_path = './data/youtube_videos'

    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language']=='English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi', axis=1)

    video_data.apply(lambda row: download_and_process_video(video_save_path, row), axis=1)

if __name__=="__main__":
    main()
