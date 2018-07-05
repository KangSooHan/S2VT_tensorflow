import cv2
import os
import numpy as np
import pandas as pd
import skimage
from cnn_util_tensorflow import *
import pdb

def preprocess_frame(image, target_height=224, target_width=224):
#    pdb.set_trace()
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width*float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1]-cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height, int(height*float(target_width)/width)))
        cropping_length = int((resized_image.shape[0]-target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0]-cropping_length,:]
    return cv2.resize(resized_image, (target_height, target_width))


def main():
    num_frames = 80
    video_path = './data/youtube_videos'
    video_save_path = './rgb_feats'
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('.avi'), videos)


    input_data = tf.placeholder(tf.float32, shape=(num_frames, 224, 224, 3), name="X")
    feats = CNN(input_data)
    value = feats.fc7()

    init = tf.global_variables_initializer()

    #pdb.set_trace()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.variable_scope("vgg16") as scope:
            for idx, video in enumerate(videos):
                print(idx, video)
                if os.path.exists(os.path.join(video_save_path, video)):
                    print('Already processed...')
                    continue
                video_fullpath = os.path.join(video_path, video)
                try:
                    cap = cv2.VideoCapture(video_fullpath)
                except:
                    pass
                frame_count = 0
                frame_list = []
                while True:
                    ret, frame = cap.read()
                    if ret is False:
                        break
                    frame_list.append(frame)
                    frame_count += 1
                frame_list = np.array(frame_list)
                #pad_frame_list = np.zeros(frame_list.shape)

                #pdb.set_trace()
                #print(frame_count)
                if frame_count >= num_frames:
                    frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
                    frame_list = frame_list[frame_indices]
                elif frame_count != 0:
                    frame_list = np.pad(frame_list, ((0,num_frames-frame_count),(0,0),(0,0),(0,0)), 'constant')
                cropped_frame_list = np.array(list(map(lambda x: preprocess_frame(x), frame_list)))
                #print(cropped_frame_list.shape)

                if idx==0:
                    sess.run(init)
                    real_sess = feats.load_weights('vgg16_weights.npz', sess)

                if cropped_frame_list.shape[0] == 0 :
                    pass
                else:
                    Save = sess.run(value, feed_dict={input_data:cropped_frame_list[:]})

                    save_full_path = os.path.join(video_save_path, video + '.npy')
                    np.save(save_full_path, Save)

if __name__=="__main__":
    main()
