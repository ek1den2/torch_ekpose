import cv2
import argparse
import numpy as np
import torch

from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device
from lib.config import cfg
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
import ffmpeg

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016')
parser.add_argument('-c', '--ckpt_path', type=str, default='pose_model.pth')
parser.add_argument('-i', '--input_image', type=str, default='coco')
parser.add_argument('-o', '--output_image', type=str, default='result.png')
parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
args = parser.parse_args()

device = get_using_device(args.device)

model = get_outputs(args.model)     
model = load_ckpt(model, args.ckpt_path, device)
rotate_code = cv2.ROTATE_180
if __name__ == "__main__":
    video_path = input("Enter video path")
    video_capture_dummy = cv2.VideoCapture(video_path)
    ret,oriImg = video_capture_dummy.read()
    shape_tuple = tuple(oriImg.shape[1::-1])
    print("Shape of image is ",shape_tuple)
    rotate_code = check_rotation(video_path)
    video_capture_dummy.release()
    
    video_capture = cv2.VideoCapture(video_path)
    
    ##New stuff
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    vid_out = cv2.VideoWriter('output.avi', fourcc, 20.0, shape_tuple) 
    ###
    
    proc_frame_list = []
    oriImg_list = []
    while True:
        # Capture frame-by-frame
        try:
            ret, oriImg = video_capture.read()
            if rotate_code is not None:
                oriImg = correct_rotation(oriImg, rotate_code)
            oriImg_list.append(oriImg)
            
            cv2.imshow('Video', oriImg)
            
    #        vid_out.write(out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except :
            break
    video_capture.release()
    cv2.destroyAllWindows()

    print("Number of frames",len(oriImg_list))
    
    count = 0
    for oriImg in oriImg_list:
        count+=1
        if count%50 == 0:
            print(count, "frames processed")
        
        try:
            shape_dst = np.min(oriImg.shape[0:2])
            print(oriImg.shape)
        except:
            break
        with torch.no_grad():
                paf, heatmap, imscale = get_outputs(
                    oriImg, model, 'rtpose')
                      
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
                    
        out = draw_humans(oriImg, humans)
   
        vid_out.write(out)
    
    # When everything is done, release the capture
    
