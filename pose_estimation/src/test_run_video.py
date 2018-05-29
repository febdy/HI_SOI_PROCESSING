import argparse
import logging
import time
import copy

import cv2
import numpy as np
import pandas as pd
from collections import Counter

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 1)  # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)  # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 0)  # 뒤집기
    else:
        dst = None
    return dst

scaling_factor = 0.75

def run_video(video_info, queue):
    video = cv2.VideoCapture(video_info)
    cnt = 0

    while True:
        ret, frame = video.read()

        if ret:  # 영상 프레임이 있으면 실행
            if cnt % 4 == 0:
                frame = Rotate(frame, 90)
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                queue.put(frame)
#                queue2.put(frame)

            cnt = cnt + 1

        else:
            queue.put(None)
#            queue2.put(None)
            video.release()
            break


fps_time = 0

# def what():
#     def_R_x = max(list(curr_R_x), key=list(curr_R_x))
#     def_R_y = max(curr_R_y, key=curr_R_y.Counter)
#     def_L_x = max(curr_L_x, key=curr_L_x.Counter)
#     def_L_y = max(curr_L_y, key=curr_L_y.Counter)
#     print('(%.2f, %.2f),(%.2f, %.2f)' % (def_R_x, def_R_y, def_L_x, def_L_y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/HUN.mp4')

    queue = Queue()

    run_video('C:/Users/BIT-USER/Desktop/HUN.mp4',queue)

    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    ret_val, image = cap.read()
    image = Rotate(image, 90)
    pre_L_x, pre_L_y, pre_R_x, pre_R_y = 0, 0, 0, 0
    curr_R_x, curr_R_y, curr_L_x, curr_L_y = 0, 0, 0, 0
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    pre_R_x = humans[0].body_parts[2].x / 2
    pre_R_y = humans[0].body_parts[2].y
    pre_L_x = humans[0].body_parts[5].x/2
    pre_L_y = humans[0].body_parts[5].y



    while(cap.isOpened()):
        ret_val, image = cap.read()
        image = Rotate(image, 90)

        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        curr_R_x = humans[0].body_parts[2].x/2
        curr_R_y = humans[0].body_parts[2].y
        curr_L_x = humans[0].body_parts[5].x/2
        curr_L_y = humans[0].body_parts[5].y

        print(pre_R_x - curr_R_x)
        if ((pre_R_x - curr_R_x) > 0.01):
            print('정신차리세요')
        elif ((pre_R_x - curr_R_x) < 0):
            print('정신차리세요')
        else:
            print('올바른 자세입니다.')

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
