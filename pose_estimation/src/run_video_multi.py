from threading import Thread, Lock
import cv2
import argparse
import logging
import time

import cv2
import numpy as np
import pandas as pd
from collections import Counter

from pose_estimation.src.estimator import TfPoseEstimator
from pose_estimation.src.networks import get_graph_path, model_wh
import sys

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

ch.setFormatter(formatter)
logger.addHandler(ch)


def rotate(src, degrees):
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


fps_time = 0
# update_cnt = 0
read_cnt = 0


class WebcamVideoStream:
    def __init__(self, src='', width=320, height=240):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        # global update_cnt
        while self.started:
            self.read_lock.acquire()
            (grabbed, frame) = self.stream.read()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

            # update_cnt = update_cnt + 1
            # print("update_cnt", update_cnt)
            time.sleep(0.01)

    def read(self):
        global read_cnt
        self.read_lock.acquire()

        grabbed = self.grabbed
        frame = self.frame
        self.read_lock.release()
        read_cnt += 1
        print("read_cnt", read_cnt)
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


def do_pose_estimation(video_info, result_queue): #if __name__ == "__main__":
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

    # cap = 'C:/Users/BIT-USER/Desktop/HUN2.mp4'
    # cap = 'C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN2.mp4'
    cap = video_info['videoPath']
    vs = WebcamVideoStream(src=cap).start()
    v_cap = cv2.VideoCapture(cap)

    # if (cap.start() == False):
    #     print("Error opening video stream or file")

    ret, image = vs.read()
    image = rotate(image, 90)
    pre_L_x, pre_L_y, pre_R_x, pre_R_y = 0, 0, 0, 0
    curr_R_x, curr_R_y, curr_L_x, curr_L_y = 0, 0, 0, 0
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    pre_R_x = humans[0].body_parts[2].x / 2
    pre_R_y = humans[0].body_parts[2].y
    pre_L_x = humans[0].body_parts[5].x / 2
    pre_L_y = humans[0].body_parts[5].y

    # print('pre_R_x', pre_R_x)
    print('pre_R_y', pre_R_y)

    e1 = cv2.getTickCount()
    i = True

    # frame_cnt = 0
    print("pose_estimation_frame ::", v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while i:
        ret, image = vs.read()
        if not ret:
            break
        else:
            # print("frame_cnt", frame_cnt)
            # frame_cnt += 1
            image = rotate(image, 90)
            humans = e.inference(image)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            curr_R_x = humans[0].body_parts[2].x / 2
            curr_R_y = humans[0].body_parts[2].y
            curr_L_x = humans[0].body_parts[5].x / 2
            curr_L_y = humans[0].body_parts[5].y

            # print('curr_R_x', curr_R_x)
            # print('curr_R_y', curr_R_y)

            # print('x좌표 차이', pre_R_x - curr_R_x)
            # print('y좌표 차이', pre_R_y - curr_R_y)
            if ((pre_R_x - curr_R_x) >= 0.01 or (pre_R_y - curr_R_y) >= 0.02):
                print('정신차리세요')
            elif ((pre_R_x - curr_R_x) <= 0 ):
                print('정신차리세요')
            else:
                print('올바른 자세입니다.')

            # logger.debug('show+')
            # cv2.putText(image,
            #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
            #             (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)

            # cv2.imshow('tf-pose-estimation result', image)
            # if cv2.waitKey(1) == 27:
            #     break

            fps_time = time.time()

            # cap.stop()
    i = False
    vs.stop()
    cv2.destroyAllWindows()

    e2 = cv2.getTickCount()
    print("correcting time :: ", (e2 - e1) / cv2.getTickFrequency())

    result_queue.put(3)


logger.debug('finished+ pose_estimation')
