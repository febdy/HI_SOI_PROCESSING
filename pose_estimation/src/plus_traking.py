import argparse
import logging
import time
import copy

import cv2
import numpy as np
import imutils
import pandas as pd
from collections import Counter

from pose_estimation.src.estimator import TfPoseEstimator
from pose_estimation.src.networks import get_graph_path, model_wh

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from img_video_ex.conn_pymongo import insert_test

conda_path = 'D:/javaForever/util/opencv/sources/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(conda_path + 'haarcascade_frontalface_default.xml')

model = load_model('face_ex.model')
scaling_factor = 0.75
kernel = np.ones((3, 3), np.uint8)

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

fps_time = 0

# def what():
#     def_R_x = max(list(curr_R_x), key=list(curr_R_x))
#     def_R_y = max(curr_R_y, key=curr_R_y.Counter)
#     def_L_x = max(curr_L_x, key=curr_L_x.Counter)
#     def_L_y = max(curr_L_y, key=curr_L_y.Counter)
#     print('(%.2f, %.2f),(%.2f, %.2f)' % (def_R_x, def_R_y, def_L_x, def_L_y))

# Tracker
def setup_tracker(ttype):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[ttype]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    return tracker

# 배열에 마스크 적용
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")

    output = np.zeros_like(array, dtype=np.uint8)

    for i, row in enumerate(imask):
        output[i, row] = array[i, row]

    return output

is_face = -1

cnt = 0
chk_move = 0

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
    cap = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/HUN2.mp4')
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

        if ret_val:
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            print('cnt: ', cnt)

            if is_face is -1:   # 얼굴이 잡히지 않았을 때
                for (x,y,w,h) in face_rects:
                    print('잡지못하고 있음')
                    img2 = cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)

                    img = cv2.resize(img2, (28,28))
                    img = img.astype("float") / 255.0
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)

                    (not_face, face) = model.predict(img)[0]

                    label = 'face' if face > not_face else 'Not face'

                    if label == 'face':
                        bg = image.copy()
                        bbox = (x,y,w,h)
                        def_x = x   # 움직임 계산할 때 기준이 되는 x, y
                        def_y = y
                        if_face = 1
                        tracker = setup_tracker(2)
                        tracking = tracker.init(image, bbox)

            elif is_face is 1: # 얼굴을 잡았을 때
                print('얼굴 잡는 중')
                diff = cv2.absdiff(bg, image)   # 기준 프레임과 다른점을 찾음
                mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                img_dilation = cv2.dilate(closing, kernel, iterations=2)
                imask = img_dilation > 0
                foreground = mask_array(image, imask)
                foreground_display = foreground.copy()

                tracking, bbox = tracker.update(foreground)
                tracking = int(tracking)

                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                if p1 == (0,0):
                    is_face = -1
                    continue
                cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
                cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)

                print('def_x : def_y', (def_x, def_y), 'p1_x : p1_y', p1)

                deteced_face = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0] + bbox[2])]
                detected_face = imutils.resize(detected_face, 3 * h, 3 * w)

                image[10: 10+detected_face.shape[1], 10: 10+detected_face.shape[0]] = detected_face

                if abs(def_x - p1[0]) > 4 or abs(def_y - p1[1]) > 4:
                    if chk_move == 0:
                        cnt = cnt + 1
                        chk_move = 1
                else:
                    chk_move = 0
        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        curr_R_x = humans[0].body_parts[2].x/2
        curr_R_y = humans[0].body_parts[2].y
        curr_L_x = humans[0].body_parts[5].x/2
        curr_L_y = humans[0].body_parts[5].y

        print(pre_R_x - curr_R_x)
        if ((pre_R_x - curr_R_x) > 0.0092):
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

######################################################################

# import argparse
# import logging
# import time
# import ast
#
# import common
# import cv2
# import numpy as np
# from estimator import TfPoseEstimator
# from networks import get_graph_path, model_wh
#
# from lifting.prob_model import Prob3dPose
# from lifting.draw import plot_pose
#
# logger = logging.getLogger('TfPoseEstimator')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
#
# fps_time = 0
#
# def Rotate(src, degrees):
#     if degrees == 90:
#         dst = cv2.transpose(src)  # 행렬 변경
#         dst = cv2.flip(dst, 1)  # 뒤집기
#
#     elif degrees == 180:
#         dst = cv2.flip(src, 0)  # 뒤집기
#
#     elif degrees == 270:
#         dst = cv2.transpose(src)  # 행렬 변경
#         dst = cv2.flip(dst, 0)  # 뒤집기
#     else:
#         dst = None
#     return dst
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='tf-pose-estimation run')
#     parser.add_argument('--image', type=str, default='./images/p1.jpg')
#     parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
#     parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
#     parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
#     args = parser.parse_args()
#     scales = ast.literal_eval(args.scales)
#
#     w, h = model_wh(args.resolution)
#     e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
#
#
#     cap = cv2.VideoCapture('HUN.mp4')
#
#     #ret_val, image = cap.read()
#     #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
#     if (cap.isOpened()== False):
#         print("Error opening video stream or file")
#     while(cap.isOpened()):
#         ret_val, image = cap.read()
#         image = Rotate(image, 90)
#
#         humans = e.inference(image)
#         image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
#
#         cur_R_x = humans[0].body_parts[2].x/2
#         cur_R_y = humans[0].body_parts[2].y
#         cur_L_x = humans[0].body_parts[5].x/2
#         cur_L_y = humans[0].body_parts[5].y
#
#         print('(%.2f, %.2f),(%.2f, %.2f)' % (cur_R_x, cur_R_y, cur_L_x, cur_L_y))
#
#
#
#         #logger.debug('show+')
#         cv2.putText(image,
#                     "FPS: %f" % (1.0 / (time.time() - fps_time)),
#                     (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (0, 255, 0), 2)
#         cv2.imshow('tf-pose-estimation result', image)
#         fps_time = time.time()
#         if cv2.waitKey(1) == 27:
#             break
#
#     t = time.time()
#     humans = e.inference(image, scales=scales)
#     elapsed = time.time() - t
#
#     logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
#
#     image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
#     # cv2.imshow('tf-pose-estimation result', image)
#     # cv2.waitKey()
#
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     a = fig.add_subplot(2, 2, 1)
#     a.set_title('Result')
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
#     bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
#
#     # show network output
#     a = fig.add_subplot(2, 2, 2)
#     plt.imshow(bgimg, alpha=0.5)
#     tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
#     plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
#     plt.colorbar()
#
#     tmp2 = e.pafMat.transpose((2, 0, 1))
#     tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
#     tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
#
#     a = fig.add_subplot(2, 2, 3)
#     a.set_title('Vectormap-x')
#     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
#     plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
#     plt.colorbar()
#
#     a = fig.add_subplot(2, 2, 4)
#     a.set_title('Vectormap-y')
#     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
#     plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
#     plt.colorbar()
#     plt.show()
#
#     import sys
#     sys.exit(0)
#
#     logger.info('3d lifting initialization.')
#     poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
#
#     image_h, image_w = image.shape[:2]
#     standard_w = 640
#     standard_h = 480
#
#     pose_2d_mpiis = []
#     visibilities = []
#     for human in humans:
#         pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
#         pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
#         visibilities.append(visibility)
#
#     pose_2d_mpiis = np.array(pose_2d_mpiis)
#     visibilities = np.array(visibilities)
#     transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
#     pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
#
#     for i, single_3d in enumerate(pose_3d):
#         plot_pose(single_3d)
#     plt.show()
#     pass