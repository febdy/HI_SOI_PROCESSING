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