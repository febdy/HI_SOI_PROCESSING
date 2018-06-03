import argparse
import logging
import time
from threading import Thread, Lock
from conn_pymongo import update_swk_result
import cv2

from pose_estimation.src.estimator import TfPoseEstimator
from pose_estimation.src.networks import get_graph_path, model_wh

from web_cam_video_stream import WebcamVideoStream

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


def do_pose_estimation(video_info):
    # if __name__ == "__main__":
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

    # cap = 'C:/Users/BIT-USER/Desktop/추요찡.mp4'
    # cap = 'C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN2.mp4'
    cap = video_info['videoPath']
    vs = WebcamVideoStream(src=cap).start()
    v_cap = cv2.VideoCapture(cap)

    # if (cap.start() == False):
    #     print("Error opening video stream or file")

    ret, image = vs.read()
    image = rotate(image, 90)
    s_pre_L_x, s_pre_L_y, s_pre_R_x, s_pre_R_y = 0, 0, 0, 0
    s_curr_R_x, s_curr_R_y, s_curr_L_x, s_curr_L_y = 0, 0, 0, 0
    w_pre_L_x, w_pre_L_y, w_pre_R_x, w_pre_R_y = 0, 0, 0, 0
    w_curr_R_x, w_curr_R_y, w_curr_L_x, w_curr_L_y = 0, 0, 0, 0
    k_pre_L_x, k_pre_L_y, k_pre_R_x, k_pre_R_y = 0, 0, 0, 0
    k_curr_R_x, k_curr_R_y, k_curr_L_x, k_curr_L_y = 0, 0, 0, 0
    humans = e.inference(image)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    # 어깨
    if 2 in humans[0].body_parts:
        s_pre_R_x = humans[0].body_parts[2].x
        s_pre_R_y = humans[0].body_parts[2].y

    if 5 in humans[0].body_parts:
        s_pre_L_x = humans[0].body_parts[5].x
        s_pre_L_y = humans[0].body_parts[5].y

    # 손목
    if 4 in humans[0].body_parts:
        w_pre_R_x = humans[0].body_parts[4].x
        w_pre_R_y = humans[0].body_parts[4].y

    if 7 in humans[0].body_parts:
        w_pre_L_x = humans[0].body_parts[7].x
        w_pre_L_y = humans[0].body_parts[7].y

    # 무릎
    if 9 in humans[0].body_parts:
        k_pre_R_x = humans[0].body_parts[9].x
        k_pre_R_y = humans[0].body_parts[9].y

    if 12 in humans[0].body_parts:
        k_pre_L_x = humans[0].body_parts[12].x
        k_pre_L_y = humans[0].body_parts[12].y

    print('처음좌표', s_pre_R_x, s_pre_R_y)
    e1 = cv2.getTickCount()
    i = True

    # frame_cnt = 0
    # print("pose_estimation_frame ::", v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    chk_move = 0
    frame_cnt = 0
    shoulder_move_cnt = 0
    wrist_move_cnt = 0
    knee_move_cnt = 0
    miss_location = []
    s_move_direction = [0, 0]  # 오른쪽 어깨/왼쪽 어깨
    w_move_direction = [0, 0]  # 오른쪽 손목/왼쪽 손목
    k_move_direction = [0, 0]  # 오른쪽 무릎/왼쪽 무릎

    while i:
        ret, image = vs.read()
        if not ret:
            video_info['shoulder_move_cnt'] = shoulder_move_cnt
            video_info['wrist_move_cnt'] = wrist_move_cnt
            video_info['knee_move_cnt'] = knee_move_cnt

            # video_info['cnt_per_5sec'] = cnt_per_5sec  # 5초 간격 cnt
            video_info['s_move_direction'] = s_move_direction
            video_info['w_move_direction'] = w_move_direction
            video_info['k_move_direction'] = k_move_direction

            update_swk_result(video_info)

            vs.stop()
            cv2.destroyAllWindows()

            e2 = cv2.getTickCount()
            print("correcting time :: ", (e2 - e1) / cv2.getTickFrequency())
            return 1
        else:
            # print("frame_cnt", frame_cnt)
            # frame_cnt += 1
            image = rotate(image, 90)
            humans = e.inference(image)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            # R 어깨
            if 2 in humans[0].body_parts:
                s_curr_R_x = humans[0].body_parts[2].x
                s_curr_R_y = humans[0].body_parts[2].y
            # L 어깨
            if 5 in humans[0].body_parts:
                s_curr_L_x = humans[0].body_parts[5].x
                s_curr_L_y = humans[0].body_parts[5].y
            # R 손목
            if 4 in humans[0].body_parts:
                w_curr_R_x = humans[0].body_parts[4].x
                w_curr_R_y = humans[0].body_parts[4].y
            # L 손목
            if 7 in humans[0].body_parts:
                w_curr_L_x = humans[0].body_parts[7].x
                w_curr_L_y = humans[0].body_parts[7].y
            # R 무릎
            if 9 in humans[0].body_parts:
                k_curr_R_x = humans[0].body_parts[9].x
                k_curr_R_y = humans[0].body_parts[9].y
            # L 무릎
            if 12 in humans[0].body_parts:
                k_curr_L_x = humans[0].body_parts[12].x
                k_curr_L_y = humans[0].body_parts[12].y

            # print('R어깨x', s_pre_R_x - s_curr_R_x)
            # print('R어깨y', s_pre_R_y - s_curr_R_y)
            # print('L어깨x', s_pre_L_x - s_curr_L_x)
            # print('L어깨y', s_pre_L_y - s_curr_L_y)
            # print('R손목x', w_pre_R_x - w_curr_R_x)
            # print('R손목y', w_pre_R_y - w_curr_R_y)
            # print('L손목x', w_pre_L_x - w_curr_L_x)
            # print('L손목y', w_pre_L_y - w_curr_L_y)
            # print('R무릎x', abs(k_pre_R_x - k_curr_R_x))
            # print('R무릎y', abs(k_pre_R_y - k_curr_R_y))
            # print('L무릎x', abs(k_pre_L_x - k_curr_L_x))
            # print('L무릎y', abs(k_pre_L_y - k_curr_L_y))

            # cv2.putText(image,
            #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
            #             (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (0, 255, 0), 2)
            # cv2.imshow('tf-pose-estimation result', image)

            # 어깨
            if abs(s_pre_R_x - s_curr_R_x) > 0 or abs(s_pre_R_y - s_pre_R_y) > 0:
                if chk_move == 0:
                    shoulder_move_cnt = shoulder_move_cnt + 1
                    chk_move = 1

                    if abs(s_pre_R_x - s_curr_R_x) > 0.02 or abs(s_pre_R_y - s_curr_R_y) > 0.03:  # 우
                        s_move_direction[0] += 1
                        print('오른어깨가 움직였습니다')
                    if abs(s_pre_L_x - s_curr_L_x) > 0.02 or abs(s_pre_L_y - s_curr_L_y) > 0.03:  # 좌
                        s_move_direction[1] += 1
                        print('왼어깨가 움직였습니다')

                    print('어깨', shoulder_move_cnt)

            # 손목
            if abs(w_pre_R_x - w_curr_R_x) > 0 or abs(w_pre_R_y - w_curr_R_y) > 0:
                if chk_move == 0:
                    wrist_move_cnt = wrist_move_cnt + 1
                    chk_move = 1

                    if abs(w_pre_R_x - w_curr_R_x) > 0.03 or abs(w_pre_R_y - w_curr_R_y) > 0.02:  # 우
                        w_move_direction[0] += 1
                        print('오른손이 움직였습니다')
                    if abs(w_pre_L_x - w_curr_L_x) > 0.01 or abs(w_pre_L_y - w_curr_L_y) > 0.02:  # 좌
                        w_move_direction[1] += 1
                        print('왼손이 움직였습니다')
                    print('손목', wrist_move_cnt)

            # 무릎
            if abs(k_pre_R_x - k_curr_R_x) > 0.03 or abs(k_pre_R_y - k_curr_R_y) > 0.03:
                if chk_move == 0:
                    knee_move_cnt = knee_move_cnt + 1
                    chk_move = 1

                    if abs(k_pre_R_x - k_curr_R_x) > 0.03 or abs(k_pre_R_y - k_curr_R_y) > 0.03:  # 우
                        k_move_direction[0] += 1
                        print('오른무릎이 움직였습니다')
                    if abs(k_pre_L_x - k_curr_L_x) > 0.03 or abs(k_pre_L_y - k_curr_L_y) > 0.03:  # 좌
                        k_move_direction[1] += 1
                        print('왼무릎이 움직였습니다')

                    print('무릎', knee_move_cnt)
            else:
                # if chk_move == 1:
                # end = int(ceil(frame_cnt / fps))
                # miss_location.append(end)
                # print("frame_cnt, end::::", frame_cnt, end)
                chk_move = 0

            # fps_time = time.time()
            # if cv2.waitKey(1) == 27:
            #     break
            # cap.stop()

    logger.debug('finished+ pose_estimation')

    return 0
