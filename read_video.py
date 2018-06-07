# import the necessary packages
from conn_pymongo import update_grade_and_time
from face_pupil_processing import do_face_correction
from pose_estimation.src.run_video_multi import do_pose_estimation
from concurrent import futures
import cv2
import time


scaling_factor = 0.75


def rotate(src, degrees):  # 프레임 회전 (프레임, 회전할각도)
    if degrees == 90:
        dst = cv2.transpose(src) # 행렬 변경
        dst = cv2.flip(dst, 1)   # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)   # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src) # 행렬 변경
        dst = cv2.flip(dst, 0)   # 뒤집기
    else:
        dst = None
    return dst


def run_video(video_info, queue):
    video = cv2.VideoCapture(video_info['videoPath'])

    cnt = 0

    while True:
        ret, frame = video.read()

        if ret:  # 영상 프레임이 있으면 실행

            if cnt % 10 == 0:
                frame = rotate(frame, 90)
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                queue.put(frame)
#                queue2.put(frame)

            cnt = cnt + 1

        else:
            queue.put(None)
#            queue2.put(None)
            video.release()
            break


def show_loop(queue):
    while True:
        from_queue = queue.get()

        if from_queue is None:
            cv2.destroyAllWindows()
            break
        else:
            cv2.imshow('pepe', from_queue)
            cv2.waitKey(1)


def calculate_total_grade(video_info):
    face_cnt = video_info["face_move_cnt"]
    eye_cnt = video_info["blink_cnt"]

    total_grade = 100 - ((face_cnt + eye_cnt)*2)

    return total_grade


def read_video(video_info):
    e1 = cv2.getTickCount()

    thread_list = []

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        face_result = executor.submit(do_face_correction, video_info)
        pose_result = executor.submit(do_pose_estimation, video_info)

        print("thread result", face_result.result(), pose_result.result())

    total_grade = calculate_total_grade(video_info)  # 총 점수

    e2 = cv2.getTickCount()
    processing_time = (e2-e1) / cv2.getTickFrequency()
    print("video_correcting time :: ", processing_time)

    video_info['total_grade'] = total_grade
    video_info['processing_time'] = processing_time
    update_grade_and_time(video_info)  # 점수, 프로세싱 시간 저장

    # 분석이 제대로 되지 않았으면 0 리턴
    if face_result.result() == 0 or pose_result.result() == 0:
        return 0
