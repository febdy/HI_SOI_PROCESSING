# import the necessary packages
from conn_pymongo import update_grade_and_time
from face_pupil_processing import do_face_correction
from pose_estimation.src.run_video_multi import do_pose_estimation
from concurrent import futures
import cv2
import queue
import time


scaling_factor = 0.75 # 비디오 프레임 resize 비율


# frame 회전 (frame, 각도)
def rotate(src, degrees):  # 프레임 회전 (프레임, 회전할각도)
    if degrees == 90:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 1)   # 뒤집기

    elif degrees == 180:
        dst = cv2.flip(src, 0)   # 뒤집기

    elif degrees == 270:
        dst = cv2.transpose(src)  # 행렬 변경
        dst = cv2.flip(dst, 0)   # 뒤집기
    else:
        dst = None
    return dst


# 동영상 frame 읽어 queue에 저장
def read_video(video_info, face_queue, pose_queue):
    video = cv2.VideoCapture(video_info['videoPath'])
    print("read_video started.")
    cnt = 0  # frame 수

    while True:
        ret, frame = video.read()  # 한 장의 frame씩 읽음

        if not ret:  # frame이 없으면 끝
            face_queue.put(None)  # 얼굴 인식에 쓸 frame queue에 None 넣음
            pose_queue.put(None)  # 몸통 인식에 쓸 frame queue에 None 넣음
            video.release()
            break
        else:  # 영상 프레임이 있으면 실행
            frame = rotate(frame, 90)  # frame 회전
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            # ^ frame 사이즈 조절

            if cnt % 60 == 0:  # 60마다 한 장씩 몸통 queue에 넣음
                pose_queue.put(frame)
            elif cnt % 5 == 0:  # 5장마다 한 장씩 얼굴 queue에 넣음
                face_queue.put(frame)
            cnt += 1

    return "read_video 끝"


# 단순 프레임 queue 실행 함수 (확인용)
def show_loop(queue):
    while True:
        from_queue = queue.get()

        if from_queue is None:
            cv2.destroyAllWindows()
            break
        else:
            cv2.imshow('running_video', from_queue)  # running_video라는 이름의 window 창에 frame 엶
            cv2.waitKey(1)  # 내부 숫자에 따라 frame 재생 속도가 달라짐. 1에 가까울수록 빠름.


# 총점 계산 [100 - {(얼굴움직임+눈깜박임)*2}]
def calculate_total_grade(video_info):
    face_cnt = video_info["face_move_cnt"]
    eye_cnt = video_info["blink_cnt"]

    total_grade = 100 - ((face_cnt + eye_cnt)*2)

    return total_grade


# frame 읽기, 얼굴, 몸통 인식 동시 실행하는 함수(쓰레드 이용)
def do_process(video_info):
    e1 = cv2.getTickCount()  # 함수 시작 시간 기록

    face_queue = queue.Queue()
    pose_queue = queue.Queue()

    # 쓰레드 생성
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        # executor.submit(함수 이름, 인자1, 인자2, ...) < 쓰레드 만들어 실행.
        read_video_result = executor.submit(read_video, video_info, face_queue, pose_queue)  # frame 읽음
        face_result = executor.submit(do_face_correction, video_info, face_queue)  # 얼굴 인식
        pose_result = executor.submit(do_pose_estimation, video_info, pose_queue)  # 몸통 인식

        print("thread result", face_result.result(), pose_result.result())

    total_grade = calculate_total_grade(video_info)  # 총 점수

    e2 = cv2.getTickCount()  # 실행 종료 시간 기록
    processing_time = (e2-e1) / cv2.getTickFrequency()
    print("video_correcting time :: ", processing_time)

    video_info['total_grade'] = total_grade
    video_info['processing_time'] = processing_time
    update_grade_and_time(video_info)  # 점수, 프로세싱 시간 저장

    # 분석이 제대로 되지 않았으면 0 리턴
    # 각 쓰레드 결과 값은 분석을 성공적으로 마쳤을 때 1, 아닐 때 0.
    if face_result.result() == 0 or pose_result.result() == 0:
        return 0
