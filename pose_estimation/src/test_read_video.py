# import the necessary packages
from multiprocessing import Process, Queue
# from conn_pymongo import insert_correct_result
# from face_processing import do_face_correction
import cv2

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
            if cnt % 4 == 0:
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


def read_video(video_info):
    e1 = cv2.getTickCount()

    queue = Queue()
    # queue2 = Queue()
    result_queue = Queue()

    video_process = Process(target=run_video, args=(video_info, queue, ))
    video_process.start()

    correction_process_1 = Process(target=do_face_correction, args=(video_info, queue, result_queue, ))
    correction_process_1.start()

    video_process.join()
    correction_process_1.join()

    # insert_correct_result(video_info, face_move_cnt)
    e2 = cv2.getTickCount()
    print("correcting time :: ", (e2-e1) / cv2.getTickFrequency())

    while not result_queue.empty():
        get_q = result_queue.get()

        if get_q is 0:
            return 0

    return 1