# import the necessary packages
from math import floor, ceil
import imutils
import cv2
import dlib
import numpy as np
from imutils import face_utils

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras import backend as K
from scipy.spatial import distance as dist

from cal_cnt_per_5sec import check_cnt_per_5sec
from conn_pymongo import update_correct_result


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


# 움직일 때 겹치는 시간 구간 잇기
def check_location(miss_location):
    i = 1
    ml_len = len(miss_location)

    while i < ml_len - 2:  # 범위 벗어난 시간이 겹치면 이어지게 만듦.
        e = miss_location[i]
        next_s = miss_location[i + 1]

        if (e - next_s) <= 1:  # 끝난 시간이 다음 움직임 시작 시간보다 +1이거나 같으면
            miss_location[i] = miss_location[i + 2]  # 다음 끝난 시간을 현재 끝난 시간으로 바꾼 뒤
            miss_location.pop(i + 2)  # 다음 시작, 끝 시간을 지운다
            miss_location.pop(i + 1)
            i = 1
            ml_len = len(miss_location)
        else:
            i = i + 2

    return miss_location


# 눈 깜빡임 종횡비 구하기
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# 얼굴 인식 실행
def do_face_correction(video_info, face_queue):
    # 얼굴 탐지
    conda_path = 'cascade/'
    # conda_path = 'C:/Users/BIT-USER/Anaconda3/Lib/site-packages/cv2/data/'
    face_cascade = cv2.CascadeClassifier(conda_path + 'haarcascade_frontalface_default.xml')

    # video = cv2.VideoCapture('C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN2.mp4')

    K.clear_session()
    model = load_model('face_ex.model')  # classify에 쓰이는 데이터셋
    kernel = np.ones((3, 3), np.uint8)

    # 영상 총 재생 시간 구하기 위한 변수들
    video = cv2.VideoCapture(video_info['videoPath'])
    fps = video.get(cv2.CAP_PROP_FPS)  # 총 프레임 수

    is_face = -1  # 얼굴 잡았는지 확인하는 flag, 아니면 -1, 맞으면 1
    face_move_cnt = 0  # 얼굴 움직인 cnt
    chk_move = 0  # 얼굴 움직이는 중인지 확인
    frame_cnt = 0  # 현재 frame cnt
    blink_cnt = 0  # 눈 깜박임

    miss_location = []  # 얼굴 움직인 time_stamp / 짝수:움직임 시작, 홀수/:움직임 끝
    face_move_cnt_per_5sec = []  # 5초 단위로 움직인 횟수 저장
    eye_blink_cnt_per_5sec = []  # 5초 단위로 눈 깜박인 횟수 저장
    move_direction = [0, 0, 0, 0]  # 상, 하, 좌, 우

    try:
        EYE_AR_THRESH = 0.3
        EYE_AR_CONSEC_FRAMES = 3

        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        # blink_cnt = 0
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        while True:
            frame = face_queue.get()  # face queue에서 한 장씩 가져옴
            print("doing face processing..")
            frame_cnt += 1 + 5  # 드랍 프레임 수만큼 더해줌

            if frame is None:  # queue에 들어있는 값이 None일 때 종료
                cv2.destroyAllWindows()
                print("1:: face_move_cnt : ", face_move_cnt)

                video_info['face_move_cnt'] = face_move_cnt  # 얼굴 움직인 횟수

                video_info['miss_location'] = miss_location.copy()  # 움직인 시작, 끝 시간
                miss_section = check_location(miss_location)  # 움직인 시간 구간으로 바꾸어 저장
                video_info['miss_section'] = miss_section

                video_info['blink_cnt'] = blink_cnt  # 깜박임
                video_info['eye_blink_cnt_per_5sec'] = eye_blink_cnt_per_5sec  # 눈 깜박임 5초 간격 cnt

                total_video_time = (video.get(cv2.CAP_PROP_FRAME_COUNT) * fps) / 1000
                video_info['total_video_time'] = round(total_video_time)  # 비디오 총 재생 시간
                video_info['face_move_cnt_per_5sec'] = face_move_cnt_per_5sec  # 얼굴 움직임 5초 간격 cnt
                video_info['move_direction'] = move_direction  # 움직인 방향별 횟수 [상, 하, 좌, 우]

                update_correct_result(video_info)  # MongoDB update.
                return 1

            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame을 회색으로 만듦
                face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)  # frame에서 얼굴을 잡아냄

                # print('face_move_cnt:', face_move_cnt)

                if is_face is -1:  # 얼굴이 잡히지 않았을 때
                    for (x, y, w, h) in face_rects:
                        # print("얼굴안잡는중")
                        # 얼굴 주위 상자 그리기 (frmae, 시작좌표, 끝좌표, 네모 색, 범위)
                        face_rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                        img = cv2.resize(face_rect, (28, 28))  # 얼굴 상자 사이즈 (28, 28)로 변경. 교육된 데이터셋이 (28, 28)
                        img = img.astype("float") / 255.0
                        img = img_to_array(img)
                        img = np.expand_dims(img, axis=0)

                        (not_face, face) = model.predict(img)[0]  # 얼굴인지 아닌지 판별

                        label = "face" if face > not_face else "Not face"  # 더 높은 %인 쪽으로 label 정함

                        if label == "face":  # 잡은 위치가 얼굴일 때
                            bg = frame.copy()  # tracking을 위한 현재 frame 복사
                            bbox = (x, y, w, h)  # 현재 얼굴 상자 좌표
                            def_x = x  # 움직임 계산할 때 기준이 되는 x, y
                            def_y = y
                            is_face = 1
                            tracker = setup_tracker(2)  # tracker-KCF 사용
                            tracking = tracker.init(frame, bbox)

                elif is_face is 1:  # 얼굴을 잡았을 때
                    # print("얼굴잡는중")
                    diff = cv2.absdiff(bg, frame)  # 기준 프레임과 다른점을 찾음
                    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # frame 회색으로 변환
                    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)  # threshold 적용
                    # threshold - 이미지 이진화(흑백으로 나타냄)
                    # 기준값 이하는 0(검은색), 이상은 1(흰색)

                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 노이즈 제거
                    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # opening의 반대
                    img_dilation = cv2.dilate(closing, kernel, iterations=2)
                    imask = img_dilation > 0
                    foreground = mask_array(frame, imask)  # frame에 mask 적용
                    foreground_display = foreground.copy()  # 복사

                    tracking, bbox = tracker.update(foreground)  # 트래킹
                    tracking = int(tracking)

                    p1 = (int(bbox[0]), int(bbox[1]))  # 트래킹 상자 시작 좌표
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 트래킹 상자 끝 좌표

                    if p1 == (0, 0):  # 트래킹 상자 시작이 0,0이면 (얼굴을 못 찾으면)
                        is_face = -1  # 다시 얼굴 찾기 시작
                        continue

                    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)  # 파란색 상자 그리기
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)  # 파란색 상자 그리기

                    # print('def_x:def_y', (def_x, def_y), 'p1_x:p1_y', p1)

                    # 감지된 얼굴 확대해서 보여주기 (눈을 못잡아서 pass)
                    detected_face = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    detected_face = imutils.resize(detected_face, 3 * h, 3 * w)  # 얼굴 잡은 영역 3배로 만들기

                    frame[10: 10 + detected_face.shape[1], 10: 10 + detected_face.shape[0]] = detected_face  # 왼쪽 상단에 표시

                    rects = detector(gray, 0)  # 눈동자 잡기 위한 frame은 gray scale이어야 함.

                    # loop over the face detections
                    for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)

                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1

                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold
                        else:
                            # if the eyes were closed for a sufficient number of
                            # then increment the total number of blinks
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                blink_cnt += 1
                                eye_blink_cnt_per_5sec = check_cnt_per_5sec(eye_blink_cnt_per_5sec, frame_cnt, fps)

                            # reset the eye frame counter
                            COUNTER = 0

                        # draw the total number of blinks on the frame along with
                        # the computed eye aspect ratio for the frame
                        # cv2.putText(frame, "Blinks: {}".format(blink_cnt), (10, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # cv2.putText(frame, "EAR: {:.4f}".format(ear), (200, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 움직인 좌표가 4 이상이면 얼굴 움직임 검사
                    if abs(def_x - p1[0]) > 4 or abs(def_y - p1[1]) > 4:
                        if chk_move == 0:  # 이전까지 움직이지 않았을 시
                            face_move_cnt = face_move_cnt + 1  # 움직임 횟수 +1
                            chk_move = 1  # 움직이는 도중이라는 flag 표시

                            # 움직인 시작 초
                            start = int(floor(frame_cnt / fps))
                            miss_location.append(start)  # 움직였을 때의 시간 기록

                            # 5초동안 cnt
                            face_move_cnt_per_5sec = check_cnt_per_5sec(face_move_cnt_per_5sec, frame_cnt, fps)

                            # 움직인 방향 move_direction[상, 하, 좌, 우]
                            if def_y - p1[1] < 0:  # 상
                                move_direction[0] += 1
                            if def_y - p1[1] > 0:  # 하
                                move_direction[1] += 1
                            if def_x - p1[0] < 0:  # 좌
                                move_direction[2] += 1
                            if def_x - p1[0] > 0:  # 우
                                move_direction[3] += 1

                            print("frame_cnt, start::::", frame_cnt, start)

                    else:
                        if chk_move == 1:  # 좌표차가 4 이하고 && 이전까지 움직이고 있었을 때
                            end = int(ceil(frame_cnt / fps))  # 끝 시간
                            miss_location.append(end)  # 움직임 끝난 시간 기록
                            print("frame_cnt, end::::", frame_cnt, end)
                        chk_move = 0  # 움직임 flag를 0으로.

            # cv2.imshow('Face Detector', frame)
            # if cv2.waitKey(1) == 27:
            #     break

    except Exception as error:
        print("Failed to correct.")
        print("Error:", repr(error))
        return
