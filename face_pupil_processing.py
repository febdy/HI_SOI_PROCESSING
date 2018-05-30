# import the necessary packages
from math import floor, ceil
import imutils
import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from web_cam_video_stream import WebcamVideoStream
from conn_pymongo import update_correct_result

# 얼굴 탐지
conda_path = 'C:/Users/feb29/Anaconda3/pkgs/opencv-3.4.1-py36_200/Library/etc/haarcascades/'
# conda_path = 'C:/Users/BIT-USER/Anaconda3/Lib/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(conda_path + 'haarcascade_frontalface_default.xml')

# video = cv2.VideoCapture('C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN2.mp4')

model = load_model('face_ex.model')
scaling_factor = 0.75
kernel = np.ones((3, 3), np.uint8)


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


# 움직일 때 겹치는 구간 잇기
def check_location(miss_location):
    i = 1
    ml_len = len(miss_location)

    while i < ml_len - 2:  # 범위 벗어난 시간이 겹치면 이어지게 만듦.
        e = miss_location[i]
        next_s = miss_location[i + 1]

        if (e - next_s) <= 1:
            miss_location[i] = miss_location[i + 2]
            miss_location.pop(i + 2)
            miss_location.pop(i + 1)
            i = 1
            ml_len = len(miss_location)
        else:
            i = i + 2

    return miss_location


# 5초 단위로 움직이는 횟수 구하기
def check_cnt_per_5sec(cnt_per_5sec, frame_cnt, fps):
    sec = round(int(frame_cnt / fps))
    print(sec, len(cnt_per_5sec))
    i = 0

    if sec != 0 and sec % 5 == 0:
        i = (sec // 5) - 1
    else:
        i = sec // 5

    while len(cnt_per_5sec) <= i:
        cnt_per_5sec.append(0)

    cnt_per_5sec[i] += 1

    return cnt_per_5sec


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
def do_face_correction(video_info):
    video_path = video_info['videoPath']
    vs = WebcamVideoStream(src=video_path).start()

    video = cv2.VideoCapture(video_info['videoPath'])
    fps = video.get(cv2.CAP_PROP_FPS)  # 총 프레임 수

    is_face = -1  # 얼굴 잡았는지 확인하는 flag, 아니면 -1, 맞으면 1
    face_move_cnt = 0  # 얼굴 움직인 cnt
    chk_move = 0  # 얼굴 움직이는 중인지 확인
    frame_cnt = 0  # 현재 frame cnt
    blink_cnt = 0  # 눈 깜박임

    miss_location = []  # 얼굴 움직인 time_stamp / 짝수:움직임 시작, 홀수/:움직임 끝
    cnt_per_5sec = []  # 5초 단위로 움직인 횟수 저장
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
            ret, frame = vs.read()
            frame = rotate(frame, 90)

            print("doing face processing..")
            frame_cnt = frame_cnt + 1 + 10  # 드랍 프레임 수만큼 더해줌

            if not ret:
                vs.stop()
                cv2.destroyAllWindows()
                print("1:: face_move_cnt : ", face_move_cnt)

                video_info['face_move_cnt'] = face_move_cnt  # 얼굴 움직인 횟수

                video_info['miss_location'] = miss_location.copy()  # 움직인 시작, 끝 시간
                miss_section = check_location(miss_location)  # 움직인 시간 구간으로 바꾸어 저장
                video_info['miss_section'] = miss_section

                video_info['blink_cnt'] = blink_cnt  # 깜빡임

                total_video_time = (video.get(cv2.CAP_PROP_FRAME_COUNT) * fps) / 1000
                video_info['total_video_time'] = round(total_video_time)  # 비디오 총 재생 시간
                video_info['cnt_per_5sec'] = cnt_per_5sec  # 5초 간격 cnt
                video_info['move_direction'] = move_direction

                update_correct_result(video_info)
                return 1

            else:
                # frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

                # print('face_move_cnt:', face_move_cnt)

                if is_face is -1:  # 얼굴이 잡히지 않았을 때
                    for (x, y, w, h) in face_rects:
                        # print("얼굴안잡는중")
                        face_rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                        img = cv2.resize(face_rect, (28, 28))
                        img = img.astype("float") / 255.0
                        img = img_to_array(img)
                        img = np.expand_dims(img, axis=0)

                        (not_face, face) = model.predict(img)[0]

                        label = "face" if face > not_face else "Not face"

                        if label == "face":
                            bg = frame.copy()
                            bbox = (x, y, w, h)
                            def_x = x  # 움직임 계산할 때 기준이 되는 x, y
                            def_y = y
                            is_face = 1
                            tracker = setup_tracker(2)
                            tracking = tracker.init(frame, bbox)

                elif is_face is 1:  # 얼굴을 잡았을 때
                    # print("얼굴잡는중")
                    diff = cv2.absdiff(bg, frame)  # 기준 프레임과 다른점을 찾음
                    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                    img_dilation = cv2.dilate(closing, kernel, iterations=2)
                    imask = img_dilation > 0
                    foreground = mask_array(frame, imask)
                    foreground_display = foreground.copy()

                    tracking, bbox = tracker.update(foreground)
                    tracking = int(tracking)

                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                    if p1 == (0, 0):
                        is_face = -1
                        continue

                    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                    # print('def_x:def_y', (def_x, def_y), 'p1_x:p1_y', p1)

                    # 감지된 얼굴 확대해서 보여주기 (눈을 못잡아서 pass)
                    detected_face = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    detected_face = imutils.resize(detected_face, 3 * h, 3 * w)

                    frame[10: 10 + detected_face.shape[1], 10: 10 + detected_face.shape[0]] = detected_face

                    rects = detector(gray, 0)

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

                            # reset the eye frame counter
                            COUNTER = 0

                        # draw the total number of blinks on the frame along with
                        # the computed eye aspect ratio for the frame
                        # cv2.putText(frame, "Blinks: {}".format(blink_cnt), (10, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # cv2.putText(frame, "EAR: {:.4f}".format(ear), (200, 30),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 움직인 좌표가 4 이하면 move 검사
                    if abs(def_x - p1[0]) > 4 or abs(def_y - p1[1]) > 4:
                        if chk_move == 0:
                            face_move_cnt = face_move_cnt + 1
                            chk_move = 1

                            # 움직인 시작 초
                            start = int(floor(frame_cnt / fps))
                            miss_location.append(start)

                            # 5초동안 cnt
                            cnt_per_5sec = check_cnt_per_5sec(cnt_per_5sec, frame_cnt, fps)

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
                        if chk_move == 1:
                            end = int(ceil(frame_cnt / fps))
                            miss_location.append(end)
                            print("frame_cnt, end::::", frame_cnt, end)
                        chk_move = 0

            # cv2.imshow('Face Detector', frame)
            # if cv2.waitKey(1) == 27:
            #     break

    except Exception as error:
        print("Failed to correct.")
        print("Error:", repr(error))
        result_queue.put(0)
        return
