# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
from img_video_ex.conn_pymongo import insert_test

# 얼굴 탐지
# conda_path = 'C:/Users/feb29/Anaconda3/pkgs/opencv-3.4.1-py36_200/Library/etc/haarcascades/'
conda_path = 'D:/javaForever/util/opencv/sources/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(conda_path + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/HUN2.mp4')
# video = cv2.VideoCapture('C:/Users/feb29/PycharmProjects/OpenCV_Ex/HUN.mp4')
model = load_model('face_ex.model')
scaling_factor = 0.75
kernel = np.ones((3, 3), np.uint8)


def rotate(src, degrees):
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


is_face = -1

cnt = 0
chk_move = 0

while True:
    ret, frame = video.read()

    frame = rotate(frame, 90)

    if ret:
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        print('cnt:', cnt)

        if is_face is -1:  # 얼굴이 잡히지 않았을 때
            for (x, y, w, h) in face_rects:
                print("얼굴안잡는중")
                img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                img = cv2.resize(img2, (28, 28))
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
            print("얼굴잡는중")
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

            print('def_x:def_y', (def_x, def_y), 'p1_x:p1_y', p1)

            detected_face = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            detected_face = imutils.resize(detected_face, 3 * h, 3 * w)

            frame[10: 10+detected_face.shape[1], 10: 10+detected_face.shape[0]] = detected_face

            if abs(def_x - p1[0]) > 4 or abs(def_y - p1[1]) > 4:
                if chk_move == 0:
                    cnt = cnt+1
                    chk_move = 1
            else:
                chk_move = 0

        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

insert_test(cnt)

video.release()
cv2.destroyAllWindows()
