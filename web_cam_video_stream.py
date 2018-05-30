from threading import Thread, Lock
import time
import cv2


update_cnt = 0
read_cnt = 0


class WebcamVideoStream:
    def __init__(self, src='', width=320, height=240):
        global update_cnt
        global read_cnt
        update_cnt = 0
        read_cnt = 0

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
            time.sleep(0.001)

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
