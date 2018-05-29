from threading import Thread
import cv2
from queue import Queue


class FileVideoStream:
    def __init__(self, path, queue_size=500):  # constructor
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):  # start a thread
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):  # read & decode frames from the video
        while True:
            print('뭐하니 너')

            if self.stopped:
                return

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                self.Q.put(frame)

            if not grabbed:
                self.stop()
                return

    def read(self):  # get next frame in the queue
        return self.Q.get()

    def has_more(self):  # check if the queue has more frames
        return self.Q.qsize() > 0

    def stop(self):  # Stop the Thread
        self.stopped = True