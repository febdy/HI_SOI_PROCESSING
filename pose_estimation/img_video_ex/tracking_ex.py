import sys  # system functions (ie. exiting the program)
import time  # for time operations
import cv2  # (OpenCV) computer vision functions (ie. tracking)


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


video = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/python_workplace/HUN.mp4')

# Read first frame
success, frame = video.read()
if not success:
    print("first frame not read")
    sys.exit()

tracker = setup_tracker(4)

# Select roi for bbox
bbox = cv2.selectROI(frame, False)
cv2.destroyAllWindows()

# Initialize tracker with first frame and bounding box
tracking_success = tracker.init(frame, bbox)

while True:
    time.sleep(0.02)

    timer = cv2.getTickCount()

    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break

    # Update tracker
    tracking_success, bbox = tracker.update(frame)

    # Draw bounding box
    if tracking_success:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # Display result
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break  # ESC pressed

cv2.destroyAllWindows()
video.release()
