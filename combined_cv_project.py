import cv2
import numpy as np
from collections import OrderedDict
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.counted = set()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []

        for (x, y, w, h) in rects:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            distances = np.zeros((len(object_centroids), len(input_centroids)))

            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    distances[i, j] = math.dist(oc, ic)

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):

                if row in used_rows or col in used_cols:
                    continue

                if distances[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances.shape[0])) - used_rows
            unused_cols = set(range(distances.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


def nothing(x):
    pass


cv2.namedWindow("Controls")

cv2.createTrackbar("Canny Low", "Controls", 50, 255, nothing)
cv2.createTrackbar("Canny High", "Controls", 150, 255, nothing)
cv2.createTrackbar("Hough Threshold", "Controls", 80, 200, nothing)
cv2.createTrackbar("Min Area", "Controls", 900, 10000, nothing)
cv2.createTrackbar("Motion Threshold", "Controls", 25, 255, nothing)

if YOLO_AVAILABLE:
    model = YOLO("yolov8n.pt")
else:
    model = None

VIDEO_PATH = "D:/Semester 6/Computer Vision/Final Project/video/People.mp4"

cap = cv2.VideoCapture("D:/Semester 6/Computer Vision/Final Project/videos/People.mp4")

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

tracker = CentroidTracker()

total_in = 0
total_out = 0

previous_positions = {}

allowed_classes = ["car", "bicycle", "person", "bus", "truck", "motorcycle"]


def classify_by_shape_and_motion(w, h, area, circles_count, lines_count, speed):

    aspect_ratio = w / float(h)

    if circles_count >= 2 and aspect_ratio > 1.0:
        return "bicycle"

    if area > 7000 or aspect_ratio > 1.4:
        return "car"

    if area < 6000 and speed > 3:
        return "person"

    if h > w and area < 8000:
        return "person"

    return "moving object"


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    display = frame.copy()

    height, width = frame.shape[:2]

    counting_line_y = height // 2

    canny_low = cv2.getTrackbarPos("Canny Low", "Controls")
    canny_high = cv2.getTrackbarPos("Canny High", "Controls")
    hough_threshold = cv2.getTrackbarPos("Hough Threshold", "Controls")
    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    motion_threshold = cv2.getTrackbarPos("Motion Threshold", "Controls")

    detections = []

    fg_mask = background_subtractor.apply(frame)

    _, thresh = cv2.threshold(
        fg_mask,
        motion_threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        detections.append((x, y, w, h, "motion"))

    if YOLO_AVAILABLE:

        results = model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                conf = float(box.conf[0])

                class_name = model.names[cls_id]

                if class_name in allowed_classes and conf > 0.35:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    w = x2 - x1
                    h = y2 - y1

                    detections.append((x1, y1, w, h, class_name))

    rects = [(x, y, w, h) for (x, y, w, h, _) in detections]

    objects = tracker.update(rects)

    current_car = 0
    current_bicycle = 0
    current_person = 0

    for i, (x, y, w, h, detected_label) in enumerate(detections):

        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blur,
            canny_low,
            canny_high
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, hough_threshold),
            minLineLength=25,
            maxLineGap=10
        )

        lines_count = 0

        if lines is not None:

            lines_count = len(lines)

            for line in lines[:10]:

                x1, y1, x2, y2 = line[0]

                cv2.line(
                    display,
                    (x + x1, y + y1),
                    (x + x2, y + y2),
                    (255, 0, 255),
                    2
                )

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=max(50, canny_high),
            param2=25,
            minRadius=8,
            maxRadius=60
        )

        circles_count = 0

        if circles is not None:

            circles = np.uint16(np.around(circles))

            circles_count = len(circles[0])

            for circle in circles[0, :5]:

                cx, cy, r = circle

                cv2.circle(
                    display,
                    (x + cx, y + cy),
                    r,
                    (0, 255, 255),
                    2
                )

                cv2.circle(
                    display,
                    (x + cx, y + cy),
                    2,
                    (0, 0, 255),
                    3
                )

        obj_id = None

        cx = x + w // 2
        cy = y + h // 2

        min_dist = float("inf")

        for object_id, centroid in objects.items():

            dist = math.dist((cx, cy), centroid)

            if dist < min_dist:
                min_dist = dist
                obj_id = object_id

        speed = 0

        if obj_id is not None:

            if obj_id in previous_positions:

                old_cx, old_cy = previous_positions[obj_id]

                speed = math.dist(
                    (cx, cy),
                    (old_cx, old_cy)
                )

            previous_positions[obj_id] = (cx, cy)

        if detected_label != "motion":

            final_label = detected_label

        else:

            final_label = classify_by_shape_and_motion(
                w,
                h,
                w * h,
                circles_count,
                lines_count,
                speed
            )

        if final_label in ["car", "bus", "truck"]:

            current_car += 1

            final_label = "car"

            box_color = (0, 255, 0)

        elif final_label == "bicycle":

            current_bicycle += 1

            box_color = (255, 255, 0)

        elif final_label == "person":

            current_person += 1

            box_color = (255, 0, 0)

        else:

            box_color = (0, 165, 255)

        if obj_id is not None and obj_id not in tracker.counted:

            if obj_id in previous_positions:

                old_y = previous_positions[obj_id][1]

                if old_y < counting_line_y <= cy:

                    total_in += 1

                    tracker.counted.add(obj_id)

                elif old_y > counting_line_y >= cy:

                    total_out += 1

                    tracker.counted.add(obj_id)

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        label_text = f"{final_label} ID:{obj_id} S:{speed:.1f}"

        cv2.putText(
            display,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

        cv2.putText(
            display,
            f"Lines:{lines_count} Circles:{circles_count}",
            (x, y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    cv2.line(
        display,
        (0, counting_line_y),
        (width, counting_line_y),
        (0, 0, 255),
        2
    )

    cv2.putText(
        display,
        f"Cars: {current_car}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Bicycles: {current_bicycle}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Persons: {current_person}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        display,
        f"Entering: {total_in}",
        (720, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Leaving: {total_out}",
        (720, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Combined Smart Surveillance & Shape Analyzer",
        display
    )

    cv2.imshow(
        "Motion Mask",
        thresh
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()