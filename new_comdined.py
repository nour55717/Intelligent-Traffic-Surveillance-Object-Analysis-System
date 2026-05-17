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
    def __init__(self, max_disappeared=30, max_distance=100):
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
        if object_id in self.objects:
            del self.objects[object_id]

        if object_id in self.disappeared:
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

            for i, old_centroid in enumerate(object_centroids):
                for j, new_centroid in enumerate(input_centroids):
                    distances[i, j] = math.dist(old_centroid, new_centroid)

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

VIDEO_PATH = "http://172.20.10.4:4747/video"
MODEL_PATH = "yolov8n.pt"

if not YOLO_AVAILABLE:
    print("Error: ultralytics is not installed.")
    print("Install it using:")
    print("pip install ultralytics")
    exit()

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video:", VIDEO_PATH)
    exit()

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 420, 360)

cv2.createTrackbar("YOLO Conf", "Controls", 45, 100, nothing)
cv2.createTrackbar("YOLO IOU", "Controls", 45, 100, nothing)
cv2.createTrackbar("Image Size", "Controls", 960, 1280, nothing)
cv2.createTrackbar("Min Box Area", "Controls", 500, 5000, nothing)
cv2.createTrackbar("Motion Threshold", "Controls", 230, 255, nothing)
cv2.createTrackbar("Motion Min Area", "Controls", 700, 10000, nothing)
cv2.createTrackbar("Counting Line", "Controls", 50, 100, nothing)
cv2.createTrackbar("Show Centers", "Controls", 0, 1, nothing)
cv2.createTrackbar("Show IDs", "Controls", 0, 1, nothing)
cv2.createTrackbar("Show Line", "Controls", 1, 1, nothing)
cv2.createTrackbar("Show Motion", "Controls", 0, 1, nothing)

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

tracker = CentroidTracker(max_disappeared=30, max_distance=100)

previous_positions = {}

total_in = 0
total_out = 0

allowed_classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

class_colors = {
    "person": (255, 80, 80),
    "bicycle": (255, 255, 0),
    "car": (0, 255, 0),
    "motorcycle": (0, 255, 255),
    "bus": (0, 180, 0),
    "truck": (0, 120, 0)
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    display = frame.copy()

    height, width = frame.shape[:2]

    yolo_conf_value = cv2.getTrackbarPos("YOLO Conf", "Controls")
    yolo_iou_value = cv2.getTrackbarPos("YOLO IOU", "Controls")
    image_size_value = cv2.getTrackbarPos("Image Size", "Controls")
    min_box_area = cv2.getTrackbarPos("Min Box Area", "Controls")
    motion_threshold = cv2.getTrackbarPos("Motion Threshold", "Controls")
    motion_min_area = cv2.getTrackbarPos("Motion Min Area", "Controls")
    counting_line_value = cv2.getTrackbarPos("Counting Line", "Controls")
    show_centers = cv2.getTrackbarPos("Show Centers", "Controls")
    show_ids = cv2.getTrackbarPos("Show IDs", "Controls")
    show_line = cv2.getTrackbarPos("Show Line", "Controls")
    show_motion = cv2.getTrackbarPos("Show Motion", "Controls")

    yolo_conf = max(yolo_conf_value / 100, 0.01)
    yolo_iou = max(yolo_iou_value / 100, 0.01)

    if image_size_value < 640:
        image_size_value = 640

    if image_size_value % 32 != 0:
        image_size_value = int(round(image_size_value / 32) * 32)

    counting_line_y = int((counting_line_value / 100) * height)

    detections = []

    fg_mask = background_subtractor.apply(frame)

    _, motion_mask = cv2.threshold(
        fg_mask,
        motion_threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    motion_contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    for contour in motion_contours:
        area = cv2.contourArea(contour)

        if area < motion_min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            motion_display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

    results = model(
        frame,
        verbose=False,
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=image_size_value
    )

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < min_box_area:
                continue

            detections.append((x1, y1, w, h, class_name, confidence))

    rects = [
        (x, y, w, h)
        for (x, y, w, h, class_name, confidence) in detections
    ]

    objects = tracker.update(rects)

    current_person = 0
    current_bicycle = 0
    current_vehicle = 0

    for (x, y, w, h, class_name, confidence) in detections:
        cx = x + w // 2
        cy = y + h // 2

        obj_id = None
        min_distance = float("inf")

        for object_id, centroid in objects.items():
            distance = math.dist((cx, cy), centroid)

            if distance < min_distance:
                min_distance = distance
                obj_id = object_id

        if obj_id is not None:
            old_position = previous_positions.get(obj_id)

            if old_position is not None:
                old_cx, old_cy = old_position

                if obj_id not in tracker.counted:
                    if old_cy < counting_line_y <= cy:
                        total_in += 1
                        tracker.counted.add(obj_id)

                    elif old_cy > counting_line_y >= cy:
                        total_out += 1
                        tracker.counted.add(obj_id)

            previous_positions[obj_id] = (cx, cy)

        if class_name == "person":
            current_person += 1
            display_label = "person"

        elif class_name == "bicycle":
            current_bicycle += 1
            display_label = "bicycle"

        elif class_name in ["car", "bus", "truck", "motorcycle"]:
            current_vehicle += 1
            display_label = class_name

        else:
            display_label = class_name

        box_color = class_colors.get(class_name, (0, 165, 255))

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            box_color,
            1
        )

        label_text = display_label

        if show_ids == 1 and obj_id is not None:
            label_text += f" ID:{obj_id}"

        label_bg_width = max(70, len(label_text) * 10)
        label_bg_height = 22

        cv2.rectangle(
            display,
            (x, max(0, y - label_bg_height)),
            (x + label_bg_width, y),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            label_text,
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        )

        if show_centers == 1:
            cv2.circle(
                display,
                (cx, cy),
                3,
                box_color,
                -1
            )

    if show_line == 1:
        cv2.line(
            display,
            (0, counting_line_y),
            (width, counting_line_y),
            (0, 0, 255),
            2
        )

    overlay = display.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (300, 155),
        (10, 10, 10),
        -1
    )

    cv2.rectangle(
        overlay,
        (700, 10),
        (950, 95),
        (10, 10, 10),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        display,
        0.35,
        0,
        display
    )

    cv2.putText(
        display,
        f"Persons: {current_person}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 80, 80),
        2
    )

    cv2.putText(
        display,
        f"Bicycles: {current_bicycle}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Vehicles: {current_vehicle}",
        (25, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Conf: {yolo_conf:.2f}",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        display,
        f"Entering: {total_in}",
        (720, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Leaving: {total_out}",
        (720, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Intelligent Traffic Surveillance System",
        display
    )

    if show_motion == 1:
        cv2.imshow(
            "Motion Detection Filter",
            motion_display
        )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()