This project was developed to create a complete intelligent surveillance pipeline capable of analyzing live scenes in real time using both deep learning and classical computer vision approaches.

The system performs:

Real-time object detection
Multi-object tracking
Motion analysis
Shape extraction
Object counting
Scene monitoring
Live stream analysis

while maintaining real-time performance and interactive control.

Core Features
Real-Time Object Detection

The system uses YOLOv8n for detecting:

Persons
Cars
Bicycles
Trucks
Buses
Motorcycles

The lightweight YOLOv8n model was selected to achieve fast inference while preserving strong detection performance.

Multi-Object Tracking

A centroid tracking algorithm is implemented to:

Assign unique IDs
Track objects across frames
Reduce identity switching
Support line-crossing counting
Intelligent Counting System

The project includes a virtual line-crossing system capable of:

Counting entering objects
Counting leaving objects
Reducing double counting
Monitoring scene flow
Motion Detection

Motion analysis is implemented using:

Background subtraction
Thresholding
Morphological operations
Motion region filtering

This allows the system to analyze moving regions independently from YOLO detection.

Shape-Based Analysis

The system integrates:

Canny Edge Detection
Hough Line Transform
Hough Circle Transform

to perform additional structural and shape-based scene analysis.

Interactive Runtime Controls

The application includes adjustable OpenCV trackbars for:

Confidence Threshold
IOU Threshold
Image Size
Motion Threshold
Minimum Detection Area
Counting Line Position
Tracking Visualization
Live Camera & Wi-Fi IP Stream Support

The system supports:

Local video files
Real-time IP camera streams

Example:

VIDEO_PATH = "http://192.168.x.x:8080/video"

This allows integration with:

Mobile IP cameras
CCTV systems
Wireless surveillance cameras
Real-time traffic monitoring setups
Streamlit Web Interface

A professional Streamlit interface is also included for:

Interactive demonstrations
Video uploads
Real-time analysis
Live statistics display
Cleaner UI presentation
Technologies Used
Python
OpenCV
YOLOv8
NumPy
Streamlit
Potential Applications
Intelligent surveillance systems
Smart traffic monitoring
Smart city infrastructure
Vehicle analysis systems
Public safety systems
Real-time scene analytics
