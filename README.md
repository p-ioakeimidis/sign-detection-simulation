
# Sign and Traffic Object Detection with YOLO and OpenVINO

A Python-based traffic sign, traffic light, and object detection system built using Ultralytics YOLO and optimized for OpenVINO. The project processes video frames, runs real-time deep learning inference to identify key road assets (like stop signs, pedestrians, and cars), calculates object dimensions, calculates real-time FPS performance, and prints responsive vehicle action cues based on detected signs.

## Input Video

This implementation processes a prerecorded **MP4 video** as its input source rather than a live camera feed. The sample video used for testing (`bosch_test_2.mp4`) is included in this GitHub repository, allowing users to reproduce the results and experiment with the object detection pipeline immediately after setup.

If you would like to use your own video, simply replace the input file or update the path passed to `cv2.VideoCapture()` in the source code.

## Features

* 🚦 Real-time multi-class traffic sign and object detection
* ⚡ Accelerated inference using OpenVINO-optimized YOLO models
* 📦 Automatic object bounding box dimension and area calculation
* ⏱️ Dynamic FPS calculation and on-screen telemetry overlay
* 🚗 Action trigger logic for autonomous behavior (e.g., stopping for stop signs)
* 📐 Pre-configured camera metrics for optional focal-length distance estimation

## Pipeline

1. Initialize absolute file paths for local deployment
2. Load the OpenVINO-optimized YOLO model
3. Initialize the OpenCV video stream capture
4. Read input frames sequentially
5. Run YOLO inference on the frame
6. Calculate processing speed and overlay FPS on the frame
7. Parse detected bounding box coordinates, areas, and confidence scores
8. Map class indices to human-readable labels
9. Trigger real-time action logs based on critical traffic signs (e.g., pedestrian sign, stop sign)
10. Render the annotated visual stream via OpenCV

## Project Structure

```
traffic_detection.py    # Main object detection and sign response script

```

# Installation

> **Note:** The instructions below are for **Windows**.

## Windows

1. Clone the repository:
```bash
https://github.com/p-ioakeimidis/sign-detection-simulation.git
cd sign-detection-simulation

```


2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate

```


3. Install the required Python packages:
```bash
pip install opencv-python ultralytics openvino

```


4. Run the application:
```bash
python sign_detection.py

```



---

## Ubuntu

To run the project on Ubuntu, first install the required system packages:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

```

Clone the repository:

```bash
git clone https://p-ioakeimidis/sign-detection-simulation.git
cd sign-detection-simulation

```

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate

```

Install the Python dependencies:

```bash
pip install opencv-python ultralytics openvino

```

Run the program:

```bash
python3 sign_detection.py

```

> **Note:** If you encounter issues displaying OpenCV windows on Ubuntu, make sure your system has the necessary GUI libraries installed. On desktop Ubuntu installations, these are typically available by default.

## Usage

Place your input video (`bosch_test_2.mp4`) and your OpenVINO model folder (`openvino_model`) in the project directory and run:

```bash
python3 sign_detection.py

```

Press **`q`** to exit the application window.

## Output Windows

The program displays a real-time visualization window:

* **Camera** – The original video frame overlayed with YOLO bounding boxes, class labels, confidence percentages, and a real-time FPS counter in the upper right-hand corner.

Additionally, the console prints bounding box metadata and action commands:

```
Bounding Box Width 45, Height 45, Area 2025
Detected object class: stop_sign
0 speed

```

## Supported Detection Classes

The model is pre-trained to detect and map 15 unique road entity classes:

* `crossed_highway_sign` / `highway_sign`
* `green_light` / `yellow_light` / `red_light`
* `no_entry_sign` / `one_way_road_sign` / `parking_sign`
* `pedestrian_sign` / `priority_sign` / `stop_sign` / `roundabout_sign`
* `car` / `pedestrian` / `roadblock`

## Techniques Used

* Deep Learning Object Detection (YOLO)
* Model Optimization & Quantization (OpenVINO)
* Computer Vision Draw Operations (OpenCV)
* Video Processing Streams
* Dynamic Frame Rate Evaluation

## Customization

You can adjust parameters inside the codebase such as:

* **Confidence Threshold:** Change `if conf >= 0.5:` to filter out lower or higher certainty predictions.
* **Distance Estimation Metrics:** Modify `KNOWN_WIDTH` and `FOCAL_LENGTH` to calibrate distance calculations according to your physical test setup.
* **Action Triggers:** Expand the `if/elif` logic at the bottom of the frame loop to map new automated outputs for traffic lights, roadblocks, or speed limits.

## Possible Improvements

* Uncomment and validate the mathematical pinhole camera distance estimation formula
* Integrate a tracking algorithm (like ByteTrack) to maintain persistent object IDs across frames
* Connect the action command outputs to a hardware serial interface (e.g., ROS2 or Arduino)
* Implement dynamic edge-case handling for flashing or unstable traffic signals

## License

This project is provided for educational and research purposes. Feel free to modify and extend it for your own applications.
