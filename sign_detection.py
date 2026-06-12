import os
import cv2
from ultralytics import YOLO

# =========================
# Paths (GitHub-friendly)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "openvino_model")
VIDEO_PATH = os.path.join(BASE_DIR, "data", "bosch_test.mp4")

# =========================
# Load model + video
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

# =========================
# Class mapping
# =========================
names = {
    0: "crossed_highway_sign",
    1: "green_light",
    2: "highway_sign",
    3: "no_entry_sign",
    4: "one_way_road_sign",
    5: "parking_sign",
    6: "pedestrian_sign",
    7: "priority_sign",
    8: "red_light",
    9: "roundabout_sign",
    10: "stop_sign",
    11: "yellow_light",
    12: "car",
    13: "pedestrian",
    14: "roadblock"
}

# =========================
# Constants (if you need later)
# =========================
KNOWN_WIDTH = 0.07
FOCAL_LENGTH = 3058

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Run YOLO model on the captured frame and store the results
    results = model(frame)
    annotator = Annotator(frame, example= names)
    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)
    # Run YOLO model on the captured frame and store the results

    KNOWN_WIDTH = 0.07
    FOCAL_LENGTH = 3058
    
    # Print detected object classes
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            # Get the class index and convert it to an integer
            width, height, area = annotator.get_bbox_dimension(bbox)
            cls = box.cls
            class_idx = int(box.cls[0])
            conf = box.conf
            if conf >= 0.5:
                # Use the model's names dictionary to get the class name
                print("Bounding Box Width {}, Height {}, Area {}".format(width, height, area))
                class_name = model.names[class_idx]
                print("Detected object class:", class_name)
                # Calculate the distance
                #distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width.item()
                #print(f"Object: {model.names[int(cls)]}, Distance: {distance:.2f} meters")

    
    if class_name == 'pedestrian_sign':
        print('reduce speed')
    elif class_name == 'stop_sign':
        print('0 speed')
    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
