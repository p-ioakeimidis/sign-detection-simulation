import os
import cv2
from ultralytics import YOLO

# =========================
# Paths (GitHub-friendly)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model_openvino")
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

    # YOLO inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # FPS calculation
    inference_time = results[0].speed["inference"]
    fps = 1000 / inference_time if inference_time > 0 else 0

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # =========================
    # Detection logic
    # =========================
    class_name = None

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf >= 0.5:
                class_name = model.names[cls_id]
                print(f"Detected: {class_name} ({conf:.2f})")

    # Simple driving logic
    if class_name == "pedestrian_sign":
        print("reduce speed")
    elif class_name == "stop_sign":
        print("0 speed")

    # Show output
    cv2.imshow("YOLO Video Test", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
