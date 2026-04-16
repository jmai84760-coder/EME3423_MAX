import cv2
import numpy as np

# ---------------------- 1. Load YOLO Model & Class File ----------------------
# Note: Ensure yolov3.weights, yolov3.cfg, and coco.names are in the same directory as this script
net = cv2.dnn.readNet("yolov3-608.weights", "yolov3-320.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ---------------------- 2. Define Target Fruits & Prices (COCO-supported) ----------------------
target_fruits = {
    "apple": 1.5,    # Apple: $1.5 each
    "banana": 0.8,   # Banana: $0.8 each
    "orange": 1.2    # Orange: $1.2 each
}
min_confidence = 0.8  # Minimum confidence threshold (80%)

# ---------------------- 3. Initialize Camera/Video ----------------------
# Use 0 for default webcam, or replace with a video path (e.g., "fruits_video.mp4")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video/camera feed fails

    height, width, channels = frame.shape

    # ---------------------- 4. Preprocess Frame for YOLO ----------------------
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)  # Get YOLO detection results

    # ---------------------- 5. Parse Detection Results ----------------------
    boxes = []       # Bounding box coordinates
    confidences = [] # Confidence scores
    class_ids = []   # Class IDs
    fruit_counts = {fruit: 0 for fruit in target_fruits}  # Track fruit counts

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter: Only target fruits with ≥80% confidence
            fruit_name = classes[class_id]
            if confidence >= min_confidence and fruit_name in target_fruits:
                # Calculate bounding box coordinates (scaled to original frame)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                fruit_counts[fruit_name] += 1  # Increment count for detected fruit

    # ---------------------- 6. Non-Max Suppression (Remove Overlapping Boxes) ----------------------
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    # ---------------------- 7. Draw Bounding Boxes & Labels ----------------------
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            fruit_name = classes[class_ids[i]]
            confidence = confidences[i]

            # Draw bounding box (green, line width 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw label (fruit name + confidence) at top-left of the box
            label = f"{fruit_name}: {confidence:.1%}"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    # ---------------------- 8. Calculate & Display Total Count & Price ----------------------
    total_fruits = sum(fruit_counts.values())
    total_price = sum(count * target_fruits[fruit] for fruit, count in fruit_counts.items())

    # Show total in top-right corner (red text)
    total_text = f"Total: {total_fruits} fruits | Price: ${total_price:.2f}"
    cv2.putText(
        frame, total_text, (width - 300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    # ---------------------- 9. Display Result Window ----------------------
    cv2.imshow("Fruit Detection & Pricing", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()