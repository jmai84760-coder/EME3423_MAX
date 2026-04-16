import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Rep counter variables
lateral_raise_count = 0
stage = "Lowered"  # States: "Lowered" or "Raised"


def calculate_angle(a, b, c):
    """Calculate angle between 3 landmarks (hip, shoulder, wrist) to measure arm position"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


# Open camera
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Flip image for natural interaction
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            # Extract shoulder/hip/wrist coordinates (right side example)
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate arm-to-body angle (hip-shoulder = body midline; shoulder-wrist = arm direction)
            arm_angle = calculate_angle(hip, shoulder, wrist)

            # Lateral raise logic: >150° = Raised, <30° = Lowered
            if arm_angle > 150 and stage == "Lowered":
                stage = "Raised"
            if arm_angle < 30 and stage == "Raised":
                stage = "Lowered"
                lateral_raise_count += 1  # Increment rep when returning to lowered position

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Display results (left side of window)
        cv2.putText(img, "Lateral Raise Counter", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Reps: {lateral_raise_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Stage: {stage}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show window
        cv2.imshow('Lateral Raise Counter', img)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
