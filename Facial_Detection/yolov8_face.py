import cv2
import torch
from ultralytics import YOLO
from inference import Inferencer  # Import the Inferencer class


def area(width, height):
    return width * height


class Emotion:
    classifications = []
    topEmotion = ""

    def __init__(self, emotion, confidence):
        self.topEmotion = emotion
        self.confidence = confidence
        Emotion.classifications.append(self)


# Load your custom YOLOv8 model for face detection
face_model = YOLO("Facial_Detection\yolov8n-face.pt")

# Load your emotion classification model
emotion_model_path = (
    "Facial_Detection\checkpoints\8.pt"  # Adjust this to your saved model path
)
emotion_inferencer = Inferencer(emotion_model_path)

# Define emotion labels
emotion_labels = [
    "Anger",
    "Happy",
    "Surprise",
    "Sad",
    "Contempt",
    "Fear",
    "Disgust",
    "Neutral",
]  # Adjust these labels based on your model's classes

# Initialize the webcam
cap = cv2.VideoCapture(0)
emotionHistory = []
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame for face detection
    results = face_model.predict(frame, verbose=False)
    currentEmotion = Emotion("", [])
    # Extract bounding boxes, classify emotions, and plot them
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        largestArea = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Increase bounding box size by a scale factor
            scale_factor = 1.5  # Adjust this factor to include more of the face
            box_width = x2 - x1
            box_height = y2 - y1

            # Calculate new coordinates by expanding from the center
            center_x = x1 + box_width // 2
            center_y = y1 + box_height // 2

            new_width = int(box_width * scale_factor)
            new_height = int(box_height * scale_factor)

            # Calculate new top-left and bottom-right coordinates
            x1 = int(center_x - new_width // 2)
            y1 = int(center_y - new_height // 2)
            x2 = int(center_x + new_width // 2)
            y2 = int(center_y + new_height // 2)

            # Ensure the new bounding box is within the frame bounds
            x1, y1, x2, y2 = (
                max(x1, 0),
                max(y1, 0),
                min(x2, frame.shape[1]),
                min(y2, frame.shape[0]),
            )
            # Extract face region
            face_img = frame[y1:y2, x1:x2]

            # Classify emotion
            emotion_class, emotion_confidence = emotion_inferencer.predict(face_img)
            emotion_label = emotion_labels[emotion_class]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put emotion label on the bounding box
            label = f"{emotion_label}: {emotion_confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            if area(box_width, box_height) > largestArea:
                largestArea = area(box_width, box_height)
                currentEmotion = Emotion(emotion_label, emotion_confidence)
    # Display the frame with bounding boxes and emotion labels
    cv2.imshow("Face Detection and Emotion Classification", frame)
    emotionHistory.append(currentEmotion)
    if len(emotionHistory) > 10:
        emotionHistory.pop(0)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Print the most common emotion in the last 10 frames
for emotion in emotionHistory:
    print(emotion.topEmotion)
