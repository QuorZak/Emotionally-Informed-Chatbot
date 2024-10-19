import cv2
import torch
from ultralytics import YOLO
from Facial_Detection.inference import Inferencer

videoLive = None
topEmotions = []


def area(width, height):
    return width * height


def get_top_emotions():
    global topEmotions
    topEmotion = max(set(topEmotions), key=topEmotions.count)
    return topEmotion


def run_facial_detection():  
    # Adjust these parameters as required
    ###################################  Model for Emotion Detection  ############################################
    emotion_model_path="best_models/59_best_epoch_acc_81.47_original_data_only.pt" # Path to the emotion classification model ### Change this if you want to use a different emotional model

    ####################################  Labels for Emotion Detection  ###################################
    emotion_labels = [ # Define emotion labels, they should match the labels used during training/what the final output layer represents
        "Anger",
        "Happy",
        "Surprise",
        "Sad",
        "Contempt",
        "Fear",
        "Disgust",
        "Neutral",
    ]
    ################################################################################################

    # Load the YOLOv8 model for face detection
    face_model = YOLO("Facial_Detection/yolov8n-face.pt")

    # Load the emotion classification model
    emotion_inferencer = Inferencer(emotion_model_path)

    global topEmotions
    emotionListLength = 60 # Number of emotion samples to store in the list, keep in mind the sample rate of the webcam
    scaleFactor = 1.2 # Increase bounding box size by a scale factor

    # Initialize the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # If no camera detected
    if not cap.isOpened():
        print("Error: Could not open webcam. Please connect camera and restart program")
        return Exception("No camera detected")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame for face detection
        results = face_model.predict(frame, verbose=False)
        currentEmotion = ""
        # Extract bounding boxes, classify emotions, and plot them
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            largestArea = 0
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                # Increase bounding box size by a scale factor
                box_width = x2 - x1
                box_height = y2 - y1

                # Calculate new coordinates by expanding from the center
                center_x = x1 + box_width // 2
                center_y = y1 + box_height // 2

                new_width = int(box_width * scaleFactor)
                new_height = int(box_height * scaleFactor)

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
            # if there are boxes, find the largest one
            if len(boxes) > 0:
                if area(box_width, box_height) > largestArea:
                    largestArea = area(box_width, box_height)
                    currentEmotion = emotion_label

        # Display the frame with bounding boxes and emotion labels
        cv2.imshow("Face Detection and Emotion Classification", frame)

        topEmotions.append(currentEmotion)
        if len(topEmotions) > emotionListLength:
            topEmotions.pop(0)

        # Break the loop if the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
