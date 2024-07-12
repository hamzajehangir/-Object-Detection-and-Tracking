# -Object-Detection-and-Tracking
  Object Detection and Tracking : Develop a system capable of detecting and  tracking objects in real-time video streams. Use  deep learning models like YOLO (You Only Look  Once) or Faster R-CNN for accurate object  detection and tracking
import cv2
import numpy as np

# Load custom YOLO model with modified architecture
net = cv2.dnn.readNet("custom_yolo.weights", "custom_yolo.cfg")

# Load video capture with custom camera settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Read frame from video capture with custom frame rate
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply custom preprocessing to frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert frame to blob with custom normalization
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), [0,0,0], 1, crop=False)
    blob = np.clip(blob, 0, 1)
    
    # Set input for custom YOLO model
    net.setInput(blob)
    
    # Run custom YOLO model with modified layers
    outs = net.forward(getOutputsNames(net))
    
    # Get detected objects with custom confidence threshold
    objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.7 and classId == 0:  # Person class
                x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                objects.append((x, y, w, h))
    
    # Draw bounding boxes around detected objects with custom colors
    for obj in objects:
        x, y, w, h = obj
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 128, 0), 2)
    
    # Display output with custom window title
    cv2.imshow('Custom Object Detection', frame)
    
    # Exit on key press with custom key
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release resources with custom message
cap.release()
cv2.destroyAllWindows()
print("Custom object detection system terminated.")


