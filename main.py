import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'testJarak.mp4'
cap = cv2.VideoCapture(video_path)
cv2.namedWindow('frame')

count = 1
while cap.isOpened():
  success, frame = cap.read()
  # sky = frame[350:450, 125:300]
  sky = frame
  if success:
    results = model(sky, save=True)
    annotated_frame = results[0].plot()
    print(annotated_frame)
    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()