from ultralytics import YOLO
import cv2

model = YOLO('runs/train/myanmar_dl/weights/best.pt')
'''
# Test on a sample image
img = cv2.imread("datasets/images/train/att.O7deG1_krSQTrKhxANrXU540GIV4s2yBzj7B4oQBErg.jpg")
results = model(img)

# Display detections
results[0].show()  # Opens a window showing bounding boxes
'''

#Test on a new image
new_img = cv2.imread("test_photo/test_5.jpg")
results = model(new_img)

# Display detections
results[0].show()  # Opens a window showing bounding boxes
'''
for box in results[0].boxes:
    cls = int(box.cls)
    label = results[0].names[cls]

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = new_img[y1:y2, x1:x2]

    cv2.imshow(label, crop)
cv2.waitKey(0)
'''