import cv2
import numpy as np

# YOLO dosyalarını yükleyin
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    image = np.copy(image)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return image

def draw_3d_box(image, bbox):
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.line(image, (x, y), (x + int(w * 0.5), y - int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + w, y), (x + int(w * 1.5), y - int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + w, y + h), (x + int(w * 1.5), y + int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x, y + h), (x + int(w * 0.5), y + int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + int(w * 0.5), y - int(h * 0.5)), (x + int(w * 1.5), y - int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + int(w * 0.5), y - int(h * 0.5)), (x + int(w * 0.5), y + int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + int(w * 1.5), y - int(h * 0.5)), (x + int(w * 1.5), y + int(h * 0.5)), (0, 255, 0), 2)
    cv2.line(image, (x + int(w * 0.5), y + int(h * 0.5)), (x + int(w * 1.5), y + int(h * 0.5)), (0, 255, 0), 2)

def process(image):
    height, width = image.shape[:2]

    # Şerit takibi için gri tonlama ve Canny kenar algılama
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 250, 120)

    # ROI tanımlama
    region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # Hough dönüşümü ile şerit çizgilerini bulma
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=200, lines=np.array([]), minLineLength=150, maxLineGap=4)
    image_with_lines = draw_lines(image, lines)

    # YOLO nesne algılama
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "car":
                draw_3d_box(image_with_lines, (x, y, w, h))

    return image_with_lines

cap = cv2.VideoCapture("video1.mp4")

while True:
    success, img = cap.read()
    if success:
        img = process(img)
        cv2.imshow("img", img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
