import cv2
import numpy as np
import time

def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def drawLines(image, lines, color=(0, 255, 0), thickness=10):
    image = np.copy(image)
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness)
    
    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return image

def process(image, threshold1, threshold2, hough_threshold, min_line_length, max_line_gap):
    height, width = image.shape[0], image.shape[1]
    region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, threshold1, threshold2)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=hough_threshold, lines=np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    imageWithLine = drawLines(image, lines, color=(0, 255, 0), thickness=10)
    return imageWithLine

def nothing(x):
    pass

try:
    cap = cv2.VideoCapture("video2.mp4")
    if not cap.isOpened():
        raise IOError("Error: Could not open video file.")
except IOError as e:
    print(e)
    exit()

cv2.namedWindow("Parameters")
cv2.createTrackbar("Canny Threshold1", "Parameters", 250, 500, nothing)
cv2.createTrackbar("Canny Threshold2", "Parameters", 120, 500, nothing)
cv2.createTrackbar("Hough Threshold", "Parameters", 200, 500, nothing)
cv2.createTrackbar("Min Line Length", "Parameters", 150, 500, nothing)
cv2.createTrackbar("Max Line Gap", "Parameters", 4, 100, nothing)

prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    threshold1 = cv2.getTrackbarPos("Canny Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Canny Threshold2", "Parameters")
    hough_threshold = cv2.getTrackbarPos("Hough Threshold", "Parameters")
    min_line_length = cv2.getTrackbarPos("Min Line Length", "Parameters")
    max_line_gap = cv2.getTrackbarPos("Max Line Gap", "Parameters")
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    img = process(img, threshold1, threshold2, hough_threshold, min_line_length, max_line_gap)

    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("img", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()