from jetbot import Robot

import cv2
import numpy as np
import nanocamera as nano

LINE_COLOR = [0, 255, 255]
DIRECTION_COLOR = [0, 0, 255]
LINE_THICKNESS = 3
SHOW_PREPROCRESULT = True

def HSL_color_selection(image):
    #Convert the input image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    #White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask) 
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size = 7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold = 50, high_threshold = 150):
    return cv2.Canny(image, low_threshold, high_threshold)


def hough_transform(image):
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 10       #Only lines that are greater than threshold will be returned.
    minLineLength = 10   #Line segments shorter than that are rejected.
    maxLineGap = 0      #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:
        return None
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.5
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
    
def draw_lane_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  LINE_COLOR, LINE_THICKNESS)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def find_lane(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    ret, binary = cv2.threshold(hist, 240, 255, cv2.THRESH_BINARY)
    median = cv2.medianBlur(binary, 5)
    hough = hough_transform(median)
    if len(hough) < 4:
        return None

    if SHOW_PREPROCRESULT:
        preproc_result = median
        cv2.imshow("Preprocessing Result frame", preproc_result)     
    return lane_lines(src, hough)

if __name__ == '__main__':
    robot = Robot()
    camera = nano.Camera(flip=0, width=640, height=480, fps=30)
    totle_direction = 0
    avg_direction = 0
    while (camera.isReady()):
        try:
            frame = camera.read()
            BGR_channels = []
            cv2.split(frame, BGR_channels)
            resize = cv2.resize(frame, (160, 140), interpolation=cv2.INTER_CUBIC)
            
            lane = find_lane(resize)
            if lane is None:
                continue

            lane_frame = draw_lane_lines(resize, lane)

            lane_frame = cv2.resize(lane_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Lane Detect frame", lane_frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        except KeyboardInterrupt:
            break
    camera.release()
    del camera