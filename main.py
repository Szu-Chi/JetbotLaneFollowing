from jetbot import Robot

import cv2
import time
import math
import numpy as np
import nanocamera as nano

import lane_detection
import control

DRAW_DIRECTION = False


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def get_direction(image, lines):
    x1 = y1 = x2 = y2 = 0
    for line in lines:
        if line is not None:
            x1 += line[0][0] / 2
            y1 += line[0][1] / 2
            # x2 += line[1][0] / 2
            # y2 += line[1][1] / 2
    # angle = angle_between((x1 - x2, y1 - y2), (0, 0))
    angle = math.atan(x1/y1)
    # print(angle, lines)
    if DRAW_DIRECTION:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),  DIRECTION_COLOR, LINE_THICKNESS)
    return angle
        

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
            
            lane = lane_detection.find_lane(resize)
            if lane is None:
                continue
            
            lane_frame = lane_detection.draw_lane_lines(resize, lane)
            lane_frame = cv2.resize(lane_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Lane Detect frame", lane_frame)
            
            direction    = get_direction(lane_frame, lane)
            totle_direction += direction - avg_direction
            avg_direction = totle_direction/10

            
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            if direction > 1 or direction < 0:
                # robot.stop() 
                continue
            control.robot_follow_direction(direction)
            
        except KeyboardInterrupt:
            break
    camera.release()
    del camera