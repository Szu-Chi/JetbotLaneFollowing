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

def get_direction(lines):
    if lines[0] is None or lines[1] is None:
        return 0
    x1 = y1 = x2 = y2 = 0
    x1 = lines[0][0]
    y1 = lines[0][1]
    
    x2 = lines[1][2]
    y2 = lines[1][3]
    
    if x1 == x2:
        return 0
    angle = math.atan((y2 - y1)/(x2 - x1))
    # print(angle)
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
            
            lane = lane_detection.find_lane(frame, 230)
            if lane is None:
                continue
            
            lane_frame = lane_detection.draw_lane_lines(frame, lane)
            lane_frame = cv2.resize(lane_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Lane Detect frame", lane_frame)
            
            direction    = get_direction(lane)
            totle_direction += direction - avg_direction
            avg_direction = totle_direction/10

            
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            if direction > 1 or direction < -1:
                # robot.stop() 
                continue
            control.robot_follow_direction(robot, direction, 0.3)
            
        except KeyboardInterrupt:
            break
    camera.release()
    del camera