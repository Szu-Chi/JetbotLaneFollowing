from jetbot import Robot

import time

ROBOT_MOVEMENT = True
# ROBOT_MOVEMENT = False

SWAP_LR_MOTOR = True
MAX_SPEED = 0.6


def calc_motor_PWR(direction, normal_speed=0.3):
    if direction is None:
        return 0, 0
    fix = direction/10
    if fix > 0:
        right_PWR = normal_speed + fix
        left_PWR = normal_speed
    else:
        right_PWR = normal_speed
        left_PWR = normal_speed - fix
    print([direction, fix, left_PWR, right_PWR])
    return left_PWR, right_PWR

def robot_move(robot, left_motor_PWR=0, right_motor_PWR=0, adjust_gain = 0.03):
    if ROBOT_MOVEMENT : 
        if SWAP_LR_MOTOR:
            robot.set_motors(-(right_motor_PWR * (1-adjust_gain)), -(left_motor_PWR * (1+adjust_gain)))
        else:
            robot.set_motors(left_motor_PWR * (1+adjust_gain), right_motor_PWR * (1-adjust_gain))
        
def robot_follow_direction(robot, direction=None, normal_speed=0.3):
    left_PWR, right_PWR = calc_motor_PWR(direction, normal_speed)
    robot_move(robot, left_PWR, right_PWR)

        

if __name__ == '__main__':
    robot = Robot()
    robot_move(0.3, 0.3)
    time.sleep(1)
    robot.stop()        
    