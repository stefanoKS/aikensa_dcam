import cv2
import sys

def initialize_camera(camNum): #Init 4k cam

    cap = cv2.VideoCapture(camNum, cv2.CAP_V4L2) #for ubuntu. It's D_SHOW for windows

    # cap = cv2.VideoCapture(camNum, cv2.CAP_ARAVIS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 134)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 128)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # cap.set(cv2.CAP_PROP_EXPOSURE, 2000)

    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1500)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)

    cap.set(cv2.CAP_PROP_GAMMA, 50)
    cap.set(cv2.CAP_PROP_GAIN, 100)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    # 4k res

    cap.set(cv2.CAP_PROP_FPS, 60) # Set the desired FPS

    return cap
