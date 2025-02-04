import cv2
import sys


def initialize_camera(camNum): #Init 4k cam

    # cap = cv2.VideoCapture(camNum, cv2.CAP_V4L2) #for ubuntu. It's DSHOW for windows
    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW)
    
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 110)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 128)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 2000)
    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2500)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    # cap.set(cv2.CAP_PROP_GAMMA, 50)
    # cap.set(cv2.CAP_PROP_GAIN, 100)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    # 4k res

    cap.set(cv2.CAP_PROP_FPS, 30) # Set the desired FPS

    return cap

def initialize_gaikan_camera(camNum): #Init 480p_gaikan cam

    # cap = cv2.VideoCapture(camNum, cv2.CAP_V4L2) #for ubuntu. It's DSHOW for windows
    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW)
    
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 110)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 128)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 2000)
    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2500)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    # cap.set(cv2.CAP_PROP_GAMMA, 50)
    # cap.set(cv2.CAP_PROP_GAIN, 100)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv2.CAP_PROP_FPS, 30) # Set the desired FPS

    return cap
