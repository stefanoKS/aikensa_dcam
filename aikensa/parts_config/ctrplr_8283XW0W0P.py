import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image

pygame.mixer.init()
ok_sound = pygame.mixer.Sound("aikensa/sound/positive_interface.wav") 
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")  
kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"


pitchSpecLH = [85, 87, 98, 98, 78, 113, 103]
pitchSpecRH = [103, 113, 78, 98, 98, 87, 85]
pitchToleranceLH = [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
pitchToleranceRH = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0]

pixelMultiplier = 0.20005 #basically multiplier from 1/arucoplanarize param -> will create a constant for this later

text_offset = 50

endoffset_y = 0



def partcheck(img, detections, hanire_detection, partid=None):

    sorted_detections = sorted(detections, key=lambda d: d.bbox.minx)


    middle_lengths = []

    detectedid = []
    customid = []

    detectedPitch = []
    deltaPitch = []

    detectedposX = []
    detectedposY = []

    pitchresult = []
    checkedPitchResult = []

    prev_center = None

    flag_pitchfuryou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0

    for detection in sorted_detections:
        
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny
        class_id = detection.category.id
        class_name = detection.category.name
        score = detection.score.value

        detectedid.append(class_id)
        detectedposX.append(x)
        detectedposY.append(y)

        #id 0 object is white clip
        #id 1 object is brown clip
        #id 2 object is yellow clip
        #id 3 object is orange clip

        if class_id == 0: #Clip is white
            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 255, 255))
        if class_id == 1: #Clip is brown
            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(139, 69, 19))
        if class_id == 2: #Clip is yellow
            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 255, 0))
        if class_id == 3: #Clip is orange
            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 165, 0))

            
        if prev_center is not None:
            length = calclength(prev_center, center)*pixelMultiplier
            middle_lengths.append(length)
            line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
            img = drawbox(img, line_center, length)
            img = drawtext(img, line_center, length)
        prev_center = center

    #not used temporarily

    # for detect_custom in hanire_detection:
    #     class_id_custom, x_custom, y_custom, _, _, _ = detect_custom
    #     class_id_custom = int(class_id_custom)
    #     customid.append(class_id_custom)

    #     if class_id_custom == 0:
    #         drawcircle(img, (x_custom*img.shape[1], y_custom*img.shape[0]), 0)
    #     elif class_id_custom == 1:
    #         drawcircle(img, (x_custom*img.shape[1], y_custom*img.shape[0]), 1)
        

    detectedPitch = middle_lengths
    #pop first and last element of the list
    checkedPitchResult = detectedPitch[1:-1]
    detectedposX = detectedposX[1:-1]
    detectedposY = detectedposY[1:-1]

    if partid == "LH":
        pitchresult = check_tolerance(checkedPitchResult, pitchSpecLH, pitchToleranceLH)

        if len(detectedPitch) == 6:
            deltaPitch = [detectedPitch[i] - pitchSpecLH[i] for i in range(len(pitchSpecLH))]
        else:
            deltaPitch = [0, 0, 0, 0, 0, 0]

        if any(result != 1 for result in pitchresult):
            flag_pitchfuryou = 1
        if any(id != 0 for id in customid):
            flag_clip_hanire = 1
        if any(result != 1 for result in pitchresult) or any(id != 0 for id in detectedid) or any(id != 0 for id in customid):
            status = "NG"
        else:
            status = "OK"

    # elif partid == "RH":
    #     pitchresult = check_tolerance(pitchSpecRH, totalLengthSpec, pitchTolerance, totalLengthTolerance, detectedPitch, total_length)
        
    #     if len(detectedPitch) == 6:
    #         deltaPitch = [detectedPitch[i] - pitchSpecRH[i] for i in range(len(pitchSpecRH))]
    #     else:
    #         deltaPitch = [0, 0, 0, 0, 0, 0]

    #     if any(result != 1 for result in pitchresult):
    #         flag_pitchfuryou = 1
    #     if any(id != 0 for id in customid):
    #         flag_clip_hanire = 1
    #     if any(result != 1 for result in pitchresult) or any(id != 1 for id in detectedid) or any(id != 0 for id in customid):
    #         status = "NG"
    #     else:
    #         status = "OK"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(img, xy_pairs, pitchresult, endoffset_y)


    # play_sound(status)
    img = draw_status_text(img, status)

    img = draw_flag_status(img, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire)


    return img, pitchresult, detectedPitch, deltaPitch, flag_clip_hanire

def play_sound(status):
    if status == "OK":
        ok_sound.play()
    elif status == "NG":
        ng_sound.play()

def get_center(bbox):
    center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
    center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
    return center_x, center_y

def print_bbox_structure(bbox):
    print(f"BoundingBox attributes: {dir(bbox)}")

def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, 40)
    color=(200,10,10)
    if flag_pitchfuryou == 1:
        draw.text((120, 10), u"クリップピッチ不良", font=font, fill=color)  
    if flag_clip_furyou == 1:
        draw.text((120, 60), u"クリップ類不良", font=font, fill=color)  
    if flag_clip_hanire == 1:
        draw.text((120, 110), u"クリップ半入れ", font=font, fill=color)
    
    # Convert back to BGR for OpenCV compatibility
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image


def draw_pitch_line(image, xy_pairs, pitchresult, endoffset_y=0):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for i in range(len(xy_pairs) - 1):
            if i < len(pitchresult) and pitchresult[i] is not None:
                if pitchresult[i] == 1:
                    lineColor = (0, 255, 0)
                else:
                    lineColor = (255, 0, 0)

            # if i == 0:
            #     offsetpos_ = (xy_pairs[i+1][0], xy_pairs[i+1][1] + endoffset_y)
            #     cv2.line(image, xy_pairs[i], offsetpos_, lineColor, 5)
            #     cv2.circle(image, xy_pairs[i], 4, (255, 0, 0), -1)
            # elif i == len(xy_pairs) - 2:
            #     offsetpos_ = (xy_pairs[i][0], xy_pairs[i][1] + endoffset_y)
            #     cv2.line(image, offsetpos_, xy_pairs[i+1], lineColor, 5)
            #     cv2.circle(image, xy_pairs[i+1], 4, (255, 0, 0), -1)
            # else:
                cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, 4)

    return None


#add "OK" and "NG"
def draw_status_text(image, status):
    # Define the position for the text: Center top of the image
    center_x = image.shape[1] // 2
    top_y = 50  # Adjust this value to change the vertical position

    # Text properties
    font_scale = 5.0  # Increased font scale for bigger text
    font_thickness = 8  # Increased font thickness for bolder text
    outline_thickness = font_thickness + 2  # Slightly thicker for the outline
    text_color = (255, 0, 0) if status == "NG" else (0, 255, 0)  # Red for NG, Green for OK
    outline_color = (0, 0, 0)  # Black for the outline

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = center_x - text_size[0] // 2
    text_y = top_y + text_size[1]

    # Draw the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness)

    # Draw the text over the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return image


def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
    
    result = [0] * len(pitchSpec)
    
    for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
        if abs(spec - detected) <= pitchTolerance[i]:
            result[i] = 1
    print(checkedPitchResult, result)
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel

def find_edge_point(image, center, direction="None", offsetval = 0):
    x, y = center[0], center[1]
    blur = 0
    brightness = 0
    contrast = 1
    lower_canny = 100
    upper_canny = 200

    #read canny value from /aikensa/param/canyparams.yaml if exist
    if os.path.exists("./aikensa/param/cannyparams.yaml"):
        with open("./aikensa/param/cannyparams.yaml") as f:
            cannyparams = yaml.load(f, Loader=yaml.FullLoader)
            blur = cannyparams["blur"]
            brightness = cannyparams["brightness"]
            contrast = cannyparams["contrast"]
            lower_canny = cannyparams["lower_canny"]
            upper_canny = cannyparams["upper_canny"]

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"gray_image.jpg_{direction}", gray_image)
    # cv2.imwrite(f"blurred_image.jpg_{direction}", blurred_image)
    # cv2.imwrite(f"canny_debug.jpg_{direction}", canny_img)


    while 0 <= x < image.shape[1]:
        if canny_img[y + offsetval, x] == 255:  # Found an edge
            # cv2.line(image, (center[0], center[1] + offsetval), (x, y + offsetval), (0, 255, 0), 1)
            # color = (0, 0, 255) if direction == "left" else (255, 0, 0)
            # cv2.circle(image, (x, y + offsetval), 5, color, -1)
            return x, y + offsetval
        
        x = x - 1 if direction == "left" else x + 1

    return None

def drawcircle(image, pos, class_id): #for ire and hanire
    #draw either green or red circle depends on the detection
    if class_id == 0:
        color = (60, 200, 60)
    elif class_id == 1:
        color = (60, 60, 200)
    #check if pos is tupple
    pos = (int(pos[0]), int(pos[1]))

    cv2.circle(img=image, center=pos, radius=30, color=color, thickness=2, lineType=cv2.LINE_8)

    return image

def drawbox(image, pos, length):
    pos = (pos[0], pos[1])
    rectangle_bgr = (255, 255, 255)
    font_scale = 1.7
    font_thickness = 4
    (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    top_left_x = pos[0] - text_width // 2 - 8
    top_left_y = pos[1] - text_height // 2 - 8 - text_offset
    bottom_right_x = pos[0] + text_width // 2 + 8
    bottom_right_y = pos[1] + text_height // 2 + 8 - text_offset
    
    cv2.rectangle(image, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y),
                  rectangle_bgr, -1)
    
    return image

def drawtext(image, pos, length):
    pos = (pos[0], pos[1])
    font_scale = 1.7
    font_thickness = 6
    text = f"{length:.1f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2 - text_offset
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
    return image

def calclength(p1, p2):
    length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return length

def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=1):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1, y1 = int(x - w // 2), int(y - h // 2)
    x2, y2 = int(x + w // 2), int(y + h // 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center_x, center_y = x, y
    return (center_x, center_y)

class BoundingBox:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

class PredictionScore:
    def __init__(self, value):
        self.value = value

class Category:
    def __init__(self, id, name):
        self.id = id
        self.name = name

class ObjectPrediction:
    def __init__(self, bbox, score, category):
        self.bbox = bbox
        self.score = score
        self.category = category