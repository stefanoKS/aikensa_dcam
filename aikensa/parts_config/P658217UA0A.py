from calendar import c
from curses import raw
import re
import stat
from unittest import result
from matplotlib.pyplot import flag
from networkx import draw
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from aikensa.scripts.scripts_img_processing import map_keypoint_xcrop_to_original


pygame.mixer.init()
ok_sound = pygame.mixer.Sound("aikensa/sound/positive_interface.wav") 
ok_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-remove-2576.wav")
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")  
ng_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-system-beep-buzzer-fail-2964.wav")
kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

pitchSpecLH = [15, 88, 117, 76, 15, 311]
pitchSpecRH = [15, 76, 88, 117, 15, 311]

idSpecLH = [0, 0, 0, 0]
idSpecRH = [1, 1, 1, 1]

pitchTolerance = [3.0, 2.0, 2.0, 2.0, 3.0, 10.0]

color = (0, 255, 0)
linecolor = (20,120,120)
text_offset = 40
endoffset_y = 0
bbox_offset = 1

pixelMultiplier = 0.1585

segmentation_pixel_start = 256
segmentation_pixel_finish = 768
segmentation_width = segmentation_pixel_finish - segmentation_pixel_start
detected_cropped_size = 84

border_width = 512

def partcheck(image, sahi_predictionList, leftSegmentation, rightSegmentation, keypointLeft, keypointRight, widgetNumber, YoloHanireModel):

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    detectedid = []

    measuredPitch = []
    resultPitch = []
    deltaPitch = []

    resultid = []

    detectedposX = []
    detectedposY = []

    detectedWidth = []

    prev_center = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0
    leftmostPitch = 0
    rightmostPitch = 0

    leftmostPointX = 0
    leftmostPointY = 0
    rightmostPointX = 0
    rightmostPointY = 0

    flag_muki = 0 #whether the hole is facing the right direction
    flag_hanire = 0 #whether the clip is half inserted

    status = "OK"
    print_status = ""

    combined_lmask = None
    ngreason = ""

    raw_image = image.copy()

    if widgetNumber == 9:
        pitchSpec = pitchSpecLH
        idSpec = idSpecLH
    
    if widgetNumber == 10:
        pitchSpec = pitchSpecRH
        idSpec = idSpecRH

    tolerance_pitch = pitchTolerance

    combined_lmask = None
    for lm in leftSegmentation:
        if lm.masks is not None:
            orig_shape = (image.shape[0] + border_width * 2 , segmentation_width + border_width * 2 )
            segmentation_xyn = lm.masks.xyn
            lmask = create_masks(segmentation_xyn, orig_shape)
            if combined_lmask is None:
                combined_lmask = np.zeros_like(lmask)
            combined_lmask = cv2.bitwise_or(combined_lmask, lmask)
            #resize back to original size
            combined_lmask = combined_lmask[border_width:-border_width, border_width:-border_width]
            combined_lmask = cv2.resize(combined_lmask, (segmentation_width, image.shape[0]))
            # cv2.imwrite("leftmask.jpg", combined_lmask)


        if lm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, deltaPitch, status, ngreason
        
    combined_rmask = None
    for rm in rightSegmentation:
        if rm.masks is not None:
            orig_shape = (image.shape[0] + border_width * 2 , segmentation_width + border_width * 2 )
            segmentation_xyn = rm.masks.xyn
            rmask = create_masks(segmentation_xyn, orig_shape)
            if combined_rmask is None:
                combined_rmask = np.zeros_like(rmask)
            combined_rmask = cv2.bitwise_or(combined_rmask, rmask)
            #remove the pad from the image (pad size is 200 around the image)
            combined_rmask = combined_rmask[border_width:-border_width, border_width:-border_width]
            combined_rmask = cv2.resize(combined_rmask, (segmentation_width, image.shape[0]))
            # cv2.imwrite("rightmask.jpg", combined_rmask)
        if rm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="small")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, deltaPitch, status, ngreason


    combined_mask = np.zeros_like(image[:, :, 0])  # Single-channel black mask
    if combined_lmask is not None and combined_rmask is not None:
        combined_mask[:, segmentation_pixel_start:segmentation_pixel_finish] = combined_lmask
        combined_mask[:, -segmentation_pixel_finish:-segmentation_pixel_start] = combined_rmask

    # cv2.imwrite("combined_mask.jpg", combined_mask)

    for keypoint in keypointLeft:
        if keypoint.keypoints.xy is None or keypoint.keypoints.xy.shape[0] == 0 or keypoint.keypoints.xy.shape[1] == 0:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, resultid, status, ngreason

        xy = keypoint.keypoints.xy
        x_pos, y_pos = xy[0, 0].tolist()
        # print ("Keypoint left xy: ", xy)
        leftmostPointX, leftmostPointY = map_keypoint_xcrop_to_original(x_start=segmentation_pixel_start, kpt_xy_crop=(x_pos, y_pos), img_width=image.shape[1])
        print ("Mapped Keypoint left xy to original: ", (leftmostPointX, leftmostPointY))
        

    for keypoint in keypointRight:
        if keypoint.keypoints.xy is None or keypoint.keypoints.xy.shape[0] == 0 or keypoint.keypoints.xy.shape[1] == 0:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, resultid, status, ngreason
        
        xy = keypoint.keypoints.xy
        # print ("Keypoint right xy: ", xy)
        x_pos, y_pos = xy[0, 0].tolist()
        rightmostPointX, rightmostPointY = map_keypoint_xcrop_to_original(x_start=-segmentation_pixel_finish, kpt_xy_crop=(x_pos, y_pos), img_width=image.shape[1])
        print ("Mapped Keypoint right xy to original: ", (rightmostPointX, rightmostPointY))


    for i, detection in enumerate(sorted_detections):

        if detection.category.id != 2:

            detectedid.append(detection.category.id)

            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)

            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

            hanireResult = check_hanire(raw_image, x, y, YoloHanireModel, border_width)
            print (f"Hanire result: {hanireResult}")
            if hanireResult == 0:
                draw_redCircle(image, x, y, w, h, [image.shape[1], image.shape[0]], thickness=6, bbox_offset=20)
                flag_hanire = 1

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                measuredPitch.append(length)
            prev_center = center

        
        if detection.category.id == 2:
            flag_muki = 1

    print("Detected IDs: ", detectedid)


    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        # leftmostWidth = detectedWidth[0]
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        # rightmostWidth = detectedWidth[-1]
      
        # Positive Yoffsetval means going down, negative means going up
        # left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = 0, Xoffsetval = 0)
        # right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = 0, Xoffsetval = 0)

        # left_edge =  leftmostPointX, leftmostPointY
        # right_edge = rightmostPointX, rightmostPointY
        
        left_edge =  leftmostPointX, detectedposY[0]
        right_edge = rightmostPointX, detectedposY[-1]

        leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
        rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

        #append the leftmost and rightmost pitch to the measuredPitch
        measuredPitch.insert(0, leftmostPitch)
        measuredPitch.append(rightmostPitch)
        #Reappend the leftmostcetner and rightmostcenter to the detectedposX and detectedposY
        detectedposX.insert(0, left_edge[0])
        detectedposY.insert(0, left_edge[1])
        detectedposX.append(right_edge[0])
        detectedposY.append(right_edge[1])


    #add total length
    #round the value to 1 decimal
    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)

    if flag_muki == 0:
        status = "NG"
        ngreason = "HOLE NOT FACING RIGHT DIRECTION"
        print_status = "エアー穴の向き不良"
        image = draw_status_text_PIL(image, status, print_status, size="normal")

        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))

        return image, measuredPitch, resultPitch, resultid, status, ngreason
    
    if flag_hanire == 1:
        status = "NG"
        ngreason = "CLIP HALF INSERTED"
        print_status = "クリップ半入れ不良"
        image = draw_status_text_PIL(image, status, print_status, size="normal")

        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))

        return image, measuredPitch, resultPitch, resultid, status, ngreason


    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))
        ngreason = "NUMBER OF CLIP MISMATCH"
        status = "NG"
        print_status = "クリップ数不足"

        image = draw_status_text_PIL(image, status, print_status, size="normal")

        return image, measuredPitch, resultPitch, resultid, status, ngreason

    if any(result != 1 for result in resultid):
        resultPitch = [0] * len (pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))
        status = "NG"
        ngreason = "CLIP ID NG"
        print_status = "クリップ類不良"
        
        image = draw_status_text_PIL(image, status, print_status, size="normal")

        return image, measuredPitch, resultPitch, resultid, status, ngreason

    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ngreason = "CLIP PITCH NG"
        print_status = "クリップピッチ不良"

    print("Resultpitch: ", resultPitch)
    print("Resultid: ", resultid)
    print("MeasuredPitch: ", measuredPitch)

    # if any(result != 1 for result in resultid):
    #     flag_clip_furyou = 1
    #     status = "NG"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=4)

    image = draw_status_text_PIL(image, status, print_status, size="normal")
    
    return image, measuredPitch, resultPitch, resultid, status, ngreason


def check_hanire(image, x, y, YoloHanireModel, border_width):
    """
    Check if the clip is half inserted by using the YoloHanireModel.
    Returns 1 if half inserted, 0 otherwise.
    """
    h_img, w_img = image.shape[:2]

    # 1) skip if image smaller than crop size
    if h_img < detected_cropped_size or w_img < detected_cropped_size:
        return 0


    half = detected_cropped_size // 2
    x0, y0 = x - half, y - half
    x1, y1 = x + half, y + half
    #convert to int
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
    if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
        return 0

    crop = image[y0:y1, x0:x1]

    hanire = YoloHanireModel(crop, stream=True, verbose=False)
    hanire = list(hanire)[0].probs.data.argmax().item()
    # Predict using the YoloHanireModel
    return hanire  # No half insertion detected



def extend_line(p1, p2):
    """Calculate the slope and intercept for a line that goes through p1 and p2"""
    x1, y1 = p1
    x2, y2 = p2

    # Parametric line formula: L(t) = (1 - t) * p1 + t * p2
    # Slope (m) of the line
    if x2 - x1 != 0:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    else:
        # Vertical line
        return None, None

def closest_point_on_line(p1, p2, p):
    """Calculate the closest point from point p to the line between p1 and p2"""
    # Vector representation of line and point
    line_vec = np.array(p2) - np.array(p1)
    p_vec = np.array(p) - np.array(p1)

    # Project p_vec onto line_vec to find the closest point
    line_len_squared = np.dot(line_vec, line_vec)
    if line_len_squared == 0:
        return p1  # p1 and p2 are the same point

    # Parametric distance along the line
    t = np.dot(p_vec, line_vec) / line_len_squared
    # Find the closest point
    closest_point = np.array(p1) + t * line_vec
    return closest_point

def create_masks(segmentation_result, orig_shape):
    mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
    for polygon in segmentation_result:
        polygon = np.array([[int(x * orig_shape[1]), int(y * orig_shape[0])] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def play_sound(status):
    if status == "OK":
        # ok_sound.play()
        ok_sound_v2.play()
    elif status == "NG":
        # ng_sound.play()
        ng_sound_v2.play()

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

def check_id(detectedid, idSpec):
    result = [0] * len(idSpec)
    for i, (spec, detected) in enumerate(zip(idSpec, detectedid)):
        if spec == detected:
            result[i] = 1
    return result

def draw_pitch_line(image, xy_pairs, pitchresult, thickness=2):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for i in range(len(xy_pairs) - 1):
            if i < len(pitchresult) and pitchresult[i] is not None:
                if pitchresult[i] == 1:
                    lineColor = (0, 255, 0)
                else:
                    lineColor = (255, 0, 0)

                cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, thickness)
                

    return None


#add "OK" and "NG"
def draw_status_text(image, status, size = "normal"):
    center_x = image.shape[1] // 2
    if size == "normal":
        top_y = 50 
        font_scale = 5.0 

    elif size == "small":
        top_y = 10
        font_scale = 2.0  
    

    # Text properties
    
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

    
def draw_status_text_PIL(image, status, print_status, size = "normal"):

    if size == "large":
        font_scale = 130.0
    if size == "normal":
        font_scale = 100.0
    elif size == "small":
        font_scale = 50.0

    if status == "OK":
        color = (10, 210, 60)

    elif status == "NG":
        color = (200, 30, 50)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, font_scale)

    draw.text((120, 5), status, font=font, fill=color)  
    draw.text((120, 100), print_status, font=font, fill=color)
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image


def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
    result = [0] * len(pitchSpec)
    for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
        if abs(spec - detected) <= pitchTolerance[i]:
            result[i] = 1
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel


def find_edge_point_mask(image, mask, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]

    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if mask[int(y + Yoffsetval), int(x - Xoffsetval)] == 0:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if mask[int(y + Yoffsetval), int(x + Xoffsetval)] == 0:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided


def find_edge_point(image, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]
    blur = 9
    brightness = 0
    contrast = 2.0
    lower_canny = 10
    upper_canny = 110

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"1adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"2gray_image_{direction}.jpg", gray_image)
    # cv2.imwrite(f"3blurred_image_{direction}.jpg", blurred_image)
    # cv2.imwrite(f"4canny_debug_{direction}.jpg", canny_img)
    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if canny_img[int(y + Yoffsetval), int(x - Xoffsetval)] == 255:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if canny_img[int(y + Yoffsetval), int(x + Xoffsetval)] == 255:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided

def drawcircle(image, pos, class_id, radius=10): #for ire and hanire
    #draw either green or red circle depends on the detection
    if class_id == 1:
        color = (60, 200, 60)
    elif class_id == 0:
        color = (60, 60, 200)
    #check if pos is tupple
    pos = (int(pos[0]), int(pos[1]))

    cv2.circle(img=image, center=pos, radius=radius, color=color, thickness=2, lineType=cv2.LINE_8)

    return image

def drawbox(image, pos, length, offset = text_offset, font_scale=1.7, font_thickness=4):
    pos = (pos[0], pos[1])
    rectangle_bgr = (255, 255, 255)
    (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    top_left_x = pos[0] - text_width // 2 - 8
    top_left_y = pos[1] - text_height // 2 - 8 - offset
    bottom_right_x = pos[0] + text_width // 2 + 8
    bottom_right_y = pos[1] + text_height // 2 + 8 - offset
    
    cv2.rectangle(image, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y),
                  rectangle_bgr, -1)
    
    return image

def drawtext(image, pos, length, font_scale=1.7, offset = text_offset, font_thickness=6):
    pos = (pos[0], pos[1])
    font_scale = font_scale
    text = f"{length:.1f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2 - offset
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
    return image

def calclength(p1, p2):
    length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return length

def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=3, bbox_offset=bbox_offset):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1, y1 = int(x - w // 2) - bbox_offset, int(y - h // 2) - bbox_offset
    x2, y2 = int(x + w // 2) + bbox_offset, int(y + h // 2) + bbox_offset
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center_x, center_y = x, y
    return (center_x, center_y)

def draw_redCircle(image, x, y, w, h, img_size, thickness=3, bbox_offset=8):
    color = (10, 10, 255)  # Red color
    #if the box extends outside the image, adjust it
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)  
    radius = int((w + h) / 4) + bbox_offset  # Calculate radius based on width and height
    cv2.circle(image, (x, y), radius, color, thickness)