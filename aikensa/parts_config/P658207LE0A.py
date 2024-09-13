import stat
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
ok_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-remove-2576.wav")
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")  
ng_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-system-beep-buzzer-fail-2964.wav")
kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

pitchSpec = [13, 82, 82, 82, 13, 2, 2]
idSpec = [ 0, 0, 0, 0]
tolerance_pitch = [3.0, 1.5, 1.5, 1.5, 3.0, 0.8, 0.8]

color = (0, 255, 0)
linecolor = (20,120,120)
text_offset = 40
endoffset_y = 0
bbox_offset = 1

pixelMultiplier = 0.1592


def partcheck(image, clip_detection_result, segmentation_result):

    # print(clip_detection_result)
    

    detectedid = []

    measuredPitch = []
    resultPitch = []
    deltaPitch = []

    resultid = []

    detectedposX = []
    detectedposY = []

    detectedWidth = []
    detectedHeight = []

    prev_center = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0

    leftmostPitch = 0
    rightmostPitch = 0

    status = "OK"
    print_status = ""

    for r in clip_detection_result:

        sorted_boxes = sorted(r.boxes, key=lambda box: float(box.xywh[0][0].cpu()))

        for box in sorted_boxes:

            detectedid.append(box.cls)

            x, y = float(box.xywh[0][0].cpu()), float(box.xywh[0][1].cpu())
            w, h  = float(box.xywh[0][2].cpu()), float(box.xywh[0][3].cpu())

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)
            detectedHeight.append(h)

            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)


    combined_mask = None

    for m in segmentation_result:
        #use image size
        orig_shape = (image.shape[0], image.shape[1])
        segmentation_xyn = m.masks.xyn
        mask = create_masks(segmentation_xyn, orig_shape)
        if combined_mask is None:
            combined_mask = np.zeros_like(mask)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        #draw the mask as overlay
        image_overlay = image.copy()
        image_overlay = cv2.addWeighted(image, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.5, 0)
        cv2.imwrite("mask_overlay.jpg", image_overlay)


    if len(detectedid) < 4:
        print_status = print_status + " クリップ数不足 "
        status = "NG"
        resultPitch = [0] * len(pitchSpec)
        measuredPitch = [0] * len(pitchSpec)

    if len(detectedid) > 4:
        print_status = print_status + " クリップ数過多 "
        status = "NG"
        resultPitch = [0] * len(pitchSpec)
        measuredPitch = [0] * len(pitchSpec)

    if len(detectedid) == 4:

        point1 = (detectedposX[0], detectedposY[0])
        point2 = (detectedposX[1], detectedposY[1])
        point3 = (detectedposX[2], detectedposY[2])
        point4 = (detectedposX[3], detectedposY[3])

        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]), int(point2[1]))
        point3 = (int(point3[0]), int(point3[1]))
        point4 = (int(point4[0]), int(point4[1]))

        slope, intercept = extend_line(point2, point3)

        closest_point_to_1 = closest_point_on_line(point2, point3, point1)
        closest_point_to_4 = closest_point_on_line(point2, point3, point4)

        closest_point_to_1 = (int(closest_point_to_1[0]), int(closest_point_to_1[1]))
        closest_point_to_4 = (int(closest_point_to_4[0]), int(closest_point_to_4[1]))

        initialposX = detectedposX.copy()
        initialposY = detectedposY.copy() 

        detectedposX[0] = closest_point_to_1[0]
        detectedposY[0] = closest_point_to_1[1]
        detectedposX[-1] = closest_point_to_4[0]
        detectedposY[-1] = closest_point_to_4[1]

        leftmostCenter = (detectedposX[0], detectedposY[0])
        leftmostWidth = detectedWidth[0] # not really useful here since we use mask from inference
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        rightmostWidth = detectedWidth[-1] # not really useful here since we use mask from inference
        adjustment_offset = 2 # not really useful here since we use mask from inference
        left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = 0, Xoffsetval = leftmostWidth + adjustment_offset)
        right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = 0, Xoffsetval = rightmostWidth + adjustment_offset)


        detectedposX.insert(0, left_edge[0])
        detectedposY.insert(0, left_edge[1])
        detectedposX.append(right_edge[0])
        detectedposY.append(right_edge[1])

        for i in range(len(detectedposX) - 1):
            measuredPitch.append(calclength((detectedposX[i], detectedposY[i]), (detectedposX[i+1], detectedposY[i+1])) * pixelMultiplier)

            if abs(measuredPitch[i] - pitchSpec[i]) < tolerance_pitch[i]:
                linecolor = (0, 255, 0)
                linethickness = 2
            else:
                linecolor = (0, 0, 255)
                linethickness = 4

            image = cv2.line(image, (int(detectedposX[i]), int(detectedposY[i])), (int(detectedposX[i+1]), int(detectedposY[i+1])), linecolor, thickness=linethickness)
        

        #Draw the clip 1 and clip 4 line
        measuredPitch.append(calclength((detectedposX[1], detectedposY[1]), (initialposX[0], initialposY[0])) * pixelMultiplier)
        measuredPitch.append(calclength((detectedposX[-2], detectedposY[-2]), (initialposX[-1], initialposY[-1])) * pixelMultiplier)

        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)

        if abs(measuredPitch[-2] - pitchSpec[-2]) < tolerance_pitch[-2]:
            linecolor = (0, 255, 0)
            linethickness = 2
        else:
            linecolor = (0, 0, 255)
            linethickness = 4

        image = cv2.line(image, (int(detectedposX[1]), int(detectedposY[1])), (int(initialposX[0]), int(initialposY[0])), linecolor, thickness=linethickness)

        if abs(measuredPitch[-1] - pitchSpec[-1]) < tolerance_pitch[-1]:
            linecolor = (0, 255, 0)
            linethickness = 2
        else:
            linecolor = (0, 0, 255)
            linethickness = 4
            print_status = print_status + "ピッチ不良 "
            status = "NG"

        image = cv2.line(image, (int(detectedposX[-2]), int(detectedposY[-2])), (int(initialposX[-1]), int(initialposY[-1])), linecolor, thickness=linethickness)

        if initialposY[0] > initialposY[1] or initialposY[3] > initialposY[2]:
            print_status = print_status + "クリップ位置不良 "
            status = "NG"

        if print_status == "":
            status = "OK"

        measuredPitch = [round(pitch, 1) for pitch in measuredPitch]
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)


    #Add print status to the top center of the image
    image = draw_status_text_PIL(image, status, print_status, size="normal")

    return image, measuredPitch, resultPitch, status


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
        color = (10, 60, 260)

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

# class BoundingBox:
#     def __init__(self, minx, miny, maxx, maxy):
#         self.minx = minx
#         self.miny = miny
#         self.maxx = maxx
#         self.maxy = maxy

# class PredictionScore:
#     def __init__(self, value):
#         self.value = value

# class Category:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name

# class ObjectPrediction:
#     def __init__(self, bbox, score, category):
#         self.bbox = bbox
#         self.score = score
#         self.category = category