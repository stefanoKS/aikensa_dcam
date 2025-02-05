import stat
import numpy as np
import cv2
import math
from torch import normal
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

pitchSpec_050P = [85, 87, 98, 98, 78, 113, 103]
pitchSpec_040P = [103, 113, 78, 98, 98, 87, 85]
pitchSpec_090P = [85, 87, 98, 98, 78, 61, 52, 38, 37, 28]
pitchSpec_080P = [28, 37, 38, 52, 61, 78, 98, 98, 87, 85]

pitchSpec_050PKENGEN = [85, 87, 98, 98, 78, 113, 103]
pitchSpec_040PKENGEN = [103, 113, 78, 98, 98, 87, 85]
pitchSpec_090PKENGEN = [85, 87, 98, 98, 78, 61, 52, 38, 37, 28, 14]
pitchSpec_080PKENGEN = [28, 37, 38, 52, 61, 78, 98, 98, 87, 85, 14]

pitchSpec_050PCLIPSOUNYUUKI = [87, 98, 98, 78, 113, 103]
pitchSpec_040PCLIPSOUNYUUKI = [103, 113, 78, 98, 98, 87]
pitchSpec_090PCLIPSOUNYUUKI = [87, 98, 98, 78, 61, 52, 38, 37, 28]
pitchSpec_080PCLIPSOUNYUUKI = [28, 37, 38, 52, 61, 78, 98, 98, 87]

pitchTolerance_050P = [2.0, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
pitchTolerance_040P = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0]
pitchTolerance_090P = [2.0, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
pitchTolerance_080P = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 2.0, 1.7]

pitchTolerance_050PCLIPSOUNYUUKI = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
pitchTolerance_040PCLIPSOUNYUUKI = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
pitchTolerance_090PCLIPSOUNYUUKI = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
pitchTolerance_080PCLIPSOUNYUUKI = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]

clipSpec_050P = [2, 1, 0, 0, 0, 0, 3, 3, 0, 1] #white is 0, brown is 1, yellow is 2, orange is 3
clipSpec_040P = [0, 1, 3, 3, 1, 1, 1, 1, 0, 2]
clipSpec_090P = [2, 1, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1]
clipSpec_080P = [0, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 0, 2]

clipSpec_050PCLIPSOUNYUUKI = [2, 1, 0, 0, 0, 0, 3, 3, 0, 1] #white is 0, brown is 1, yellow is 2, orange is 3
clipSpec_040PCLIPSOUNYUUKI = [0, 1, 3, 3, 1, 1, 1, 1, 0, 2]
clipSpec_090PCLIPSOUNYUUKI = [2, 1, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 1]
clipSpec_080PCLIPSOUNYUUKI = [0, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 0, 2]

pitchSpec_Katabu = [14]
pitchTolerance_Katabu = [1.7]

color = (0, 255, 0)
text_offset = 40
endoffset_y = 0
bbox_offset = 10

# segmentation_width = 1640

pixelMultiplier = 0.1607
pixelMultiplier_katabumarking = 0.1607


def partcheck(image, img_katabumarking, sahi_predictionList, katabumarking_detection, partname):
        
    print(f"Partname: {partname}")


    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    katabumarking_lengths = []

    detectedid = []
    idSpec = []

    measuredPitch = []
    resultPitch = []
    deltaPitch = []

    resultid = []

    detectedposX = []
    detectedposY = []

    detectedposX_katabumarking = []
    detectedposY_katabumarking = []

    detectedWidth = []

    prev_center = None
    prev_center_katabumarking = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0

    leftmostPitch = 0
    rightmostPitch = 0

    status = "OK"
    print_status = ""

    combined_infer_mask = None

    if partname == "P82833W050P":
        pitchSpec = pitchSpec_050P
        tolerance_pitch = pitchTolerance_050P
        idSpec = clipSpec_050P

    elif partname == "P82832W040P":
        pitchSpec = pitchSpec_040P
        tolerance_pitch = pitchTolerance_040P
        idSpec = clipSpec_040P

    elif partname == "P82833W090P":
        pitchSpec = pitchSpec_090P
        tolerance_pitch = pitchTolerance_090P
        idSpec = clipSpec_090P

    elif partname == "P82832W080P":
        pitchSpec = pitchSpec_080P
        tolerance_pitch = pitchTolerance_080P
        idSpec = clipSpec_080P



    elif partname == "P82833W050PKENGEN":
        pitchSpec = pitchSpec_050PKENGEN
        tolerance_pitch = pitchTolerance_050P
        idSpec = clipSpec_050P

    elif partname == "P82832W040PKENGEN":
        pitchSpec = pitchSpec_040PKENGEN
        tolerance_pitch = pitchTolerance_040P
        idSpec = clipSpec_040P

    elif partname == "P82833W090PKENGEN":
        pitchSpec = pitchSpec_090PKENGEN
        tolerance_pitch = pitchTolerance_090P
        idSpec = clipSpec_090P
    
    elif partname == "P82832W080PKENGEN":
        pitchSpec = pitchSpec_080PKENGEN
        tolerance_pitch = pitchTolerance_080P
        idSpec = clipSpec_080P



    elif partname == "P82833W050PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_050PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_050PCLIPSOUNYUUKI
        idSpec = clipSpec_050P

    elif partname == "P82832W040PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_040PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_040PCLIPSOUNYUUKI
        idSpec = clipSpec_040P

    elif partname == "P82833W090PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_090PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_090PCLIPSOUNYUUKI
        idSpec = clipSpec_090P

    elif partname == "P82832W080PCLIPSOUNYUUKI":
        pitchSpec = pitchSpec_080PCLIPSOUNYUUKI
        tolerance_pitch = pitchTolerance_080PCLIPSOUNYUUKI
        idSpec = clipSpec_080P




    #KATABU MARKING DETECTION
    #class 0 is for clip, class 1 is for katabu marking
    for r in katabumarking_detection:
        for box in r.boxes:
            x_marking, y_marking = float(box.xywh[0][0].cpu()), float(box.xywh[0][1].cpu())
            w_marking, h_marking = float(box.xywh[0][2].cpu()), float(box.xywh[0][3].cpu())
            class_id_marking = int(box.cls.cpu())

            if class_id_marking == 0:
                color = (0, 255, 0)
            elif class_id_marking == 1:
                color = (100, 100, 200)

            center_katabummarking = draw_bounding_box(img_katabumarking, 
                                       x_marking, y_marking, 
                                       w_marking, h_marking, 
                                       [img_katabumarking.shape[1], img_katabumarking.shape[0]], color=color,
                                       bbox_offset=3, thickness=2)

            if class_id_marking == 1:
                if partname in ["P82833W050P", "P82833W090P", "P82833W050PKENGEN", "P82833W090PKENGEN"]:
                    center_katabummarking = (int(x_marking - w_marking/2), int(y_marking))
                elif partname in ["P82832W040P", "P82832W080P", "P82832W040PKENGEN", "P82832W080PKEGEN"]:
                    center_katabummarking = (int(x_marking + w_marking/2), int(y_marking))
            
            if prev_center_katabumarking is not None:
                length = calclength(prev_center_katabumarking, center_katabummarking)*pixelMultiplier_katabumarking
                katabumarking_lengths.append(length)
                line_center = ((prev_center_katabumarking[0] + center_katabummarking[0]) // 2, (prev_center_katabumarking[1] + center_katabummarking[1]) // 2)
                img_katabumarking = drawbox(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)
                img_katabumarking = drawtext(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)

            prev_center_katabumarking = center_katabummarking

            detectedposX_katabumarking.append(center_katabummarking[0])
            detectedposY_katabumarking.append(center_katabummarking[1])
  
        katabupitchresult = check_tolerance(katabumarking_lengths, pitchSpec_Katabu, pitchTolerance_Katabu)

        xy_pairs_katabumarking = list(zip(detectedposX_katabumarking, detectedposY_katabumarking))
        draw_pitch_line(img_katabumarking, xy_pairs_katabumarking, katabupitchresult, thickness=2)

        #pick only the first element if array consists of more than 1 element -> detection POKAYOKE (if detection is not that great)
        if len(katabumarking_lengths) > 1:
            katabumarking_lengths = katabumarking_lengths[:1]
        #since there is only one katabu marking, we can just use the first element -> detection POKAYOKE (if detection is not that great)
        print(f"Katabu Marking Length: {katabumarking_lengths}")

        #if katabumarking_lengths is empty, then it is NG
        if katabumarking_lengths == []:
            status = "NG"
            print_status = print_status + "型部マーキング認識不良"
            print(f"Status:{print_status}")
            measuredPitch = [0] * len(pitchSpec)
            resultPitch = [0] * len(pitchSpec)
            resultid = [0] * len(idSpec)
            image = draw_status_text_PIL(image, status, print_status, size = "normal")

            return image, img_katabumarking, measuredPitch, resultPitch, resultid, status
    
        
    for i, detection in enumerate(sorted_detections):
        detectedid.append(detection.category.id)
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detectedposX.append(x)
        detectedposY.append(y)
        detectedWidth.append(w)

        center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

        if prev_center is not None:
            length = calclength(prev_center, center)*pixelMultiplier
            measuredPitch.append(length)
            line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
            image = drawbox(image, line_center, length, font_scale=2.0, offset=40, font_thickness=2)
            image = drawtext(image, line_center, length, font_scale=2.0, offset=40, font_thickness=2)
        prev_center = center

    #POP The first and last element for the KENGEN and normal
    if partname in ["P82833W050P", "P82832W040P", "P82833W090P", "P82832W080P", "P82833W050PKENGEN", "P82832W040PKENGEN", "P82833W090PKENGEN", "P82832W080PKENGEN"]:
        #Pop the first and last element
        detectedposX.pop(0)
        detectedposX.pop(-1)
        detectedposY.pop(0)
        detectedposY.pop(-1)
        detectedWidth.pop(0)
        detectedWidth.pop(-1)
        measuredPitch.pop(0)
        measuredPitch.pop(-1)

        print("Element Popped")

        
    if detectedid != idSpec:
        status = "NG"
        print_status = print_status + "NG クリップ入れ間違い"
        # print(f"Status:{print_status}")
        measuredPitch = [0] * len(pitchSpec)
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        image = draw_status_text_PIL(image, status, print_status, size = "normal")
        cv2.imwrite("test.png", image)
        return image, img_katabumarking, measuredPitch, resultPitch, resultid, status
    
    # if len(measuredPitch) != len(pitchSpec):
    #     status = "NG"
    #     print_status = "NG クリップ数"
    #     print(f"Status:{print_status}")
    #     measuredPitch = [0] * len(pitchSpec)
    #     resultPitch = [0] * len(pitchSpec)
    #     resultid = [0] * len(idSpec)
    #     draw_status_text_PIL(image, status, print_status, size = "normal")

    #     return image, img_katabumarking, measuredPitch, resultPitch, resultid, status
    
    if katabumarking_lengths is not None:
        if katabumarking_lengths and katabumarking_lengths[0] != 0:
            measuredPitch.append(round(katabumarking_lengths[0], 1))

    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    #print measured pitch, print ID
    # print(f"Spec:,{pitchSpec}")

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)
        # print(f"Result Pitch: {resultPitch}")
        # print(f"Result ID: {resultid}")

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)

    if any(result != 1 for result in resultPitch):
        print_status = print_status + " ピッチ不良"
        status = "NG"
        # print(f"Status:{print_status}")
        image  = draw_status_text_PIL(image, status, print_status, size = "normal")

    print(f"Measured Pitch: {measuredPitch}")
    print(f"Detected ID: {detectedid}")
    print(f"Result Pitch: {resultPitch}")

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)

    if status == "OK":
        image = draw_status_text(image, status, size = "normal")
    
    return image, img_katabumarking, measuredPitch, resultPitch, resultid, status


def draw_status_text_PIL(image, status, print_status, size = "normal"):

    if size == "large":
        font_scale = 50.0
    if size == "normal":
        font_scale = 90.0
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

    draw.text((300, 5), status, font=font, fill=color)  
    draw.text((300, 100), print_status, font=font, fill=color)
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", image)
    return image


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
    # Define the position for the text: Center top of the image
    center_x = image.shape[1] // 2
    if size == "normal":
        top_y = 50  # Adjust this value to change the vertical position
        font_scale = 5.0  # Increased font scale for bigger text

    elif size == "small":
        top_y = 10
        font_scale = 2.0  # Increased font scale for bigger text
    

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
    blur = 11
    brightness = 0
    contrast = 3.0
    lower_canny = 15
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