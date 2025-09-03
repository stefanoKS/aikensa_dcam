from calendar import c
import re
import stat
from turtle import right
from unittest import result
from networkx import draw
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image

from aikensa.scripts.scripts_img_processing import create_masks, draw_bounding_box, get_center, find_edge_point_mask, calclength, check_tolerance, check_id, draw_pitch_line
from aikensa.scripts.scripts_img_processing import draw_status_text_PIL, getMostRightPoint, getMostLeftPoint


pitchSpecLH = [15, 123, 122, 81, 81, 122, 123, 15, 37, 40]
pitchSpecRH = [15, 123, 122, 81, 81, 122, 123, 15, 37, 40]

idSpecLH_with_epto = [0, 0, 2, 0, 0, 0, 0, 0]
idSpecLH           = [0, 0, 0, 0, 0, 0, 0]
idSpecRH_with_epto = [1, 1, 1, 1, 1, 2, 1, 1]
idSpecRH           = [1, 1, 1, 1, 1, 1, 1]

idSpec_epto = 2

tolerance_pitch =       [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2, 4]
tolerance_pitch_LH =    [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2, 4]

epto_width = 40
epto_width_tolerance = 3

epto_clip_pitch = 37
epto_clip_pitch_tolerance = 2

color = (0, 255, 0)
linecolor = (20,120,120)
text_offset = 40
endoffset_y = 0
bbox_offset = 1

pixelMultiplier = 0.1996 #0.1592
pixelMultiplier_eptoLH = 0.21
pixelMultiplier_eptoRH = 0.20

segmentation_pixel_start = 0
segmentation_pixel_finish = 256
segmentation_width = segmentation_pixel_finish - segmentation_pixel_start

border_width = 256


def partcheck(image, sahi_predictionList, leftSegmentation, rightSegmentation, partSide):

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

    status = "OK"
    print_status = ""
    ngreason = ""

    if partSide == "LH":
        pitchSpec = pitchSpecLH
        idSpec = idSpecLH
        idSpec_with_epto = idSpecLH_with_epto
    elif partSide == "RH":
        pitchSpec = pitchSpecRH
        idSpec = idSpecRH
        idSpec_with_epto = idSpecRH_with_epto

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
            image = draw_status_text_PIL(image, status, print_status, size="small")

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
        combined_mask[:, -segmentation_pixel_finish:] = combined_rmask


        # Save combined_mask with incrementing number if file exists
        # base_filename = "combined_mask"
        # ext = ".jpg"
        # filename = f"{base_filename}{ext}"
        # counter = 1
        # while os.path.exists(filename):
        #     filename = f"{base_filename}_{counter}{ext}"
        #     counter += 1
        # cv2.imwrite(filename, combined_mask)

    for i, detection in enumerate(sorted_detections):

        detectedid.append(detection.category.id)
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny



        if detection.category.id == 0:
            #brown for brown clip
            color = (170, 0, 100)
        elif detection.category.id == 1:
            #green for green clip
            color = (10, 200, 10)
        elif detection.category.id == 2:
            #blue for epto tape
            color = (255, 0, 10)

        center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)
        left = getMostLeftPoint(x, y, w, h)
        right = getMostRightPoint(x, y, w, h)

        if detection.category.id != idSpec_epto:
            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                #round 1 decimal
                length = round(length, 1)
                measuredPitch.append(length)
            prev_center = center

        if detection.category.id == idSpec_epto:
            epto_width = calclength(left, right)*pixelMultiplier
            #round 1 decimal
            epto_width = round(epto_width, 1)
            if partSide == "LH":
                epto_edge_point = right
                #Draw small circle on the left edge of the epto tape
            if partSide == "RH":
                epto_edge_point = left

    print("Detected IDs:", detectedid)
    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        leftmostWidth = detectedWidth[0]
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        rightmostWidth = detectedWidth[-1]
      
        # Positive Yoffsetval means going down, negative means going up
        left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = -10, Xoffsetval = 0)
        right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = -10, Xoffsetval = 0)

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

    # print("Measured Pitches:", measuredPitch)

    #add total length
    #round the value to 1 decimal
    # totalLength = sum(measuredPitch)
    # measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    #check if epto is in detectedid
    if idSpec_epto not in detectedid:
        measuredPitch = [0] * (len(pitchSpec))
        resultPitch = [0] * (len(pitchSpec))
        resultid = [0] * (len(idSpec))
        status = "NG"
        ngreason = "EPTO TAPE NOT FOUND"
        print_status = "EPTOテープが見つかりません"
        image = draw_status_text_PIL(image, status, print_status, size="small")
        return image, measuredPitch, resultPitch, resultid, status, ngreason

    if detectedid != idSpec_with_epto:
        measuredPitch = [0] * (len(pitchSpec))
        resultPitch = [0] * (len(pitchSpec))
        resultid = [0] * (len(idSpec))
        status = "NG"
        ngreason = "CLIP MISMATCH"
        print_status = "クリップ不良"
        image = draw_status_text_PIL(image, status, print_status, size="small")
        return image, measuredPitch, resultPitch, resultid, status, ngreason
    
    if detectedid == idSpec_with_epto:
        if partSide == "LH":
            epto_left = abs(detectedposX[3] - epto_edge_point[0])
            epto_left = round(epto_left * pixelMultiplier_eptoLH, 1)
            measuredPitch.append(epto_left)
            measuredPitch.append(epto_width)

        if partSide == "RH":
            epto_right = abs(epto_edge_point[0] - detectedposX[5])
            epto_right = round(epto_right * pixelMultiplier_eptoRH, 1)
            measuredPitch.append(epto_right)
            measuredPitch.append(epto_width)

        print ("Measured Pitches with EPTO:", measuredPitch)

        if measuredPitch[-1] < epto_width - epto_width_tolerance or measuredPitch[-1] > epto_width + epto_width_tolerance:
            measuredPitch = [0] * (len(pitchSpec))
            resultPitch = [0] * (len(pitchSpec))
            resultid = [0] * (len(idSpec))
            status = "NG"
            ngreason = "EPTO TAPE WIDTH NG"
            print_status = "EPTOテープ幅不良"
            image = draw_status_text_PIL(image, status, print_status, size="small")
            return image, measuredPitch, resultPitch, resultid, status, ngreason
        
        if measuredPitch[-2] < epto_clip_pitch - epto_clip_pitch_tolerance or measuredPitch[-2] > epto_clip_pitch + epto_clip_pitch_tolerance:
            measuredPitch = [0] * (len(pitchSpec))
            resultPitch = [0] * (len(pitchSpec))
            resultid = [0] * (len(idSpec))
            status = "NG"
            ngreason = "EPTO SET POSITION NG"
            print_status = "EPTOセット位置不良"
            image = draw_status_text_PIL(image, status, print_status, size="small")
            return image, measuredPitch, resultPitch, resultid, status, ngreason
        
        if len(measuredPitch) == len(pitchSpec):
            resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)

            if any(result != 1 for result in resultPitch):
                # measuredPitch = [0] * (len(pitchSpec))
                # resultPitch = [0] * (len(pitchSpec))
                # resultid = [0] * (len(idSpec))
                status = "NG"
                ngreason = "CLIP PITCH NG"
                print_status = "クリップピッチ不良"
                # image = draw_status_text_PIL(image, status, print_status, size="small")
                # return image, measuredPitch, resultPitch, resultid, status, ngreason



    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=2)

    image = draw_status_text_PIL(image, status, print_status, size="small")
    
    return image, measuredPitch, resultPitch, resultid, status, ngreason
