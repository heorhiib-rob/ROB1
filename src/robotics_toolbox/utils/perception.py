#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa

REF_COL = (108, 130, 130) #HSV
MAX_V  = 30
MAX_S  = 30
H_MARG = 20

def segmentate(image_bgr, ref_col = REF_COL):
    """
    Segmentate a BGR image by a reference colour
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    H,S,V = map(int, ref_col)

    hue_diff = np.abs(img[:, :, 0].astype(np.int16) - H)
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)
    
    s_diff = np.abs(img[:, :, 1].astype(np.int16) - S)
    v_diff = np.abs(img[:, :, 2].astype(np.int16) - V)

    mask = (
        (hue_diff <= H_MARG) &
        (s_diff <= MAX_S) &
        (v_diff <= MAX_V)
    )

    return mask.astype(np.uint8) * 255
    

def find_hoop(bin_img):
    """
    Returns camera coordinates of the hoop center.
    """
    circ_score = lambda a, c : 4*np.pi*a/(c**2) # a function for evaluating contours similarity to a circle

    best_cnt = None
    best_circ = None
    best_score = 0

    contours, _ = cv2.findContours(
        bin_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        # Area that the contour is enclosing
        hull = cv2.convexHull(cnt)
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(hull)

        # Perimeter of a contour
        perimeter = cv2.arcLength(
            hull,
            True
        )
        if area < 10000 or perimeter == 0:
            continue # noise

        score = circ_score(area, perimeter)
        if score > best_score:
            best_score = score
            best_circ = ((x,y),radius)
            best_cnt = cnt

    return best_cnt, best_circ

def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    image_points = []
    world_points = []

    # DONE HW03: Detect circle in each image
    kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (5, 5)
            )
    for image, pos in zip(images, hoop_positions):
        # 1st segmentate
        bin_im = segmentate(image)

        bin_im = cv2.dilate(
            bin_im,
            kernel,
            iterations=1
        )

        # bin_im = cv2.morphologyEx(
        #             bin_im,
        #             cv2.MORPH_CLOSE,
        #             kernel
        #         )

        # Find hoop center
        cnt, ((cx,cy),r) = find_hoop(bin_im)

        #DEBUG
        # cv2.drawContours(
        #     image,
        #     [cnt],
        #     -1,
        #     (0,0,255),
        #     2
        # )
        # cv2.circle(
        #     image,
        #     (round(cx),round(cy)),
        #     round(r),
        #     (0,255,0),
        #     2
        # )
        # cv2.namedWindow("fig", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("fig", 800, 600)

        # cv2.imshow("fig", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        image_points.append([cx,cy])

        # World coords
        x,y,z = pos["translation_vector"]
        world_points.append([x, y])

    # DONE HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.

    world_points = np.array(world_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    print("World points:", world_points.shape)
    print("Image points:", image_points.shape)

    H, status = cv2.findHomography(
        image_points,
        world_points,
        cv2.RANSAC
    )

    return H
