#!/usr/bin/env python

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

DRAW_BOUNDING_BOX = False
DRAW_LANDMARK_INDICES = False

DOWNSCALE_IMAGE = True
FRAME_WIDTH = 400  # width of downscaled frame
FRAME_HEIGHT = 225  # height of downscaled frame

EYE_AR_THRESH = 0.3  # minimum threshold for eye aspect ration to register blink
EYE_AR_CONSEC_FRAMES = 2  # number of consecutive frames the eye must be below the threshold


# function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute and return the eye aspect ratio
    return (a + b) / (2.0 * c)


# function to extract the key landmarks
def key_landmarks(shape):
    image_points = np.array([
        shape[33],  # nose tip
        shape[8],  # chin
        shape[36],  # right eye right corner
        shape[45],  # left eye left corner
        shape[48],  # right mouth corner
        shape[54],  # left mouth corner
    ], dtype='double')
    return image_points


# function returns if landmark on index is a key landmark
def is_key_landmark(index):
    return index == 33 or index == 8 or index == 36 or index == 45 or index == 48 or index == 54


# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# access webcam
print("[INFO] accessing web cam video stream...")
cap = cv2.VideoCapture(0)

# declare 2d image points array
image_points = np.array([(0, 0)] * 6, dtype="double")

# 3d model points
model_points = np.array([
    (0.0, 0.0, 0.0),  # nose tip 34
    (0.0, -330.0, -65.0),  # chin 9
    (-225.0, 170.0, -135.0),  # right eye right corner 37
    (225.0, 170.0, -135.0),  # left eye left corner 46
    (-150.0, -150.0, -125.0),  # right mouth corner 49
    (150.0, -150.0, -125.0)  # left mouth corner 55
])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# get image shape
size = (0, 0)
if DOWNSCALE_IMAGE:
    size = (FRAME_WIDTH, FRAME_HEIGHT)
else:
    test_img = cap.read()
    size = test_img.shape()

# get camera matrix and distance coefficients
camera_matrix = np.array([[size[1], 0, size[1] / 2], [0, size[1], size[0] / 2], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion

# variables
counter = 0  # frame counter
total = 0  # total number of blinks

while True:

    # capture image
    ret, frame = cap.read()

    # break the loop, if no image captured
    if not ret:
        print("Error: Failed to capture image")
        break

    # resize image to new resolution
    if DOWNSCALE_IMAGE:
        frame = imutils.resize(frame, FRAME_WIDTH, FRAME_HEIGHT)

    # convert image to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the greyscale frame
    rects = detector(grey, 0)

    # draw current blink count
    cv2.putText(frame, "Blinks: {}".format(total), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # check to see if a face was detected
    if len(rects) > 0:

        # draw the total number of faces on the screen
        cv2.putText(frame, "Face(s) found: {}".format(len(rects)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # loop over all the detected faces
        for rect in rects:

            # compute the bounding box of the face and draw it on the frame
            if DRAW_BOUNDING_BOX:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(grey, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lEyeStart:lEyeEnd]
            leftEAR = eye_aspect_ratio(leftEye)

            rightEye = shape[rEyeStart:rEyeEnd]
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                counter += 1

            # otherwise, the eye aspect ratio is not below the blink threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if counter >= EYE_AR_CONSEC_FRAMES:
                    total += 1

                # reset the eye frame counter
                counter = 0

            # draw the eye aspect ratio of the current face
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # extract key landmarks
            image_points = key_landmarks(shape)

            # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
            for (i, (x, y)) in enumerate(shape):
                color = (0, 255, 0) if is_key_landmark(i) else (0, 0, 255)
                cv2.circle(frame, (x, y), 1, color, -1)
                if DRAW_LANDMARK_INDICES:
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # project a 3D point (0, 0 , 1000.0) onto the image plane
            # we use this to draw a line sticking out of the nose_end_point2D
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            print("Rotation Vector:\n {0}".format(rotation_vector))
            print("Translation Vector:\n {0}".format(translation_vector))


            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    # show the frame
    cv2.imshow('Output Image', frame)

    # exit on key press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Remove all windows when finished
cv2.destroyAllWindows()
