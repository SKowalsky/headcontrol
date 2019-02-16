#!/usr/bin/env python

from imutils import face_utils
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

SKIP_FRAMES = 3  # only process every nth frame
FRAME_WIDTH = 400  # width of downscaled frame
FRAME_HEIGHT = 225  # height of downscaled frame
EYE_AR_THRESH = 0.2  # minimum threshold for eye aspect ration to register blink
EYE_AR_CONSEC_FRAMES = 1  # number of consecutive frames the eye must be below the threshold


# function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# access webcam
print("[INFO] accessing web cam video stream...")
cap = cv2.VideoCapture(0)

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (355, 391),  # Nose tip 34
    (389, 541),  # Chin 9
    (327, 227),  # Left eye left corner 37
    (533, 301),  # Right eye right corne 46
    (345, 465),  # Left Mouth corner 49
    (455, 415)  # Right mouth corner 55
], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip 34
    (0.0, -330.0, -65.0),  # Chin 9
    (-225.0, 170.0, -135.0),  # Left eye left corner 37
    (225.0, 170.0, -135.0),  # Right eye right corne 46
    (-150.0, -150.0, -125.0),  # Left Mouth corner 49
    (150.0, -150.0, -125.0)  # Right mouth corner 55

])

# variables
skipf = 0  # frames skipped
counter = 0  # frame counter
total = 0  # total number of blinks

while True:

    # skip every Nth frame
    skipf += 1
    if skipf < SKIP_FRAMES:
        continue
    skipf = 0

    # capture image
    ret, image = cap.read()

    # break the loop, if no image captured
    if not ret:
        break

    # resize image to new resolution
    frame = imutils.resize(image, FRAME_WIDTH, FRAME_HEIGHT)

    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get image shape
    size = gray.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:

        # loop over all the detected faces
        for rect in rects:

            # compute the bounding box of the face and draw it on the frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            # (could also handle the EAR of both eyes separately)
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

            # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
            for (i_counter, (x, y)) in enumerate(shape):

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Face(s) found: {}".format(len(rects)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Blinks: {}".format(total), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if i_counter == 33:
                    # save key landmark to key point list
                    image_points[0] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                elif i_counter == 8:
                    # save key landmark to key point list
                    image_points[1] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                elif i_counter == 36:
                    # save key landmark to key point list
                    image_points[2] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                elif i_counter == 45:
                    # save key landmark to key point list
                    image_points[3] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                elif i_counter == 48:
                    # save key landmark to key point list
                    image_points[4] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                elif i_counter == 54:
                    # save key landmark to key point list
                    image_points[5] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 255, 0), 1)

                else:
                    # write all other landmarks on frame in red
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 0, 255), 1)

            # add pose estimation
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")

            dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # print("Camera Matrix :\n {0}".format(camera_matrix))
            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # project a 3D point (0, 0 , 1000.0) onto the image plane
            # we use this to draw a line sticking out of the nose_end_point2D
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

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