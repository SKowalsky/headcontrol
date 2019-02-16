#!/usr/bin/env python

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

SKIP_FRAMES = 3
FRAME_WIDTH = 400
FRAME_HEIGHT = 225

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

# print out the default resolution of the camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))
print("input frame width {}".format(frame_width))
print("input frame height {}".format(frame_height))

# now use lower resolution
print("output frame width {}".format(FRAME_WIDTH))
print("output frame height {}".format(FRAME_HEIGHT))

# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (355, 391),     # Nose tip 34
    (389, 541),     # Chin 9
    (327, 227),     # Left eye left corner 37
    (533, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (455, 415)      # Right mouth corner 55
], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip 34
    (0.0, -330.0, -65.0),        # Chin 9
    (-225.0, 170.0, -135.0),     # Left eye left corner 37
    (225.0, 170.0, -135.0),      # Right eye right corne 46
    (-150.0, -150.0, -125.0),    # Left Mouth corner 49
    (150.0, -150.0, -125.0)      # Right mouth corner 55

])

# variables
skipf = 0

while True:

    # skip every Nth frame
    skipf+=1
    if skipf < SKIP_FRAMES:
        continue
    skipf = 0

    # capture image
    ret, image = cap.read()

    # break the loop, if no image captured
    if ret == False:
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
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # loop over all the detected faces
        for rect in rects:

            # compute the bounding box of the face and draw it on the frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks and draw each of them
            for (i_counter, (x, y)) in enumerate(shape):

                if i_counter == 33:
                    # save key landmark to key point list
                    image_points[0] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                elif i_counter == 8:
                    # save key landmark to key point list
                    image_points[1] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                elif i_counter == 36:
                    # save key landmark to key point list
                    image_points[2] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                elif i_counter == 45:
                    # save key landmark to key point list
                    image_points[3] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                elif i_counter == 48:
                    # save key landmark to key point list
                    image_points[4] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                elif i_counter == 54:
                    # save key landmark to key point list
                    image_points[5] = np.array([x, y], dtype='double')

                    # write on frame in green
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                else:
                    # write all other landmarks on frame in red
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i_counter + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


            # add pose estimation
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

            dist_coeffs = np.zeros((4,1)) # assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # print("Camera Matrix :\n {0}".format(camera_matrix))
            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # project a 3D point (0, 0 , 1000.0) onto the image plane
            # we use this to draw a line sticking out of the nose_end_point2D
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(frame, p1, p2, (255,0,0), 2)


    # show the frame
    cv2.imshow('Output Image', frame)

    # exit on key press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Remove all windows when finished
cv2.destroyAllWindows()