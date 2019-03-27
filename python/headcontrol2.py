#!/usr/bin/env python

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

face_landmark_path = './shape_predictor_68_face_landmarks.dat'  # landmark predictor file
DOWNSCALE_IMAGE = False     # downscale image
DOWNSCALE_FACTOR = 1    # downscale factor (1 = 100%)
EYE_AR_THRESH = 0.25  # minimum threshold for eye aspect ration to register blink

Y_HIGH = 10
Y_LOW = 10
X_HIGH = 10
X_LOW = 10

# 3d model points (referenced from
# http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp)
object_pts = np.float32([
    [6.825897, 6.760612, 4.402142],     # 33: left brow left corner
    [1.330353, 7.122144, 6.903745],     # 29: left brow right corner
    [-1.330353, 7.122144, 6.903745],    # 34: right brow left corner
    [-6.825897, 6.760612, 4.402142],    # 38: right brow right corner
    [5.311432, 5.485328, 3.987654],     # 13: left eye left corner
    [1.789930, 5.393625, 4.413414],     # 17: left eye right corner
    [-1.789930, 5.393625, 4.413414],    # 25: right eye left corner
    [-5.311432, 5.485328, 3.987654],    # 21: right eye right corner
    [2.005628, 1.409845, 6.165652],     # 55: nose left corner
    [-2.005628, 1.409845, 6.165652],    # 49: nose right corner
    [2.774015, -2.080775, 5.048531],    # 43: mouth left corner
    [-2.774015, -2.080775, 5.048531],   # 39: mouth right corner
    [0.000000, -3.116408, 6.097667],    # 45: mouth central bottom corner
    [0.000000, -7.415691, 4.070434]     # 06: chin corner
])

# reproject 3D points world coordinate axis to verify result pose
reprojectsrc = np.float32([
    [10.0, 10.0, 10.0],
    [10.0, 10.0, -10.0],
    [10.0, -10.0, -10.0],
    [10.0, -10.0, 10.0],
    [-10.0, 10.0, 10.0],
    [-10.0, 10.0, -10.0],
    [-10.0, -10.0, -10.0],
    [-10.0, -10.0, 10.0]
])

# line pairs for bounding box
line_pairs = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lEyeStart, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rEyeStart, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
# END DEF

def eye_closed(shape):
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lEyeStart:lEyeEnd]
    leftEAR = eye_aspect_ratio(leftEye)

    rightEye = shape[rEyeStart:rEyeEnd]
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # check to see if the eye aspect ratio is below the blink threshold
    return ear < EYE_AR_THRESH
# END DEF

# function to calculate the euler angle (head pose)
def get_head_pose(shape, camera_matrix):
    # fill in 2D reference points
    image_pts = np.float32([
        shape[17],  # left brow left corner
        shape[21],  # left brow right corner
        shape[22],  # right brow left corner
        shape[26],  # right brow right corner
        shape[36],  # left eye left corner
        shape[39],  # left eye right corner
        shape[42],  # right eye left corner
        shape[45],  # right eye right corner
        shape[31],  # nose left corner
        shape[35],  # nose right corner
        shape[48],  # mouth left corner
        shape[54],  # mouth right corner
        shape[57],  # mouth central bottom corner
        shape[8]    # chin corner
    ])

    # calculate pose
    (success, rotation_vec, translation_vec) = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs)

    # reproject
    (reprojectdst, jacobian) = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calculate the euler angle
    (rotation_mat, _) = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    (_, _, _, _, _, _, euler_angle) = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
# END DEF

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

# access webcam
print("[INFO] accessing web cam video stream...")
cap = cv2.VideoCapture(0)

# capture test image and get size
ret, test_img = cap.read()
size = test_img.shape

# calculate new frame size
frame_width = int(size[0] * DOWNSCALE_FACTOR)
frame_height = int(size[1] * DOWNSCALE_FACTOR)

# get camera matrix and distance coefficients
camera_matrix = np.array([[size[1], 0, size[1] / 2], [0, size[1], size[0] / 2], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion

X = 200
Y = 200

while cap.isOpened():

    # capture image
    ret, frame = cap.read()

    # break the loop, if no image captured
    if not ret:
        print("Error: Failed to capture image")
        break

    # resize image to new resolution
    if DOWNSCALE_IMAGE:
        frame = imutils.resize(frame, frame_width, frame_height)

    # detect faces in the frame
    rects = detector(frame, 0)

    # loop over all the detected faces
    for rect in rects:

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        #if eye_closed(shape):
        #    cv2.circle(frame, (20, 100), 10, (0, 255, 0), -1)
        #else:
        #    cv2.circle(frame, (20, 100), 10, (0, 255, 0), 1)

        # call head pose function
        reprojectdst, euler_angle = get_head_pose(shape, camera_matrix)

        # draw all landmark points
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # draw bounding box
        for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

        # print out rotation values
        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=1)
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=1)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), thickness=1)

        # draw moving circle
        if(euler_angle[0, 0] > Y_HIGH):
            Y += 5
        elif(euler_angle[0, 0] < Y_LOW):
            Y -=5

        if(euler_angle[1, 0] > X_HIGH):
            X -=5
        elif(euler_angle[1, 0] < X_LOW):
            X +=5

        cv2.circle(frame, (X, Y), 20, (255, 0, 0), -1)

    cv2.imshow("Head Control", frame)

    # quit program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Remove all windows when finished
cv2.destroyAllWindows()
