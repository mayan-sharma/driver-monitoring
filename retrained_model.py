import multiprocessing
import dlib
import cv2
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import time

model = './models/eye_eyebrows_predictor.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

lStart, lEnd = 10, 16
rStart, rEnd = 16, 22

# state
EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 3
DROWSY_CONSEC_FRAMES = 20

HEAD_TILT_CONSEC_FRAMES = 20

FRAME_COUNTER = 0
BLINK_FLAG_COUNTER = 0
DROWSY_FLAG_COUNTER = 0

NO_BLINKS = 0
LAST_BLINK = None
BLINK_RATE_INTERVAL = 5

# utility functions

# calcultes ear
def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    C = dist.euclidean(eye[0], eye[3])
    
    # eye aspect ratio
    ear = (A + B) / (2 * C)
    
    return ear
    
# pre-process image (grayscale + histogram equalization)
def pre_process(frame):
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # histogram equilization
    image = cv2.equalizeHist(gray)
    
    return image

video = cv2.VideoCapture(0)

while True:
    
    FRAME_COUNTER += 1
    # fetch check and frame
    check, frame = video.read()
    
    # pre-process image (grayscale + histogram equalization)
    image = pre_process(frame)
    
    # get faces from detector
    faces = detector(image)
    
    # iterate over each face
    for face in faces:

        # returns all landmarks feature in face
        landmarks = predictor(image, face)
        
        # convert landmarks object to numpy array
        landmarks = face_utils.shape_to_np(landmarks)

        # gets eyes from landmarks array (landmarks contains 68 points)
        leftEye = landmarks[lStart: lEnd]
        rightEye = landmarks[rStart: rEnd]
        
        x1, y1 = leftEye[3][0], leftEye[3][1]
        x2, y2 = rightEye[0][0], rightEye[0][1]
        slope = (y2-y1)/(x2-x1)
        
        # calculates ear for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # calculate average ear
        ear = (leftEAR + rightEAR) / 2
        
        # gets convexHull for both eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
        
        # draw contours around the eye
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # 
        if (LAST_BLINK is not None and ((time.time() - LAST_BLINK) > BLINK_RATE_INTERVAL)):
            cv2.putText(frame, "Gaze Detected!", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        # if head tilt
        if slope < -0.4 or slope > 0.4:
            cv2.putText(frame, "Keep your head straight!", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # if eye is closed
        if ear < EYE_AR_THRESHOLD:
            BLINK_FLAG_COUNTER += 1
            DROWSY_FLAG_COUNTER += 1
            
            if (DROWSY_FLAG_COUNTER > DROWSY_CONSEC_FRAMES):
                LAST_BLINK = None
                cv2.putText(frame, "Drowsiness Detected!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        else:
            # if eye was closed for more than EYE_AR_CONSEC_FRAMES value
            if BLINK_FLAG_COUNTER > EYE_AR_CONSEC_FRAMES:
                NO_BLINKS += 1
                LAST_BLINK = time.time()
                
            # reset FLAG_COUNTER
            BLINK_FLAG_COUNTER = 0
            DROWSY_FLAG_COUNTER = 0
            
        # prints blinks
        cv2.putText(frame, "Blinks: {}".format(NO_BLINKS), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    
    fps = video.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, "Fps: {}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # display frame
    cv2.imshow('Frame', frame)
    
    # gets key pressed
    key = cv2.waitKey(1)
    
    # breaks if key is q
    if key == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()
