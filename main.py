#### import dependencies ####
import cv2
# mediapipe is our core dependency to be able to do the project
import mediapipe as mp
import numpy as np
# mp_drawing -> give us all of our drwaing utilities so when we actually visualize our poses we will use this drawing utilities
mp_drawing = mp.solutions.drawing_utils
# mp_pose -> import our pose estimation model
mp_pose = mp.solutions.pose

# setting up our video capture device
# we can pass a video path or pass a 0 to open the live camera
cap = cv2.VideoCapture(0)

#### Setup mediapipe instance ####
# mp_pose.Pose -> access our pose estimation model, choose the confidence based how much you want the model to be accurate
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # frame is a feed of webcam
        # ret -> the return variable, frame -> give us the image from our camera
        ret, frame = cap.read()

        # Recolor image to RGB because when we pass the image to mediapipe we want to be in RGB format because the default is bgr
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # apply some performance tuning by setting whether or not it's writable = to false
        # to save a bunch of memory once we pass this to our pose estimation model
        image.flags.writeable = False

        # Make detection by accessing the pose model
        # then stores our results and by processing it we're going to get our detections back and store it in result
        results = pose.process(image)

        # setting our image writable back to true
        image.flags.writeable = True
        # Recolor back to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw our detection to our image
        # draw_landmarks -> using our drawing utilities that we imported up to draw to our particular image
        # results.pose_landmarks -> print the coordinates for each and every landmark within our body it represents each individual point that's represented as part of the pose estimation model
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, # mp_pose.POSE_CONNECTIONS -> which landmarks are connected to which for example the nose will be connected to left&right inner eye so it will draw or build up
                                  # mp_drawing.DrawingSpec -> specifications for our drawing components the first one represent the color we want our different dots
                                  mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                                  # for the connection
                                  mp_drawing.DrawingSpec(color=(200,200,200), thickness=2, circle_radius=2)
                                  )
        cv2.imshow('Action Recognition', image)

        if cv2.waitKey(7) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()