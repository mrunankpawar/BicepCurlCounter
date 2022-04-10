#!/usr/bin/env python
# coding: utf-8

# # Install and import dependencies 

# In[2]:


get_ipython().system('pip install mediapipe opencv-python')
get_ipython().system('pip install anvil-uplink')


# In[4]:


import cv2 #import openCV in our notebook
import mediapipe as mp
import numpy as np
import anvil.server

anvil.server.connect("DN4IOJDUWV3WG2DHOZW3XDIG-B3J3I5AUCL37F7PT")
mp_drawing = mp.solutions.drawing_utils #helps to visualize the pose
mp_pose = mp.solutions.pose


# In[3]:


# Video Feed
cap = cv2.VideoCapture(0)
while cap.isOpened():
  ret, frame = cap.read()
  cv2.imshow('AI Pose Estimation', frame)

  if cv2.waitKey(10) & 0xFF  == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()


# # Make Detections

# In[4]:


cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                 )
        
        cv2.imshow('AI Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# # Determining Joints

# ![title](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

# In[17]:


cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass    
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                 )
        
        print (results)
        
        cv2.imshow('AI Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[6]:


len(landmarks)


# In[7]:


for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)


# In[8]:


landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]


# In[9]:


landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]


# In[10]:


landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]


# # Calculate angles

# In[11]:


def calculate_angle(a,b,c):
#     a = start; a[0] = x; a[1] = y
#     b = mid; b[0] = x; b[1] = y
#     c = end; c[0] = x; c[1] = y

    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)
    
#     np.arctan2(y, x)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     Converting radians to degree (radian*180/pi)
    angle = np.abs(radians*180.0/np.pi) 
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle


# In[12]:


shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]


# In[13]:


shoulder, elbow, wrist


# In[14]:


calculate_angle(shoulder, elbow, wrist)


# In[15]:


tuple(np.multiply(elbow, [640, 480]).astype(int))


# In[22]:


cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            #Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            #visualize
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [1280, 1000]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
#             print(landmarks)
        except:
            pass    
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                 )
        
        print (results)
        
        cv2.imshow('AI Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# # Curl Counter

# In[26]:


cap = cv2.VideoCapture(0)

# Curl Counter Variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            #Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            #visualize
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [1280, 1000]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            #Curl Counter Logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter+=1
                print(counter)
                
        except:
            pass    
        
        #Render Curl Counter
        #setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                 )
        
        cv2.imshow('AI Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


@anvil.server.callable
def 

