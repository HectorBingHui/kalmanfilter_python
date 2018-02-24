
"""
Using harcasing to detect faces and then initialize kalman filter for tracking 
"""
import cv2
from smartvision.util.image import draw
import numpy as np
from time import sleep
from math import sqrt
from time import sleep


def main():

    cap = cv2.VideoCapture(0)
    stateSize = 4 
    measSize = 4
    contrSize = 0 

    kf = cv2.KalmanFilter(stateSize,measSize,contrSize)

    state = np.ones(4,np.float32)

    means = np.ones(4,np.float32)


    dT = 0 # velocity

    # to initialize noises and matrixes.

    kf.transitionMatrix = np.array(
        [[1, 0, dT, 0], [0, 1, 0, dT], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kf.processNoiseCov = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kf.measurementNoiseCov = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kf.errorCovPost = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    
    ticks  = 0 
    found = False

    while (cap.isOpened()):
        ret , frame = cap.read()
        frame = cv2.resize(frame,(600,480))

        #to get event count time
        preticks = ticks
        ticks = cv2.getTickCount()
        dT = (ticks - preticks ) / cv2.getTickFrequency()

        #=== to run detection===
        prefix = 'pathto/cv2/data/'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(prefix + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.2,2,0,(50,50))
        
        # in facee is found start kf preidction, and upadate meas for correction 
        # as for now only detecting one face 
        for x,y,w,h in faces:
            kf.statePost = np.array([x,y,w,h],np.float32)
            cv2.rectangle(frame,(x,y),(x+w-1,y+h-1),(255,0,200),1)
           
            means = np.array([x,y,w,h],np.float32)
            print('Detection:', means)
            
  
        if (faces != []):
            found = True
        
        else:
            found = False
        
        #=======kalmanfilter==========
        if (found):
            
            #satart prediction 
            prediction = kf.predict()
            prediction= [prediction]
            print('Prediction:' , np.squeeze(prediction))
            
            # draw on frame based on prediction
            frame = draw(frame,prediction)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,'Prediction',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

            #correcton based on measurement update added some noise 
            measure_noice = sqrt(kf.measurementNoiseCov[0,0]) * np.random.randn(1, 2)
            measure_noice = np.append(measure_noice,1)
            measure_noice = np.append(measure_noice,1)
            means = np.dot(kf.measurementMatrix, means)  # supposed to add noise as well, bugss
            means = np.absolute(means)
            means = np.around(means,decimals=2)
            print('Update', means)
        
            kf.correct(means)

            # adding randome noise to predict next state 
            
            process_noise = sqrt(kf.processNoiseCov[0, 0]) * np.random.randn(1, 2)
            process_noise = np.append(process_noise,1)
            process_noise = np.append(process_noise,1)
            state = np.dot(kf.transitionMatrix, prediction) + process_noise
            stateNext= np.squeeze(np.around(state[2,:],decimals=2))
            kf.statePost = np.array(stateNext,np.float32)
        
        

        cv2.imshow('KalmanFilter',frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()