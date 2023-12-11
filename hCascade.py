#conda environment: arch
import cv2 as cv
#check opencv#
#img = cv.imread("test.png")
#cv.imshow("Display window", img)
#k = cv.waitKey(0)
###

#video = cv.VideoCapture('../datasets/PedestrianDetection/PedestriansCompilation.mp4') #path of video
#video = cv.VideoCapture('../datasets/PedestrianDetection/Pedestrians.mp4') #path of video
video = cv.VideoCapture('../datasets/PedestrianDetection/PedestriansDetection.mp4') #path of video
pedestrianFile = 'haarcascade_fullbody.xml'
pedestrianFile = 'haarcascade_fullbody.xml'
pedestrianTracker = cv.CascadeClassifier(pedestrianFile)


while video.isOpened():
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)        
    else:
        print("no se pudo leer el frame")
        break

    pedestrians = pedestrianTracker.detectMultiScale(gray)

    for(x, y, w, h) in pedestrians:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5) #BGR

    cv.imshow('myVideo', frame)
    if cv.waitKey(25) == ord('q'):
        break


video.release()
cv.destroyAllWindows()








