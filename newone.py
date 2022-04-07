import cv2

# face detection while using the front camera on
# load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_cap = cv2.VideoCapture(0)

while True:
    #read the real face input from the web cam
    _, frame = video_cap.read()

    # convert into grayscale
   # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect files
    detect_faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1,  minNeighbors=4)

    # draw rectangle around the faces
    for(a, b, w, h) in detect_faces:
        cv2.rectangle(frame, (a, b), (a+w, b+h), (179, 245, 66), 2)

    # display the output
    cv2.imshow('detect face',frame)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

video_cap.release()

cv2.destroyAllWindows()



