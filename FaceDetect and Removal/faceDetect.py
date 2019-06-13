import numpy as np
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")

#added
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),-1)
    # cv2.imshow('gray', gray)
    img = cv2.flip(img, +1);
    cv2.imshow('img',img)





    # skinMask = cv2.inRange(gray, lower, upper)
 
    # # apply a series of erosions and dilations to the mask
    # # using an elliptical kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
 
    # # blur the mask to help remove noise, then apply the
    # # mask to the img
    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    # skin = cv2.bitwise_and(img, img, mask = skinMask)
 
    # # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([img, skin]))























    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()