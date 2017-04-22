import numpy as np
import cv2
from sklearn.externals import joblib
from picamera.array import PiRGBArray
from picamera import PiCamera
from skimage.feature import hog

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Defining the camera parameters: frame rate and resolution
cap = PiCamera()
cap.resolution = (640, 480)
cap.framerate = 32
im = PiRGBArray(cap, size=(640, 480))

for frame in cap.capture_continuous(im, format="bgr", use_video_port=True):

    image = frame.array
  
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    
    
    # Threshold the image
    # ret, im_th = cv2.threshold(im_gray.copy(), 120, 255, cv2.THRESH_BINARY_INV)
    
    im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    im_th = cv2.medianBlur(im_th, 7)
    
    # Find contours in the binary image 'im_th'

    im_th = cv2.dilate(im_th, np.ones((3,3), np.uint8))
    im_th = cv2.erode(im_th, np.ones((3,3), np.uint8))

    im_th = cv2.morphologyEx(im_th, cv2.MORPH_ELLIPSE, (7,7))
    im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
   

    
    im_th, contours0, hierarchy  = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    area = [cv2.contourArea(cnt) for cnt in contours0]

    # hull = [cv2.convexHull(cnt) for cnt in contours0]

    # Draw contours in the original image 'im' with contours0 as input

    # cv2.drawContours(frame, contours0, -1, (0,0,255), 2, cv2.LINE_AA, hierarchy, abs(-1))
    

    # Rectangular bounding box around each number/contour
    rects = [cv2.boundingRect(cnt) for cnt in contours0]
    
    
    # Draw the bounding box around the numbers(Making it visual)
    for rect, i in zip(rects, range(0, len(rects))):
        
            
     # Make the rectangular region around the digit
     leng = int(rect[3] * 1.6)
     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
     roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        

     moments = [cv2.moments(cnt) for cnt in contours0]
     
     #Check if any regions were found
     if roi.any() and rect[3] < 200 and rect[3] > 30 and rect[2] < 250 and rect[2] > 5:
        # Draw rectangles
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Resize the image
        roi = cv2.resize(roi, (28, 28), im_th, interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = cv2.erode(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(image, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)
        
        
   

    # Display the resulting frame
    # cv2.imshow('frame', image)
    # cv2.imshow('Threshold', im_th)
    # Save image to folder that connected to the OPSORO website
    cv2.imwrite("../OPSORO/OS/src/opsoro/apps/testapp/static/images/example.JPEG", image)

    im.truncate(0)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    
