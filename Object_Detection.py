from PIL import Image
import cv2
import numpy as np
import requests

# Load the image from the URL and resize it
image = Image.open(requests.get('https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg', stream=True).raw)
image = image.resize((450,250))

# Convert the image to a NumPy array
image_arr = np.array(image)

# Convert the image to grayscale
grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)

dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)

car_cascade_src = 'C:/Users/Monster/Desktop/cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)

cnt = 0
for (x,y,w,h) in cars:
  cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
  cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)

cascade_src = 'C:/Users/Monster/Desktop/cars.xml'
video_src = 'C:/Users/Monster/Desktop/car.mp4'

cap = cv2.VideoCapture(0)
car_cascade = cv2.CascadeClassifier(cascade_src)
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Detect cars in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    
    # Draw bounding boxes around the detected cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Display the original image and the grayscale image side by side
cv2.imshow('Original Image', image_arr)
cv2.imshow('Grayscale Image', blur)
cv2.imshow('Grayscale Image', dilated)
cv2.imshow('Grayscale Image', closing)



# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
