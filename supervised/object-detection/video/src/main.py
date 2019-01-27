import cv2
import tensorflow as tf

cap = cv2.VideoCapture(0)

def capture_image():
    ret, image_np = cap.read()
    return image_np

def display_image(image_np):
    cv2.imshow('object detection', image_np)

def render_loop():
    while True:
        display_image(capture_image())
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

render_loop()
