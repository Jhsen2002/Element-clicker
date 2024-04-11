import mss
import numpy as np
import cv2
import pyautogui
import pydirectinput
import keyboard
import torch
import time

# exp 26: humanbenchmark
# exp 34: kovaaks
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp34/weights/best.pt')

# with mss.mss() as sct:
#     monitor = {'top': 40, 'left': 320, 'width': 640 , 'height': 640}
    
#     while not keyboard.is_pressed('q'):
#         img = np.array(sct.grab(monitor))
#         results = model(img)
#         rl = results.xyxy[0].tolist()


#         # if any results do the following
#         if rl and rl[0][4] > .4:
      
#             # get xmax, ymax coordinates
#             x = int(rl[0][2])
#             y = int(rl[0][3])
#             # width = xmax - xmin
#             w = int(rl[0][2] - rl[0][0])

#             # height = ymax - ymin
#             h = int(rl[0][3] - rl[0][1])

#             # center = 1/2 the width
#             center = 1/2 * w
#             xpos = (int(((x - (w/2)) )- pyautogui.position()[0]))
#             ypos = 4 * (int(((y - (h / 2)) )- pyautogui.position()[1]))
#             pydirectinput.moveRel( 0,ypos, relative= True)
#             break

#         cv2.imshow('s', np.squeeze(results.render()))
#         cv2.waitKey(1)


# cv2.destroyAllWindows()

# Define center box dimensions
screen_width, screen_height = 1280,720
center_box_width = 5
center_box_height = 5

# Calculate the top-left and bottom-right coordinates of the center box
box_left = int((screen_width - center_box_width) / 2)
box_top = int((screen_height - center_box_height) / 2)
box_right = box_left + center_box_width
box_bottom = box_top + center_box_height

# Monitor region
with mss.mss() as sct:
    monitor = {'top': 80, 'left': 300, 'width': 640, 'height': 640}
    
    while not keyboard.is_pressed('q'):
        img = np.array(sct.grab(monitor))
        results = model(img)
        detections = results.xyxy[0].tolist()

        # Process detections
        for detection in detections:
            if detection[4] > 0.1:
                x_center = (detection[0] + detection[2]) / 2
                y_center = (detection[1] + detection[3]) / 2

                # Check if target is within the center box
                if not (screen_width / 2 - center_box_width / 2 < x_center < screen_width / 2 + center_box_width / 2 and
                        screen_height / 2 - center_box_height / 2 < y_center < screen_height / 2 + center_box_height / 2):
                    # Calculate movement required
                    x_move = int((x_center - screen_width / 2))+320
                    y_move = int((y_center - screen_height / 2))+40
                    pydirectinput.moveRel(int(x_move), int(y_move), relative=True)

        cv2.rectangle(img, (box_left, box_top), (box_right, box_bottom), color=(0, 255, 0), thickness=2)

        cv2.imshow('s', img)

        cv2.imshow('s', np.squeeze(results.render()))

        cv2.waitKey(1)

cv2.destroyAllWindows()


