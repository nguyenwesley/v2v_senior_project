######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
heightm = input_details[0]['shape'][1]
widthm = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def perspective(img):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    dst_size = (1280, 720)
    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, matrix, dst_size)
    return result

def inv_perspective(img):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    dst_size = (1280, 720)
    dst = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    src = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    matrix = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, matrix, dst_size)
    return result

def rgbr(image):
    red = image[:, :, 2]
    return red

def labb(image):
    LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    brightness = LAB[:, :, 2]
    return brightness

def luvl(image):
    LUV = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    lightness = LUV[:, :, 1]
    return lightness

def hlss(image):
    HLS =cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    saturation = HLS[:, :, 2]
    return saturation

def finalbinary(image, threshbright = (175,255), threshsat = (220,250), threshlight = (215,255), threshr = (230,255)):
    saturation = hlss(image)
    lightness = luvl(image)
    brightness = labb(image)
    red = rgbr(image)
    finalsobel = sobel(image)
    binarylit = binarycolor(lightness, threshlight)
    binarysat = binarycolor(saturation, threshsat)
    binarybright = binarycolor(brightness, threshbright)
    binaryred = binarycolor(red, threshr)
    binary = combined(finalsobel, binaryred, binarysat, binarybright, binarylit)
    return binary

def binarycolor(color, thresh):
    binary = np.zeros_like(color)
    binary[(color >= thresh[0]) & (color <= thresh[1])] = 1
    return binary

def sobel(image, thresh=(30,70)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    xsobel = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    ysobel = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    gradmag = np.sqrt(xsobel ** 2 + ysobel ** 2)
    scaledfactor = np.max(gradmag)/255
    gradmag = (gradmag/scaledfactor).astype(np.uint8)
    finalsobel = np.zeros_like(gradmag)
    finalsobel[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return finalsobel

def combined(sobel, r, s, l, b):
    combined_binary = np.zeros_like(sobel)
    combined_binary[(sobel == 1) | (r == 1) | (s == 1) | (l == 1) | (b == 1)] = 255
    return combined_binary

def histogram(binary):
    hist = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    return hist

def windows(image):
    windowN = 10
    margin = 100
    windowH = np.int(image.shape[0]/windowN)
    minP = 50
    hist = histogram(image)
    midpoint = np.int(hist.shape[0] / 2)
    leftlane = np.argmax(hist[:midpoint])
    rightlane = np.argmax(hist[midpoint:]) + midpoint
    points = image.nonzero()
    ycoord = np.array(points[0])
    xcoord = np.array(points[1])
    currentleft = leftlane
    currentright = rightlane
    leftlist = []
    rightlist = []
    for window in range(windowN):
        windowbottom = image.shape[0]-((1+window)*windowH)
        windowtop = image.shape[0]-(window*windowH)
        LLwindow = currentleft - margin
        RLwindow = currentleft + margin
        LRwindow = currentright - margin
        RRwindow = currentright + margin
        cv2.rectangle(image, (LLwindow, windowbottom), (RLwindow, windowtop), [0, 0, 255], 3)
        cv2.rectangle(image, (LRwindow, windowbottom), (RRwindow, windowtop), [0, 0, 255], 3)
        Lwindowpnts = ((xcoord >= LLwindow) & (xcoord < RLwindow) & (ycoord >= windowbottom) & (windowtop > ycoord)).nonzero()[0]
        Rwindowpnts = ((xcoord >= LRwindow) & (xcoord < RRwindow) & (ycoord >= windowbottom) & (windowtop > ycoord)).nonzero()[0]
        leftlist.append(Lwindowpnts)
        rightlist.append(Rwindowpnts)
        if len(Lwindowpnts) > minP:
            currentleft = np.int(np.mean(xcoord[Lwindowpnts]))
        if len(Rwindowpnts) > minP:
            currentright = np.int(np.mean(xcoord[Rwindowpnts]))
    leftlist = np.concatenate(leftlist)
    rightlist = np.concatenate(rightlist)
    leftlinex = xcoord[leftlist]
    leftliney = ycoord[leftlist]
    rightlinex = xcoord[rightlist]
    rightliney = ycoord[rightlist]
    finalleftline = np.polyfit(leftliney, leftlinex, 2)
    finalrightline = np.polyfit(rightliney, rightlinex, 2)
    return finalleftline, finalrightline

# def fastercalc(image, left, right):
#     margin = 100
#     points = image.nonzero()
#     ycoord = np.array(points[0])
#     xcoord = np.array(points[1])
#     left_lane_pixels =
#     right_lane_pixels =

#     return

def background(image):
    back = np.zeros_like(image)
    return back

def plotpoints(image, leftline, rightline, ploty):
    left_lane = leftline[0] * ploty ** 2 + leftline[1] * ploty + leftline[2]
    right_lane = rightline[0] * ploty ** 2 + rightline[1] * ploty + rightline[2]
    pts_left = np.array([np.transpose(np.vstack([left_lane, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane, ploty])))])
    points = np.hstack((pts_left, pts_right))
    back = background(image)
    cv2.fillPoly(back, np.int_(points), color=[0, 255, 0])
    cv2.polylines(back, np.int32([pts_left]), isClosed=False, color=(255, 0, 255), thickness=20)
    cv2.polylines(back, np.int32([pts_right]), isClosed=False, color=(0, 255, 255), thickness=20)
    return back

def pipeline(image):
    bird = perspective(image)
    binary = finalbinary(bird)
    leftline, rightline = windows(binary)
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    back = plotpoints(image, leftline, rightline, ploty)
    finalback = inv_perspective(back)
    result = cv2.addWeighted(image, 1, finalback, 0.5, 0)
    return result

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):	

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = pipeline(frame)
    frame_resized = cv2.resize(frame_rgb, (widthm, heightm))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame_rgb, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame_rgb, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame_rgb)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
