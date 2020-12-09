# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
import datetime
import importlib.util

#MQTT Stuff
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
#End of MQTT Stuff

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

#MQTT Stuff

#IP of the Raspberry Pi this code is running on.
#Set static IP for Pi or manually change IP everytime.

#Wesley Pi Static IP
#hostID = '192.168.0.200'

#Adam Pi Static IP
hostID = '10.0.0.125'

moveTopic = 'vehicle/Move'
#Spare Topics when needed, remove if not used.
loremTopic = ''
ipsumTopic = ''

#End of MQTT stuff


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

if labels[0] == '???':
    del(labels[0])

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

def perspective(img, ypnts):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    dst_size = (1280, 720)
    src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    matrix = cv2.getPerspectiveTransform(src, dst)
    i = 0
    global carypix
    for i in range(len(ypnts)):
        p = (img.shape[1]/2,ypnts[i])
        if ypnts[i] > 480:
            z = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2])/((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
            z = int(z)
            carypix.append(z)  
        else:
            z = 0
            carypix.append(z)
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

def finalbinary(image, threshbright = (210,255), threshsat = (180,255), threshlight = (220,255), threshr = (210,255)):
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

def midpoint(image, leftlane, rightlane):
    y = image.shape[0]
    x = image.shape[1]
    xL = leftlane[0]*y**2 + leftlane[1]*y + leftlane[2]
    xR = rightlane[0]*y**2 + rightlane[1]*y + rightlane[2]
    mid = ((x/2)-xL)/(xR-xL)*100
    mid = int(mid)
    return mid
        
def conversion(theta):
    for n in range(len(carypix)):
        distancepx = 720-carypix[n]
        distanceyft = (distancepx/240)*55
        z = distanceyft
        #z = distanceyft*(1/np.cos(theta))
        global distanceft
        distanceft.append(int(z))
    return 
    
def pipeline(image,ypnts,theta):
    bird = perspective(image,ypnts)
    conversion(theta)
    binary = finalbinary(bird)
    leftline, rightline = windows(binary)
    center = midpoint(image, leftline, rightline)
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    back = plotpoints(image, leftline, rightline, ploty)
    finalback = inv_perspective(back)
    result = cv2.addWeighted(image, 1, finalback, 0.5, 0)
    
    #Package movement payload to send to moveTopic.
    payload = constructMovePay(center)
    mqttSend(payload, moveTopic)

    result = cv2.putText(result, str(center), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    result = cv2.putText(result, 'Midpoint', (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    result = cv2.putText(result, 'Distance(ft)', (50, (80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    global distanceft
    global carypix
    for n in range(len(distanceft)):
        if distanceft[n] == 165:
            cv2.putText(result, 'Vehicle To Distant', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        else:
            result = cv2.putText(result, str(distanceft[n]), (50, (140+n*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    
    del carypix[:]
    del distanceft[:]
    del ypnts[:]
    return result

#Assemble the movement payload into specific format.
#payloadSize = 8
#payload[0-2] = midpoint value
#payload[3-5] = distance from another vehicle value
#payload[6-7] = decides if vehicle is moving forward or backwards
def constructMovePay(center):
    payload = ""
    payload += str(center).zfill(3)
    payload += str(min(distanceft)).zfill(3)
    #Setting default movement as Forward
    payload += "10"

#MQTT Function that publishes current center of the vehicle.
#mqttSend is modular and doesn't require a function to construct its payload.
def mqttSend(payload, topic):
    publish.single(topic, str(payload), hostname=hostID)
    print("\nTime payload was sent:- ", datetime.datetime.now())
    print("Payload: ", payload)

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
x = 0
carypix = []
distanceft = []
ypnts = []

while(video.isOpened()):		
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    x = x + 1
    if x == 25:
        x = 0
        ymax = 0
        theta = 0
    
        #frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame
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
                theta = np.arctan(ymin/(xmin-(frame_rgb.shape[1]/2)))
                
                if xmin > 650:
                    cv2.rectangle(frame_rgb, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    # Draw label
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame_rgb, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame_rgb, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    ypnts.append(ymax)
                    
		# All the results have been drawn on the frame, so it's time to display it.
        lanes = pipeline(frame_rgb, ypnts, theta)
		
        cv2.imshow('Object detector', lanes)
	
    # Press 'q' to quit
    elif cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
