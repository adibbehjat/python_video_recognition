# Part of source codes, tutorials, contributions:
# https://realpython.com/blog/python/face-detection-in-python-using-a-webcam/
# http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html
# http://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
# http://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html#gsc.tab=0
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
# http://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image

# Import the libraries necessary for the system
# External libraries: Numpy and PIL (Python Image Library)
import cv2, sys, os
import numpy as np
from PIL import Image

# Provide the HaarCascades, or face template. Usually found in the OpenCV directory
# The frontal face HaarCascades template captures the face 'skin' region. 
cascPath = "opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml"

# Hair and other elements are excluded. However, you're welcome to add more templates, 
# such as for detecting eyes, ears, nose, objects, etc. More detection templated 
# can be found in the Haarcascades directory

""" Request from the user two inputs """
# Tolerance is the padding for the box around the detected face. Otherwise, if padding is not provided,
# the box will 'stick' to close to the skin of the face.
tolerance = int(raw_input("Please specify the padding, ideally between 20 - 50 (int): "))

# Confidence level outputted by OpenCV is a value between 100 - 0. 100 means that there is absolutely
# no similarity/equality between the detected face and recognized face, 0 means that there is an 
# exact match between the recognized face and detected face.
confidence_level = int(raw_input("Please specify the required confidence, ideally <40 (int): "))

# Facial Mapper
faceCascade = cv2.CascadeClassifier(cascPath)

# The training function aggregator. Takes in one argument which is the location of your training set.
# The training set contains images of the face that needs to be recognized. Refer to the following
# tutorial for recommendations on naming:
# http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
def get_images_and_labels(path = "./faces/"):
	
	# Declare variables
	images = []
	labels = []

	# Append all the absolute image paths in a list image_paths
	image_paths = [os.path.join(path, f) for f in os.listdir(path) if  f.endswith('.jpg') and f.startswith('1_')]

	# Loop through the training images
	for image_path in image_paths:
		
		# Read the image and convert to grayscale. All, if not most, algorithms REQUIRE the images
		# to be grayscaled in order to capture the contour of the face more easily
		image_pil = Image.open(image_path).convert('L')

		# Convert the image format into numpy array - this is how OpenCV handles facial structure
		image = np.array(image_pil,'uint8')

		# Set the label of the image (must be an int)
		name = 1
		# If you have multiple entities in your images that you'd like to have recognized, then you can 
		# specify unique label for each persona. For example, label number 1 for yourself, 
		# label number 2 for someone else, etc.

		# Detect the face in the image based on the HaarCascade object. For more configuration, you can pass
		# addition variables into the request
		faces = faceCascade.detectMultiScale(image)

		"""detectMultiScale Expanded"""
		# faces = faceCascade.detectMultiScale(
		#	 image, # image source
		#	 scaleFactor=1.1, # magnifies the detected face. Must be more than 1.0
		#	 minNeighbors=5, # neighbors define the elements on the face. Ideal range is between 3 - 6
		#	 minSize=(30, 30), # The minimum size of the detected face. Depends on video frame input size
		#	 flags = 0 # Default value for HaarCascade Sizing
		# )


		# If face is detected, append the face to images and
		# to label the labels
		for (x,y,w,h) in faces:

			# This shows the detected face in a new window for the person to review
			cv2.imshow("Recognizing Face",image[y: y + h, x: x + w])

			# Appends the image to the image array
			images.append(image[y:y+h,x:x+w])

			# Appends the image to the label array
			labels.append(name)

			# Let us know what's up
			print "Adding faces to training set..."

			# Pause for 1 second/100ms
			cv2.waitKey(100)

	# Return the images found and labels
	return images, labels

# Initiate the training first
recognizer = cv2.face.createLBPHFaceRecognizer()
# Face recognizer using LBPH Algorithm
# Eigenface Recognizer - createEigenFaceRecognizer()
# Fisherface Recognizer  - createFisherFaceRecognizer()
# Local Binary Patterns Histograms Face Recognizer - createLBPHFaceRecognizer()
# An explanation of each algorithm can be found here: http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html

# Call the training set aggregator
images, labels = get_images_and_labels()

# Any open windows from the "imshow" call (line 83)
cv2.destroyAllWindows()

# Conduct the training with the provided images. You can consider this as matrix combination
recognizer.train(images, np.array(labels))

# Open and initiate the default (0) camera. 
# You can access different cameras using different ints
video_capture = cv2.VideoCapture(0)

# Declare variables for box/padding optimization. 
# Those two points are corner points for a square
pntA = [0,0]
pntB = [0,0]

# Loop through the video frames
while True:

	# Capture frame-by-frame. From the documentation:
	# The methods/functions combine VideoCapture::grab and VideoCapture::retrieve in one call.
	# This is the most convenient method for reading video files or capturing data from 
	# decode and return the just grabbed frame
	ret, frame = video_capture.read()

	# Color the frame gray. Since the frame is captured from openCV's Video Capture object,
	# the image will already be in an np.array format (unlike PIL where you have to transform)
	# the object into an np.array 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect the face
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=6,
		minSize=(50, 50),
		flags = 0
	)

	# Draw a rectangle around the faces
	# 'frame' is the video frame

	for (x, y, w, h) in faces:

		# based on the learnings, predict who the individuals are in the video frame
		name_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])

		# This is where the distinction is made between individuals
		# If you planned on recognizing multiple individuals, then labels can be use (like
		# switch command in C)
		if conf < confidence_level:
			color = (0,255,255)
			text = "ADMIN"
			overlayImage = True
		else:
			color = (255,255,255)
			text = "ASSET"
			overlayImage = False


		# Smoothen the detection box
		# Smoothen the way how the vertices are detected and how the face is boxed.
		# 30 seems to be the ideal tolerance point in my experience. Otherwise, if this padding is changed
		# the box will turn out to be very erratic

		# Define point A of the rectangle
		if x > pntA[0] + tolerance or x < pntA[0] - tolerance:
			pntA[0] = x

		if y > pntA[1] + tolerance or y < pntA[1] - tolerance:
			pntA[1] = y



		# Define point B of the rectangle
		if (x+w) > pntB[0] + tolerance or (x+w) < pntB[0] - tolerance:
			pntB[0] = x + w

		if (y+h) > pntB[1] + tolerance or (y+h) < pntB[1] - tolerance:
			pntB[1] = y + h


		# Draw the rectangle or overlay image
		if overlayImage:

			# Overlay the Person of Interest image

			# Specify the coordinates of the Admin
			y_offset = pntA[1] - tolerance
			x_offset = pntA[0] - tolerance

			# Find the overlay image. Note I use svg because it has scalability capabilities
			overlay = cv2.imread("Admin_Square.svg", -1)

			# Define the ideal size of the box and stablize
			overlay_size = pntB[0] - pntA[0] + 2*tolerance

			# Resize the image to fit my face nicely
			overlay = cv2.resize(overlay, (overlay_size, overlay_size))

			# # SVG's transparent region will turn out black because we're trying to transpose an image that has
			# # four channels (rgba) into three channels (rgb), which is the video frame. 

			# # Capture the shape information of the overlay for image blending purposes
			# rows,cols,channels = overlay.shape

			# # Specify the region of interest (ROI) where I would like to place the overlay in the frame
			# roi = frame[y_offset:y_offset+rows, x_offset:x_offset+cols]

			# # Like everything in OpenCV, make sure you color the overlay gray for ease of detection
			# gray_overlay = cv2.cvtColor(overlay,cv2.COLOR_BGR2GRAY)

			# """Let's create a mask of the image so we can transpose it unto the frame"""
			# # What is thresholding? From the documentation:
			# # "If pixel value is greater than a threshold value, it is assigned one value (may be white), 
			# # else it is assigned another value (may be black). 
			# # The function used is cv2.threshold. 
			# # First argument is the source image, which should be a grayscale image. 
			# # Second argument is the threshold value which is used to classify the pixel values."
			# # Threshold Binary basically means it's either 1 or 0. In other words, if it's leaning black
			# # color it black (value of 10). If leaning white, color it white.
			# ret, mask = cv2.threshold(gray_overlay, 0, 255, cv2.THRESH_BINARY)

			# # This should be straight forward. It's a bit wise masking condition, where we want to capture
			# # the 'black' (empty space) region - which is mask inv
			# mask_inv = cv2.bitwise_not(mask)

			# # Let's do some image arithmatics on dummy masks
			# frame_img = cv2.bitwise_and(roi,roi,mask = mask_inv)
			# overlay_img = cv2.bitwise_and(overlay,overlay,mask = mask)

			# # NOW, we know that SVG's have 4 channels (RGB + A), but video frames have only 3 (RGB).
			# # In order to add them, we need them to be of equal matrix size. Therefore, we should add
			# # an extra channel vector for the frame_img OR remove the alpha vector from overlay.

			# # Easier to remove the overlay's alpha
			# b_channel, g_channel, r_channel, a_channel = cv2.split(overlay_img)
			# # If you want to create dummy alpha channel, you can use the following command
			# # alpha_channel = np.ones((185, 198)) * 50
			# overlay_rgb = cv2.merge((b_channel, g_channel, r_channel))

			# # Apply slight blur if needed
			# overlay_rgb = cv2.GaussianBlur(overlay_rgb,(11,11),0)
			# # overlay_rgb = cv2.bilateralFilter(overlay_rgb,9,75,75)

			# # Sum both images given the bit operation - both images MUST be same size AND same channel size
			# dst = cv2.add(frame_img,overlay_rgb)

			# transpose the image blend into the frame
			# frame[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst

			# # The following code basically overlays the image within the frame's 'matrix', and ensure
			# # the transperancy regions are respected within all color regions
			# # To be more concrete, the information you're reading is:
			# # frame[x,y,color], where you can split a single pixel into 4 regions:
			# # c = 0 is Blue
			# # c = 1 is Green
			# # c = 2 is Red
			# # c = 3 is Alpha (transparency)

			# Create the frame ONLY if it's within the range of the video frame. For more detail on what's going on
			# here, please read the commented section on top
			for c in range(0,3):
				if (y_offset > 0 and x_offset > 0) and (y_offset + overlay_size < frame.shape[0] and x_offset + overlay_size < frame.shape[1]):
					frame[y_offset:y_offset+overlay.shape[0], x_offset:x_offset+overlay.shape[1], c] =  overlay[:,:,c] * (overlay[:,:,3]/255.0) + frame[y_offset:y_offset+overlay.shape[0], x_offset:x_offset+overlay.shape[1], c] * (1.0 - overlay[:,:,3]/255.0)
		else:

			# Create a simple square for others
			cv2.rectangle(frame, (pntA[0]-tolerance, pntA[1]-tolerance), (pntB[0] + tolerance, pntB[1] + tolerance), color, thickness=4, lineType=8, shift=0)
		
		# Draw the text (ADMIN or ASSET)
		# Selected font
		font = cv2.FONT_HERSHEY_DUPLEX

		# Define the position of text
		midpoint = (pntA[1]-tolerance) + ((pntB[1] + tolerance) - (pntA[1]-tolerance))/2 + 10

		# Put the text
		cv2.putText(frame,text,(pntB[0] + tolerance + 20, midpoint), font,1,color)

	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()