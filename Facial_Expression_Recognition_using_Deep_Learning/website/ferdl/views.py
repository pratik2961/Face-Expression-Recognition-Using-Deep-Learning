from django.http import HttpResponse
from django.shortcuts import render , redirect
from .models import Registrationtable , Feedback
from django.contrib.sessions.models import Session
from django.contrib import messages
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os

def index(request):
    return render(request,'index.html')
    
def register(request):
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')

def home(request):
    if request.session.has_key('mobile'):
        return render(request,'home.html')
    else:
        return render(request,'index.html')

def registerdata(request):
    savedata = Registrationtable()
    savedata.username = request.POST.get('u_name')
    savedata.password = request.POST.get('password')
    savedata.mobile_no = request.POST.get('mobile')
    savedata.gender = request.POST.get('gender')
    try:
        savedata.save()
        request.session['mobile'] = request.POST['mobile']
        #messages.success(request,"Registred successfully")
        return render(request,'home.html')
    except:
        messages.error(request," Something went wrong ")
        return render(request,'index.html')

def loginin(request):
    try:
        password = request.POST['password']
        mobile = request.POST['mobile']
        x = Registrationtable.objects.get(mobile_no = mobile)
        if password == x.password and mobile == x.mobile_no:
            #messages.success(request,"Logined successfully")
            request.session['mobile'] = request.POST['mobile']
            if request.session.has_key('mobile'):
                return render(request,'home.html')
            else:
                messages.error(request," Login first ")
                return render(request,'index.html')

        else:
            messages.error(request,"Login is failed ")
            return render(request,'index.html')

    except:
        messages.error(request," Something went wrong ")
        return render(request,'index.html')

def opencamera(request):
    try:
        face_classifier = cv2.CascadeClassifier(r'C:\Users\Parshav Panchal\Desktop\website\website\ferdl\haarcascade_frontalface_default.xml')
        classifier =load_model(r'C:\Users\Parshav Panchal\Desktop\website\website\ferdl\Emotion_little_vgg.h5')

        class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

        cap = cv2.VideoCapture(0)

        while True:
            # Grab a single frame of video
            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)
            
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imshow('Press q for exit',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       
        cap.release()
        cv2.destroyAllWindows()
        return render(request,'home.html')
    except:
            messages.error(request," Something went wrong ")
            return render(request,'home.html')

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
            
def mask(request):
    try:
        prototxtPath = r"C:\Users\Parshav Panchal\Desktop\website2\website\ferdl\deploy.prototxt"
        weightsPath = r"C:\Users\Parshav Panchal\Desktop\website2\website\ferdl\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        maskNet = load_model(r"C:\Users\Parshav Panchal\Desktop\website2\website\ferdl\mask_detector.model")

        # initialize the video stream
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Press q for exit", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
        return render(request,'home.html')
    except:
        messages.error(request," Something went wrong ")
        return render(request,'home.html')    
def userprofile(request):
    if request.session.has_key('mobile'):
        mobile = request.session['mobile']
        x = Registrationtable.objects.get(mobile_no = mobile)
        return render(request,'profile.html',{'user':x})
    else:
        messages.error(request," Login first ")
        return render(request,'index.html')

def feedback(request):
    if request.session.has_key('mobile'):
        return render(request,'feedback.html')
    else:
        messages.error(request," Login first ")
        return render(request,'index.html')

def delete_account(request):
    if request.session.has_key('mobile'):
        return render(request,'deleteaccount.html')
    else:
        messages.error(request," Login first ")
        return render(request,'index.html')

def logout(request):
    if request.session.has_key('mobile'):
        del request.session['mobile']
        messages.error(request,"You are logouted successfully")
        return render(request,'index.html')
    else:
        messages.error(request," Login first ")
        return render(request,'index.html')

def submitfeedback(request):
    try:
        if request.POST.get('submit'):
            if request.session.has_key('mobile'):
                if request.session.has_key('mobile'):
                    savefeedback = Feedback()
                    mobile = request.session['mobile']
                    userid = Registrationtable.objects.get(mobile_no = mobile)
                    savefeedback.user_id = userid.id
                    savefeedback.feedback = request.POST.get('feedback')
                    if savefeedback.feedback == "" or savefeedback.feedback.isspace():
                        messages.error(request, " Write something in feedback first then submit  ")
                        return render(request, 'home.html')
                    else:
                        savefeedback.save()
                        return render(request, 'home.html')

            else:
                messages.error(request," Login first ")
                return render(request,'index.html')
        if request.POST.get("back"):
            return render(request, 'home.html')
    except:
        messages.error(request, " Something went wrong ")
        return render(request, 'index.html')
   
def accountdel(request):
    if request.session.has_key('mobile'):
        try:
            if request.session.has_key('mobile'):
                mobile = request.POST.get('mobile')
                password = request.POST.get('password')
                x = Registrationtable.objects.get(mobile_no = mobile)
                if mobile == request.session['mobile']:
                    if mobile == x.mobile_no and password == x.password:
                        deleteacc = Registrationtable.objects.get(mobile_no = mobile)
                        deleteacc.delete()
                        del request.session['mobile']
                        messages.success(request,"Deleted successfully")
                        return render(request,'index.html')
                    else:
                        messages.error(request,"Mismatch information")
                        return render(request,'home.html')
                else:
                    messages.error(request,"Mismatch mobile number")
                    return render(request,'home.html')
        except:
            messages.error(request,"Mismatch information")
            return render(request,'home.html')
    else:
        messages.error(request," Login first ")
        return render(request,'index.html')



