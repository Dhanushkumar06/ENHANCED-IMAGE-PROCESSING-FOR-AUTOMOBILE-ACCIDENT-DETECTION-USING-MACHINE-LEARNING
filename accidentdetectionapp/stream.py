import cv2
import pandas

import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
import vonage
import time
from django.core.mail import send_mail
from django.conf import settings
import pusher
from .models import *
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn







def check(t1,img,model):
    results = model(img)
    results.print()
    dummy_array = np.array(results.xyxy[0])
    dummy_array = dummy_array.astype(int)
    dummy_array = dummy_array[dummy_array[:,0].argsort()]
    return detect(dummy_array,t1)    

def send_notification(id,flag):
    pusher_client = pusher.Pusher(
    app_id='1328110',
    key='4da6311b184ace45d1dc',
    secret='469709e6b17fadfab16f',
    cluster='ap2',
    ssl=True
    )
    if flag:
        notif = Notifications(notification="accident happened",lattitude=47.5,longitude=122.33,accepted=0)
        
        notif.save()
    if id==1:
        pusher_client.trigger('my-channel', 'my-event', {'message2': 'Urgent\n please send ambulance as soon as possible at xyz address.'})
    if id==2:
        pusher_client.trigger('my-channel', 'my-event', {'request': 'Request Sent'})
    return


def detect(boxes,t1):
    n = len(boxes)
    for i in range(n):
        x1,y1,w1,h1 = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        for j in range(i+1,n):
            x2,y2,w2,h2 = boxes[j][0],boxes[j][1],boxes[j][2],boxes[j][3]
            if x2<(w1):
                xmin = min(x1,x2)
                xmax = max(w1,w2)
                ymin = min(y1,y2)
                ymax = max(h1,h2)
                print(xmin,xmax,ymin,ymax)
                print((xmin>=t1[0] & t1[0]<=xmax) | (xmin>=t1[2] & t1[2]<=xmax))
                print((ymin>=t1[1] & t1[1]<=ymax) | (ymin>=t1[3] & t1[3]<=ymax))
                if ((xmin>=t1[0] & t1[0]<=xmax) | (xmin>=t1[2] & t1[2]<=xmax)) & ((ymin>=t1[1] & t1[1]<=ymax) | (ymin>=t1[3] & t1[3]<=ymax)):
                    # print("are you here")
                    return True


    return False
from twilio.rest import Client
from geopy.geocoders import Nominatim
import geocoder
def send_message():
    geoLoc = Nominatim(user_agent="GetLoc")
    g = geocoder.ip('me')
    locname = geoLoc.reverse(g.latlng)
    account_sid = 'AC44faf901d3b67a3d78bc2e86172cf227'
    auth_token = '1792fda1352f38b5c6c6debb49695379'
    client = Client(account_sid, auth_token)
    client.messages.create(
                 body="Accident detected in "+"IFET College of Engineering https://goo.gl/maps/iYViWwEw1WNmPfwYA  ",
                 from_= '+12515720222',
                 to= '+919361454677' )


from django.core.mail import EmailMessage
from django.conf import settings

def sendmail():
    hospital = Hospital.objects.all()
    print(hospital.values())
    gmail_list = []
    
    for hos in hospital.values():
        gmail_list.append(hos['email'])
    
    # Define email subject and body
    email_subject = "Urgent please send ambulance."
    email_body = "Accident happened at IFET College of Engineeing, send ambulance as soon as possible. Google map link :- https://goo.gl/maps/iYViWwEw1WNmPfwYA"
    
    # Create an EmailMessage object
    email = EmailMessage(
        subject=email_subject,
        body=email_body,
        from_email=settings.EMAIL_HOST_USER,
        to=gmail_list,
    )
    
    # Attach an image to the email (replace 'path_to_your_image.jpg' with the actual path to your image)
    email.attach_file('C:\\Users\\Acer\\Desktop\\Accdetection\\image.jpg')
    
    # Send the email
    email.send()
    
    print("Email sent successfully.")

class streaming(object):
    def __init__(self):
        print("hello")
        self.flag=True
        self.video_capture = cv2.VideoCapture(0)
        # self.video_capture = cv2.VideoCapture("C:\\Users\\LENOVO\\Downloads\\accident3.mp4")
        self.model1=torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Acer\\Desktop\\Accdetection\\best (2).pt',device='cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\Acer\\Desktop\\Accdetection\\accident2.pt',device='cpu')
# model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\hp\\Desktop\\accident.pt')

    def get_frame(self):
        ret, frame = self.video_capture.read()
        cv2.imwrite("image.jpg",frame)
        imgs = cv2.imread("image.jpg")
        results = self.model1(imgs)
        # results.show()
        get = results.print()
        # print(get)

        dummy_array = np.array(results.xyxy[0])
        dummy_array = dummy_array.astype(int)
        # results = self.model(imgs)
        # dummy_array = np.array(results.xyxy[0])
        # dummy_array = dummy_array.astype(int)
        # # print(dummy_array[0])
        # dummy_array = dummy_array[dummy_array[:,0].argsort()]
        # print(dummy_array[0])
        # result= detect(dummy_array,t1)
        # coordinates=[]
        # df=results.pandas().xyxy[0]
        # if df.shape[0]!=0:
        #     print("*"*80)
        #     # print(df.head())
        #     print("Accident Happened")
        #     for i in range(df.shape[0]):
        #         # print((df.loc[i]['xmin']),df.loc[i]['ymin'], df.loc[i]['xmax'],df.loc[i]['ymax'])
        #         frame = cv2.rectangle(frame, (int(df.loc[i]['xmin']),int(df.loc[i]['ymin'])), (int(df.loc[i]['xmax']),int(df.loc[i]['ymax'])), (0,0,255), 2)
        #     # coordinates.append(results.pandas().xyxy[0].iloc[:,:-3])
        #     print("*"*80)
        # print("*"*80)
        # print(results.pandas().xyxy[0].head())
        # print(results.pandas().xyxy[0].shape)
        # print("*"*80)
        # # frame=cv2.imread(results.pandas().xyxy[0])
        # print(coordinates)
        # print(coordinates[0])

        if ret==False:
            pass
        else:
            jpeg = cv2.imencode('.jpg', frame)[1]
            # send_message()
            # sendmail()
            # time.sleep(10)
            # frame,her = vid.read()
        # cv.imshow("hekk",frame)
            
	# /print(dummy_array)
            for i in dummy_array:
                if check(i,imgs,self.model) and self.flag:
                    print("&"*40)
                    print("accident")
                    send_message()
                    send_notification(1,True)
                    send_notification(2,False)
                    sendmail()
                    self.flag = False
            # update_data()
            return jpeg.tobytes()
    


    
# <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d8641.14378656012!2d79.61492387888401!3d11.919254287645979!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3a5358fe9fd583c5%3A0xbc72123191afd4f1!2sIFET%20College%20of%20Engineering%20(Autonomous%20Institution)!5e0!3m2!1sen!2sin!4v1693992860565!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>