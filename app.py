import cv2
import streamlit as st
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from pymongo import MongoClient

cluster=MongoClient("mongodb+srv://admin:admin@cluster0.nx86a.mongodb.net/FirstDatabase?retryWrites=true&w=majority")
db = cluster["FirstDatabase"]
collection=db["logIn"]
collectionLogOut=db["logOut"]
collectionEmployees=db["employees"]

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }



col1,col2=st.columns(2)
c1,c2,c3=st.columns(3)
sidebar=st.sidebar.selectbox("select Operation",["Employee Registration","Track Employee","Track Information"])
container=st.container()





def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def tick():
    time_string = time.strftime('%H:%M:%S')


def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        st.error('Some file missing')



def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids



def takeImages(Id,name):
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImages/")

    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")

    FRAME_WINDOW = st.image([])

    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImages\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame

                FRAME_WINDOW.image(gray)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 100:
                TrainImages(serial,name,Id)
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()



def TrainImages(serial,name,Id):
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    faces, ID = getImagesAndLabels("TrainingImages")

    try:
        recognizer.train(faces, np.array(ID))
    except:
        st.error("Profile Not Saved !!!")
        return

    recognizer.save("TrainingImageLabel\Trainner.yml")
    collectionEmployees.insert_one({"serialNo": serial, "EmployeeId": Id, "name": name})
    st.success("Profile Saved Successfully")


def TrackImages(stop):
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        st.error("Data Missing")
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        st.error("Details Missing")
        cam.release()
        cv2.destroyAllWindows()

    FRAME_WINDOW = st.image([])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]

                st.success("Employee with " + ID + " Is Logged In Suucesfully")
                collection.insert_one({"EmployeeId": ID, "Name": bb, "status": "logIn", "date": str(date),
                                       "time": str(timeStamp)})
                time.sleep(15)
            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        FRAME_WINDOW.image(im)
        if (cv2.waitKey(1) == ord('q')) or stop:
            break

    cam.release()
    cv2.destroyAllWindows()


def logOutTracking(stop):
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImages")

    try:
        recognizer.train(faces, np.array(ID))
    except:
        st.error("Please Register someone first!!!")
        return

    recognizer.save("TrainingImageLabel\Trainner.yml")
    st.success("Profile Saved Successfully")


def logOutTracking(stop):
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")


    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        st.error("Data Missing")
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        st.error("Details Missing")
        cam.release()
        cv2.destroyAllWindows()

    FRAME_WINDOW = st.image([])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]

                st.success("Employee with " + ID + " Is Logged Out Suucesfully")
                collectionLogOut.insert_one({"EmployeeId": ID, "Name": bb, "status": "logOut", "date": str(date),
                                       "time": str(timeStamp)})
                time.sleep(15)
            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        FRAME_WINDOW.image(im)
        if (cv2.waitKey(1) == ord('q')) or stop:
            break

    cam.release()
    cv2.destroyAllWindows()



if sidebar=="Employee Registration":
    st.title("Employee Registration")
    with container:
        run,stop,save=False,False,False
        Id,name="",""

        with col1:
            res = 0
            exists = os.path.isfile("StudentDetails\StudentDetails.csv")
            if exists:
                with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for l in reader1:
                        res = res + 1

                csvFile1.close()
            else:
                res = 0

            st.text('Total Registrations till now  : ' + str(res-4))
            Id= st.text_input('Employee ID')
            name=st.text_input("Enter Employee Name")
            run = st.button("Track Image")
        with col2:
            if run:
                takeImages(Id,name)



if sidebar=="Track Employee":
    st.title("Track Employees")
    with container:
        with c1:
            logIn = st.button("Log In")
        with c2:
            logOut=st.button("Log Out")
        with c3:
            stop = st.button("Stop Tracking")


    if logIn:
        TrackImages(stop)
    if logOut:
        logOutTracking(stop)

if sidebar=="Track Information":
    st.subheader("Employee Track Information")

    value=st.selectbox("Select Info",["logIn","logOut"])
    if value=="logIn":
        x = collection.find()
        data=[]
        d=st.date_input("Select The date").strftime('%d-%m-%Y')

        if d:
            for i in x:
                if d==i["date"]:
                    data.append(i)

        df = pd.DataFrame(
           data,
            columns=(["EmployeeId","Name","status","date","time"]))
        st.table(df)

    if value == "logOut":
        st.subheader("wait")






