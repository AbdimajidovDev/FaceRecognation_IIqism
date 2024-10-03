import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests

url = 'http://127.0.0.1:8000/'
response = requests.get(url=url)

# Bu pahtga proektimizni 1-qismi(malumotlar saqlanadigan swagger mavjud bo'lgan) url manzili yoziladi.
path = '/home/tolqinjon/PycharmProjects/Open_cv/images/'


images = []
classNames = []
mylist = os.listdir(path)
# print('mylist: ', mylist)


for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
# print("classNames: ", classNames)

# Bu funksiya berilgan tasvirlar ro'yxatini qabul qiladi va har bir tasvirdagi yuzni aniqlab, uning enkodingini oladi
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Tasvirni BGR dan RGB formatiga o'zgartirish
        encodes = face_recognition.face_encodings(img) # Tasvirdan yuz enkodinglarini olish. Enkodinglar yuz xususiyatlarining raqamli ko'rinishi bo'lib yuzni tanib olishda ishlatiladi.
        if encodes:
            encode = encodes[0]
            encodeList.append(encode)
        else:
            print("No face found in image, skipping encoding.")
    return encodeList


# Kamerage tushgan shaxs ma'lumotlari va vaqti haqida ma'lumotlarni LoginInformation nomli fayldan ko'chirib oladi
def markAttendance(name):
    with open('LoginInformation.csv', 'r+') as file:
        myDataList = file.readlines()
        nameList = []
        lst = [i.replace('\n', '') for i in myDataList]
        # print('lst: ', lst)

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # print("nameList: ", nameList)

        if name is not nameList:
            now = datetime.now()
            dtString = now.strftime(' %d %h, %Y %H:%M')
            inf = f'\n{name},{dtString}'

            # Ro'yxatda biz izlagan shaxs bo'lsa Ekranda tasdiqlanadi aks holda Bashda False ko'rinadi
            if inf.strip() not in lst:
                file.writelines(inf)
            else:
                print(False)

# Yuzni tanish jarayonida yuz malumotlarini raqamli kordinata ko'rinishida saqlanadi
encodeListKnown = findEncodings(images)
print('Encode Complate')

# Kamera yordamida real vaqt rejimida video tasvirini olish uchun ishlatiladi.
cap = cv2.VideoCapture(0)


# Kamera tasvirini har soniya olish uchun takroriy sikl bajariladi
while True:

    # Rasm muvaffaqiyatli bo'lsa succes aks holda sikl to'xtaydi
    success, img = cap.read()
    if not success:
        break

    # tasvirni qayta ishlash va tasvir rangini BGR formatidan RGB formatiga o'zgartiradi, chunki face_recognition RGB formatda ishlaydi.
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Yuzni taqqoslash. Bazadagi shaxs va real vaqtda kameradagi shaxs o'zara taqqoslanadi
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # Agar shaxs Bazaga kiritilgan bo'lsa yashil kvadratda full name chop etiladi aks holda ekranda qizil kvadratda "Access is limited!" ko'rinishida chop etiladi
        if matches[matchIndex]:
            img_name = classNames[matchIndex]
            # print('img_name: ',  img_name)

            if response.status_code == 200:
                user_data_list = response.json()
                print('user_data_list: ', user_data_list)

                for user_data in user_data_list:
                    img_info = user_data.get("image")

                    if img_info:
                        info = f"{user_data.get('last_name')} {user_data.get('first_name')}"

                    if 'images/' + img_name == str(img_info).split('.')[0]:
                        name = info

            else:
                print("Error: Unable to fetch user data")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        else:

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Access is limited!", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance("Nomalum shaxs: ")

    cv2.imshow('Webcam', img)

    # Dastur ishga tushganda "q" tugmasini bosish orqali uni yakunlash
    if cv2.waitKey(1) == ord("q"):
        break
