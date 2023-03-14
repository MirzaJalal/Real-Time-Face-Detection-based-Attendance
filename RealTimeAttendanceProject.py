import numpy as np
import os
from datetime import datetime
import cv2
import face_recognition

path = 'TrainingDataset'
imageArray = []
studentNames = []

# TODO: reading images from file path
for singleImageFile in os.listdir(path):
    currentImage =cv2.imread(f'{path}/{singleImageFile}')
    imageArray.append(currentImage)
    studentNames.append(os.path.splitext(singleImageFile)[0])

print("The students are: ", studentNames)

# TODO: to find the image encoding
def computeImageEncoding(imageArray):
    encodingList = []
    for image in imageArray:
        # converting image from bgr to rgb
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodingList.append(encode)
    return encodingList
print('Encoding is Done!!')

# TODO: attendance marking
def markAttendance(studentName):
    with open('attendanceReport.csv', 'r+') as file:
        data_list = file.readlines()
        print(data_list)
        name_list_array = []
        for row in data_list:
            making_entry = row.split(',')
            name_list_array.append(making_entry[0])
        if studentName not in name_list_array:
            now = datetime.now()
            dateAndTime = now.strftime('%H:%M:%S')
            file.writelines(f'\n{studentName},{dateAndTime}')

encodingKnownList = computeImageEncoding(imageArray)
print("Total Number of Students: ", len(encodingKnownList))
print('Waiting for Capturing Image!!')

# TODO: initializing the webcam
capture = cv2.VideoCapture(0)

# TODO: loop to iterate each image frame one by one
while True:
    success, image = capture.read()
    # TODO: reducing the image size one fifth (.20) to speed up as the process is real time
    small_image = cv2.resize(image, (0, 0), None, 0.20, 0.20)
    # TODO: converting the smaller image from bgr to rgb
    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    # TODO: enabling to locate multiple faces and sending all the face locations for face encoding in next step
    facesInCurrentFrame = face_recognition.face_locations(small_image)
    # TODO: image encoding
    currentFrameEncodings = face_recognition.face_encodings(small_image, facesInCurrentFrame) #CNN
    face_recognition.face_encodings(image)

    # TODO: one by one it will grab one face location from faces current frame list...
    # ... and it will grab the encoding of encode face from encodes current face
    for encodeFace, faceLocation in zip(currentFrameEncodings, facesInCurrentFrame):
        matchings = face_recognition.compare_faces(encodingKnownList, encodeFace)
        # TODO: calculating each face distance and the lowest face distance probability is the best match
        faceDistance = face_recognition.face_distance(encodingKnownList, encodeFace)

        # TODO: list of face distance
        print(faceDistance)

        matchIndex = np.argmin(faceDistance)

        # TODO: bounding box creation and writing the name
        if matchings[matchIndex]:
            studentName = studentNames[matchIndex].upper()
            print(studentName)

            y1,x2,y2,x1 = faceLocation
            # TODO: multiplying by 5 as the small image was one fifth of actual image
            y1, x2, y2, x1 = y1*5, x2*5, y2*5, x1*5
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 255), -1) #cv2.filled can be used instead of -1
            # TODO: displaying name in the box
            cv2.putText(image, studentName, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (0, 0, 255), 3)
            markAttendance(studentName)

        else:
            print("Not Detected!")

    cv2.imshow("Web Camera", image)
    cv2.waitKey(1)
