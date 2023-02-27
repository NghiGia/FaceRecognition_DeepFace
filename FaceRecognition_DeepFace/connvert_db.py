import os
import json
import mediapipe as mp
from deepface import DeepFace
import json
import cv2
import time
employees = []
employees_convert=[]
list_json=[]


start_time=time.time()

file_path="C:/Users/Admin/Desktop/Python/FaceRecognition_DeepFace/train_db/Team_SPS_train"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

if os.path.isdir(file_path) is not True:
    raise ValueError("Passed db_path does not exist!")
else:

    for r, _, f in os.walk(file_path):
        for file in f:
            if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
            ):
                exact_path = r + "/" + file
                employees.append(exact_path)
    if len(employees) == 0:
        raise ValueError(
            "There is no image in ",file_path," folder! Validate .jpg or .png files exist in this path.",)
    else:
         print(employees)
# if len(employees) != 0:
#     count=0
#     for each in  employees:
#         try:
#             each_employee=DeepFace.represent_custom(img_path=each)
#             employees_convert.append(each_employee)
#             print(count)
#             count+=1
#         except:
#             print('Error')
#     with open("converted_db.json", 'w') as fp:
#         json.dump(employees_convert,fp,indent=2)
#     # (os.path.splitext(os.path.basename(img_path))[0])
#     print("Saved")
# print(employees_convert)

if len(employees) != 0:
    count=0
    for each in  employees:
        image=cv2.imread(each)
        with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            try:
                if results.detections:

                    # Iterate over the found faces.
                    for face_no, face in enumerate(results.detections):
                        # Display the face number upon which we are iterating upon.
                        # print(f'FACE NUMBER: {face_no + 1}')
                        # print('---------------------------------')

                        # Display the face confidence.
                        # print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')

                        # Get the face bounding box and face key points coordinates.
                        face_data = face.location_data

                        # Display the face bounding box coordinates.
                        # print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')
                        xmin = face_data.relative_bounding_box.xmin
                        ymin = face_data.relative_bounding_box.ymin
                        width = face_data.relative_bounding_box.width
                        height = face_data.relative_bounding_box.height
                        h, w, c = image.shape
                        # print('width:  ', w)
                        # print('height: ', h)
                        xleft = face_data.relative_bounding_box.xmin * w
                        xleft = int(xleft)
                        xtop = face_data.relative_bounding_box.ymin * h
                        xtop = int(xtop)
                        xright = face_data.relative_bounding_box.width * w + xleft
                        xright = int(xright)
                        xbottom = face_data.relative_bounding_box.height * h + xtop
                        xbottom = int(xbottom)
                        region_obj = {xtop, xbottom, xleft, xright}
                        crop_img = image[xtop: xbottom, xleft: xright]
                        cv2.imshow("Test",crop_img)
                        cv2.waitKey(10)
            except:
                print(str(count) +"   Not find")
            try:
                each_employee = DeepFace.represent_custom2(img_path=crop_img,img_name=each)
                employees_convert.append(each_employee)
                print(count)
                count += 1

                with open("converted_db.json", 'r+') as fp:
                    json.dump(employees_convert, fp)
            except:
                print('Error')
print("Saved")
print("--- %s seconds ---" % (time.time() - start_time))

