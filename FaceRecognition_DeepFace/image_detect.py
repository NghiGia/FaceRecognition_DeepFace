import cv2
import mediapipe as mp
from deepface import DeepFace
from deepface.commons import distance as dst
import json
import time
import os
import re




start=time.time()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

distances = []
distances_threshold=[]
employee=[]
employees=[]
employee_threshold=[]
region_obj = []
extracted_face=[]
extracted_faces=[]
index=0
count=0
count_match=0
count_unmatch=0
count_notfound=0
count_90 =0
count_80 =0
count_70 =0


pattern_order = r'[0-9]'
json_file_path = "converted_db.json"
model_name = "VGG-Face"
distance_metric = "cosine"

with open(json_file_path, 'r') as j:
    source_representation = json.loads(j.read())
print(len(source_representation))
# for i in range(0,len(source_representation)):
#     print(source_representation[i]["embedding"])

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

file_path="C:/Users/Admin/Pictures/Camera Roll"

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
         pass

if len(employees) != 0:
    count=0
    for each in  employees:
        image = cv2.imread(each)
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
                        cv2.imshow("test",crop_img)

                        try:
                            target_representation = DeepFace.represent_custom2(crop_img, img_name="")
                            # print(target_representation["embedding"])
                            # print(len(target_representation))
                            for i in range(0, len(source_representation)):
                                distance = dst.findCosineDistance(source_representation[i]["embedding"],
                                                                  target_representation["embedding"])
                                employee.append(source_representation[i]["name"])
                                distances.append(distance)
                            # distances = []
                            threshold = dst.findThreshold(model_name, distance_metric)
                            for i in range(0, len(distances)):
                                if (distances[i] <= threshold):
                                    employee_threshold.append(employee[i])
                                    distances_threshold.append(distances[i])
                            temp_min = distances_threshold[0]
                            for i in range(1, len(distances_threshold)):
                                if distances_threshold[i] < temp_min:
                                    temp_min = distances_threshold[i]
                                    index = i

                            print(str(count) + "/" + str(len(employees)) + " " + re.sub('\W+', '', re.sub(pattern_order, '',  os.path.splitext( os.path.basename(each))[0])).replace("_","") + " " + re.sub('\W+', '', re.sub(pattern_order, '', employee_threshold[index])).replace("_", ""))
                            if re.sub('\W+', '', re.sub(pattern_order, '', os.path.splitext(os.path.basename(each))[0])).replace("_","") == re.sub('\W+', '', re.sub(pattern_order, '', employee_threshold[index])).replace("_", ""):
                                count_match += 1
                                percent=(threshold - temp_min) / threshold
                                if percent >=0.9 :
                                    count_90 +=1
                                elif percent >=  0.8 and percent < 0.9:
                                    count_80 +=1
                                elif percent >=  .07 and percent < 0.8:
                                    count_70 +=1
                            else:
                                count_unmatch += 1
                            datashow = employee_threshold[index] + str((threshold - temp_min) / threshold)
                            cv2.putText(image, datashow, (xtop, xleft), 0, 2, 255)
                            cv2.imwrite("C:/Users/Admin/Desktop/Python/FaceRecognition_DeepFace/train_db/Extract_Team_SPS/"+re.sub('\W+', '', re.sub(pattern_order, '',  os.path.splitext( os.path.basename(each))[0])).replace("_","")+"_"+str(count)+".jpg",image)
                            # print("C:/Users/Admin/Desktop/Python/FaceRecognition_DeepFace/train_db/Extract_Team_SPS/"+re.sub('\W+', '', re.sub(pattern_order, '',  os.path.splitext( os.path.basename(each))[0])).replace("_","")+"_"+str(count)+".jpg")
                            count += 1
                            distances = []
                            distances_threshold = []
                            employee_threshold = []
                            employee = []
                            index = 0
                        except:
                            cv2.putText(image, "Unknown", (xtop, xleft), 0, 2, 255)
                            print(str(count) + "/" + str(len(employees)) + " " + re.sub('\W+', '',re.sub(pattern_order, '',os.path.splitext(os.path.basename(each))[0])).replace("_","") + " " + "Not found")
                            count_notfound += 1
                            count += 1
            except:
                print("Not find")



print("Unmatch: " +str(count_unmatch))
print("Match: " +str(count_match))
print("Not found: " +str(count_notfound))
print(">=90: " +str(count_90))
print(">=80: " +str(count_80))
print(">=70: " +str(count_70))
print("The time of execution of above program is :",
      (time.time()-start) * 10**3, "ms")