import cv2
import mediapipe as mp
from deepface import DeepFace
from deepface.commons import distance as dst
import json
import numpy as np
import time

distances = []
distances_threshold=[]
employee=[]
employee_threshold=[]
region_obj = []
extracted_face=[]
extracted_faces=[]
index=0

json_file_path = "converted_db.json"

with open(json_file_path, 'r') as j:
    source_representation = json.loads(j.read())

for i in range(0,len(source_representation)):
    print(source_representation[i]["embedding"])
model_name = "VGG-Face"
distance_metric = "cosine"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            continue
        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
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
        # Flip the image horizontally for a selfie-view display.
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
                    xmin=face_data.relative_bounding_box.xmin
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
                    region_obj={xtop,xbottom,xleft,xright}
                    crop_img = image[xtop: xbottom, xleft: xright]
                    cv2.imshow('cropped', crop_img)

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
                            if (distances[i] <= 0.1):
                                employee_threshold.append(employee[i])
                                distances_threshold.append(distances[i])
                        temp_min = distances_threshold[0]
                        for i in range(1, len(distances_threshold)):
                            if distances_threshold[i] < temp_min:
                                temp_min = distances_threshold[i]
                                index = i
                        print(temp_min)

                        # print( str(1-temp_min)+"%")
                        datashow= employee_threshold[index]+str((threshold-temp_min)/threshold)
                        cv2.putText(image, datashow, (xtop, xleft), 0, 2, 255)
                        distances = []
                        distances_threshold = []
                        employee_threshold = []
                        employee = []
                        index = 0
                    except:
                        cv2.putText(image, "Unknown", (xtop, xleft), 0, 2, 255)
                        print("Not found ==========================================")

        except:
            print("Not find face")


        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)
        cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('test', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
