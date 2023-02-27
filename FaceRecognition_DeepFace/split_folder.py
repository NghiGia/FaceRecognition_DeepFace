import splitfolders # or import splitfolders
input_folder = "C:/Users/Admin/Desktop/Python/FaceRecognition_DeepFace/105_classes_pins_dataset"
output = "C:/Users/Admin/Desktop/Python/FaceRecognition_DeepFace/train_db" #where you want the split datasets saved. one will be created if it does not exist or none is set

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.9, .05, .05))