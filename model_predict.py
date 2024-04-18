from ultralytics import YOLO


# Load pretrained model
model = YOLO('/Users/seunghunjang/Desktop/WOOTD/results/BGR_X/Train_Models/train3/weights/best.pt')


# Load Pretrained BGR Model
model_BGR = YOLO('/Users/seunghunjang/Desktop/WOOTD/results/BGR/Train_BGR_Models/train3/weights/best.pt')


# Predict Image Path
predic_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR_X'
predic_BGR_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR'


# Save Dir
result_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR_X/Predict'
result_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR/BGR_Predict'


# Model Test
result = model.predict(source=predic_img_path, save=True, save_txt=True, project=result_path)


# Model Test
result_bgr = model.predict(source=predic_BGR_img_path, save=True, save_txt=True, project=result_bgr_path)