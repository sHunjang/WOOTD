from ultralytics import YOLO

# Model Load
model = YOLO('yolov8n-cls.pt')


# Image Path
img_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/Dataset_BGR'

# Other Wannabe Style Dataset
otherStyle_Dataset = 'OtherStyle_Dataset'


# Model Save Path
save_model_path = 'results/BGR/Train_BGR_Models'


# BGR Model Train
result_bgt = model.train(data=img_bgr_path, epochs=40, cache=True, project=save_model_path)