from ultralytics import YOLO

# Model Load
model_BGR = YOLO('yolov8n-cls_BGR.pt')


# Image Path
img_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/Dataset_BGR'

# Other Wannabe Style Dataset
otherStyle_Dataset = 'OtherStyle_Dataset'


# Model Save Path
save_model_path = 'results/BGR/Train_BGR_Models'


# BGR Model Train
result_bgt = model_BGR.train(data=otherStyle_Dataset, epochs=50, cache=True, imgsz=480, project=save_model_path)