from ultralytics import YOLO

# Model Load
model = YOLO('yolov8n-cls.pt')
model_BGR = YOLO('yolov8n-cls_BGR.pt')


# Image Path
img_path = '/Users/seunghunjang/Desktop/WOOTD/Dataset'
img_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/Dataset_BGR'

# Result Path
save_model_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR_X/Train_Models'

# BGR Model Save Path
save_bgr_model_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR/Train_BGR_Models'


# Model Train
result = model.train(data=img_path, epochs=10, cache=True, imgsz=480, project=save_model_path)

# BGR Model Train
result_bgt = model_BGR.train(data=img_bgr_path, epochs=10, cache=True, imgsz=480, project=save_bgr_model_path)