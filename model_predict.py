from ultralytics import YOLO


# Load pretrained model Path
#model = YOLO('/Users/seunghunjang/Desktop/WOOTD/20240430/Train_Model/train/weights/best.pt')


# Load Pretrained BGR Model Path
model_BGR = YOLO('/Users/seunghunjang/Desktop/WOOTD/20240430/Train_Model/Epoch 25 Model 2/train/weights/best.pt')


# Predict Image Path
predic_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR_X'
predic_BGR_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR'


# Top Bottom Combination Path
top_bottom_path = '/Users/seunghunjang/Desktop/WOOTD/Top_Bottom_Combination'

# 20240430~ Test
test_path = '/Users/seunghunjang/Desktop/WOOTD/20240430/Predict_Results'

# Save Dir Path
result_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR_X/Predict'
result_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR/BGR_Predict'
result_combination_path = '/Users/seunghunjang/Desktop/WOOTD/results/Combination'


# None Remove BackGround Model Test
#result = model.predict(source=predic_img_path, save=True, save_txt=True, project=result_path)

# Remove BackGround Model Test
#result_bgr = model_BGR.predict(source=predic_BGR_img_path, save=True, save_txt=True, project=result_bgr_path)

# Model Test Only Top/Bottom Clothes Combination
result_combination = model_BGR.predict(source=top_bottom_path, save=True, save_txt=True, project=test_path)
