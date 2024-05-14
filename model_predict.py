from ultralytics import YOLO


# Load pretrained model Path
#model = YOLO('/Users/seunghunjang/Desktop/WOOTD/20240430/Train_Model/train/weights/best.pt')


# Load Pretrained BGR Model Path
model_BGR_bestPT = YOLO('20240514/Train_Model/train6/weights/best.pt')
model_BGR_lastPT = YOLO('20240514/Train_Model/train6/weights/last.pt')


# Predict Image Path
predic_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR_X'
predic_BGR_img_path = '/Users/seunghunjang/Desktop/WOOTD/TestDataset/BGR'


# Top Bottom Combination Path
top_bottom_path = '/Users/seunghunjang/Desktop/WOOTD/Top_Bottom_Combination'

# 20240430~ Test
test_path_best = '/Users/seunghunjang/Desktop/WOOTD/20240514/Predict_Results/bestPT'
test_path_last = '/Users/seunghunjang/Desktop/WOOTD/20240514/Predict_Results/lastPT'

# Save Dir Path
result_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR_X/Predict'
result_bgr_path = '/Users/seunghunjang/Desktop/WOOTD/results/BGR/BGR_Predict'
result_combination_path = '/Users/seunghunjang/Desktop/WOOTD/results/Combination'


# None Remove BackGround Model Test
#result = model.predict(source=predic_img_path, save=True, save_txt=True, project=result_path)

# Remove BackGround Model Test
#result_bgr = model_BGR.predict(source=predic_BGR_img_path, save=True, save_txt=True, project=result_bgr_path)

# Model Test Only Top/Bottom Clothes Combination
result_combination = model_BGR_bestPT.predict(source=top_bottom_path, save=True, save_txt=True, project=test_path_best)

result_combination = model_BGR_lastPT.predict(source=top_bottom_path, save=True, save_txt=True, project=test_path_last)