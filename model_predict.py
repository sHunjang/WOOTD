from ultralytics import YOLO



# Load Pretrained BGR Model Path
model = YOLO('')


# Predict Image Path
predic_BGR_img_path = 'TestDataset'


# Top Bottom Combination Path
top_bottom_path = 'Top_Bottom_Combination'


# Test Combination
test_path = 'test_Combination'


# Model Test Only Top/Bottom Clothes Combination
result_combination = model.predict(source=test_path, save=True, save_txt=True, project='test')