from ultralytics import YOLO



# Load Pretrained BGR Model Path
model = YOLO('results/BGR/Train_BGR_Models/train28/weights/best.pt')


# Predict Image Path
predic_BGR_img_path = 'TestDataset'


# Top Bottom Combination Path
top_bottom_path = 'Top_Bottom_Combination'


# Test Combination
test_path = 'test_set'


# Model Test Only Top/Bottom Clothes Combination
result_combination = model.predict(source=test_path, save=True, save_txt=True, project='results/BGR/BGR_Predict')