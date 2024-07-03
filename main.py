
from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(exec_path,'house.jpg'), result_count=5)
for each_pred, each_prob in zip(predictions, probabilities):
    print(f'{each_red} : {each_prob}')

