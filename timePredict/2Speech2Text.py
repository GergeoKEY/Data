import pandas as pd
import time
import speech_recognition as sr
from pydub import AudioSegment
from ModelTrainer2 import ModelTrainer
import os
# 设置要使用的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # 指定使用 GPU 2 和 3
class SpeechRecognitionTool:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # 确保数据目录存在
        os.makedirs("data", exist_ok=True)
        os.makedirs("xgboost_model", exist_ok=True)
        
        self.data_path = "data/500audio2Text.csv"
        self.model_path = "xgboost_model/audio2Text_model.json"
        self.execution_data = self.load_execution_data()
        self.model_trainer = ModelTrainer(
            feature_columns=["duration"],
            target_column="execution_time",
            model_path=self.model_path
        )
        self.model_trained = False

    def speech_to_text(self, file_path: str):
        try:
            audio_seg = AudioSegment.from_file(file_path)
            duration = len(audio_seg) / 1000  

            with sr.AudioFile(file_path) as source:
                print(f"Processing: {file_path}")
                audio = self.recognizer.record(source)
                start_time = time.perf_counter()
                text_res = self.recognizer.recognize_sphinx(audio)
                execution_time = time.perf_counter() - start_time
        except Exception as e:
            print(f"[Error] {file_path}: {str(e)}")
            text_res = None
            execution_time = 0.0
            duration = 0.0

        self.execution_data.append({
            "duration": duration,
            "execution_time": execution_time
        })
        self.save_execution_data()
        return text_res

    def save_execution_data(self):
        df = pd.DataFrame(self.execution_data)
        df.to_csv(self.data_path, index=False)

    def load_execution_data(self):
        if os.path.exists(self.data_path):
            return pd.read_csv(self.data_path).to_dict("records")
        return []

    def train_model(self):
        df = pd.DataFrame(self.execution_data)
        if len(df) < 5:
            print("Not enough data to train the model.")
            return False
        success = self.model_trainer.train_model(df)
        self.model_trained = success
        return success

    def predict_execution_time(self, duration: float) -> float:
        if not self.model_trained:
            print("Model not trained. Please train the model first.")
            return None
        return round(self.model_trainer.predict([duration]), 2)

def main():
    tool = SpeechRecognitionTool()

    # 训练模型
    print("Training model...")
    if tool.train_model():
        print("Model trained successfully.")
        
        # 预测执行时间
        print("Predicting execution time for 121s audio...")
        predicted_time = tool.predict_execution_time(121)
        if predicted_time is not None:
            print(f"Predicted Time: {predicted_time} seconds")
    else:
        print("Model training failed. Please collect more data first.")

if __name__ == "__main__":
    main()
