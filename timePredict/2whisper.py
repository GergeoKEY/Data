import pandas as pd
import time
from pydub import AudioSegment
import whisper
from ModelTrainer2 import ModelTrainer
import os
import numpy as np
import librosa
import torch
import glob



class SpeechRecognitionTool:
    def __init__(self, whisper_model_size="base"):
        print(f"加载 Whisper 模型（{whisper_model_size}）中...")
        self.model_size = whisper_model_size
        start_time = time.perf_counter()
        self.model = whisper.load_model(whisper_model_size)
        self.model_load_time = time.perf_counter() - start_time
        print(f"模型加载完成。加载时间: {self.model_load_time:.2f} 秒")
        
        self.data_path = "data/whisper_data.csv"
        self.model_path = "xgboost_model/1whisper_final.json"
        self.load_model_path = "xgboost_model/1whisper_load_model.json"
        self.execution_data = self.load_execution_data()
        
        # 特征
        self.features = [
            "duration",  # 音频时长
            "audio_entropy",  # 音频熵（复杂度）
            "audio_energy",  # 音频能量
        ]
        
        # 执行时间预测模型
        self.model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="process_time",
            model_path=self.model_path
        )
        
        # 加载时间预测模型
        self.load_model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="load_time",
            model_path=self.load_model_path
        )
        
        self.model_trained = False
        self.load_model_trained = False

    def measure_model_load_time(self, repeat=3):
        """测量模型加载时间，多次测量取平均值"""
        load_times = []
        for _ in range(repeat):
            # 清除之前的模型
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 测量加载时间
            start_time = time.perf_counter()
            self.model = whisper.load_model(self.model_size)
            load_time = time.perf_counter() - start_time
            load_times.append(load_time)
            
            print(f"模型加载时间: {load_time:.2f} 秒")
            
        # 返回平均加载时间
        avg_load_time = sum(load_times) / len(load_times)
        print(f"平均模型加载时间: {avg_load_time:.2f} 秒")
        return avg_load_time

    def calculate_audio_features(self, audio_path):
        """计算音频特征"""
        try:
            # 使用pydub获取基本信息
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000  # 秒
            
            # 使用librosa计算音频熵和能量
            y, sr = librosa.load(audio_path)
            # 将音频信号转换为振幅谱
            S = np.abs(librosa.stft(y))
            # 归一化振幅谱
            p = S / np.sum(S)
            # 计算音频熵（避免log2(0)）
            audio_entropy = -np.sum(p * np.log2(p + 1e-10))
            audio_energy = np.sum(y**2)  # 计算音频能量
            
            return {
                "duration": duration,
                "audio_entropy": audio_entropy,
                "audio_energy": audio_energy
            }
        except Exception as e:
            print(f"计算音频特征时出错: {str(e)}")
            return None

    def speech_to_text(self, file_path: str, measure_load=False):
        try:
            print(f"Processing: {file_path}")
            
            # 计算音频特征
            audio_features = self.calculate_audio_features(file_path)
            if audio_features is None:
                return None
            
            # 测量加载时间（如果需要）
            load_time = 0
            if measure_load:
                load_time = self.measure_model_load_time()
            
            # 记录实际处理时间
            process_start_time = time.perf_counter()
            result = self.model.transcribe(file_path)
            process_time = time.perf_counter() - process_start_time
            print(f"音频处理时间: {process_time:.2f} 秒")
            
            # 总执行时间
            execution_time = load_time + process_time
            text_res = result["text"]
            print(f"识别内容：{text_res}")
            print(f"总执行时间: {execution_time:.2f} 秒")
            
            # 记录所有时间信息
            self.execution_data.append({
                **audio_features,  # 音频特征
                "load_time": load_time,      # 模型加载时间
                "process_time": process_time, # 实际处理时间
                "execution_time": execution_time  # 总执行时间
            })
            
            self.save_execution_data()
            return text_res
            
        except Exception as e:
            print(f"[Error] {file_path}: {str(e)}")
            return None

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
            print("数据量不足，无法训练模型")
            return False
            
        # 训练处理时间预测模型
        success = self.model_trainer.train_model(df)
        self.model_trained = success
        
        # 检查是否有足够的加载时间数据
        if 'load_time' in df.columns and df['load_time'].sum() > 0:
            # 训练加载时间预测模型
            print("训练加载时间预测模型...")
            load_success = self.load_model_trainer.train_model(df)
            self.load_model_trained = load_success
            print(f"加载时间模型训练{'成功' if load_success else '失败'}")
        else:
            print("没有足够的加载时间数据，跳过加载时间模型训练")
        
        return success

    def predict_process_time(self, duration: float, audio_entropy: float = 0.0, 
                            audio_energy: float = 0.0) -> float:
        """预测纯处理时间（不包括模型加载）"""
        if not self.model_trained:
            print("模型未训练，请先训练模型")
            return None
            
        # 创建特征数组，确保是一维的
        features = np.array([[duration, audio_entropy, audio_energy]])
            
        # 预测处理时间
        return round(self.model_trainer.predict(features), 2)
    
    def predict_load_time(self, duration: float, audio_entropy: float = 0.0, 
                         audio_energy: float = 0.0) -> float:
        """预测模型加载时间"""
        if not self.load_model_trained:
            print("加载时间模型未训练，请先训练模型")
            return None
            
        # 创建特征数组，确保是一维的
        features = np.array([[duration, audio_entropy, audio_energy]])
            
        # 预测加载时间
        return round(self.load_model_trainer.predict(features), 2)
        
    def predict_total_time(self, duration: float, audio_entropy: float = 0.0, 
                          audio_energy: float = 0.0) -> float:
        """预测总时间（模型加载+处理）"""
        process_time = self.predict_process_time(duration, audio_entropy, audio_energy)
        
        # 如果加载时间模型已训练，则加上预测的加载时间
        if self.load_model_trained:
            load_time = self.predict_load_time(duration, audio_entropy, audio_energy)
            return round(process_time + load_time, 2)
        # 否则使用实际测量的加载时间
        else:
            return round(process_time + self.model_load_time, 2)


def main():
    tool = SpeechRecognitionTool(whisper_model_size="base")

    # # 处理GAIA音频文件集
    # print("Processing audio files...")
    
    # # 查找所有音频文件
    # audio_dir = "audio"
    # audio_files = []
    
    # # 支持的音频扩展名
    # for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff', '.opus']:
    #     audio_files.extend(glob.glob(os.path.join(audio_dir, f"*{ext}")))
    
    # # 处理所有找到的音频文件
    # for file in audio_files:
    #     # 每次都测量加载时间
    #     measure_load = True
    #     result = tool.speech_to_text(file, measure_load=measure_load)
    #     print(f"Text: {result}")

    # 训练模型
    print("Training model...")
    if tool.train_model():
        print("模型训练成功")
    else:
        print("模型训练失败")
        return

    # 执行预测
    print("Predicting execution time...")
    
    # 预测处理时间（不含加载）
    predicted_process_time = tool.predict_process_time(
        duration=22.335,
        audio_entropy=17.058574676513672,  # 示例值
        audio_energy=1467.9956    # 示例值
    )
    
    # 预测总时间（含加载）
    predicted_total_time = tool.predict_total_time(
        duration=22.335,
        audio_entropy=17.058574676513672,  # 示例值
        audio_energy=1467.9956    # 示例值
    )
    
    if predicted_process_time is not None:
        print(f"Predicted Process Time (without loading): {predicted_process_time} seconds")
    if predicted_total_time is not None:
        print(f"Predicted Total Time (with loading): {predicted_total_time} seconds")


if __name__ == "__main__":
    main()
