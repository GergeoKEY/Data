import pandas as pd
import time
import os
import numpy as np
import torch
from PIL import Image
import cv2
import librosa
from pydub import AudioSegment
from ModelTrainer import ModelTrainer
from enum import Enum
import joblib

class TaskType(Enum):
    OCR = "ocr"
    SPEECH = "speech"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"

class TaskPredictor:
    def __init__(self, task_type: TaskType):
        """
        通用任务预测工具
        :param task_type: 任务类型
        """
        self.task_type = task_type
        self.data_path = f"/home/xingzhuang/workplace/yyh/timepredict/data/{task_type.value}_data.csv"
        self.model_path = f"/home/xingzhuang/workplace/yyh/timepredict/model/{task_type.value}_model.pkl"
        self.load_model_path = f"/home/xingzhuang/workplace/yyh/timepredict/model/{task_type.value}_load_model.pkl"
        self.execution_data = self.load_execution_data()
        
        # 根据任务类型选择特征
        self.features = self._get_features_for_task_type()
        
        # 执行时间预测模型
        self.model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="execution_time",
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
        self.model_load_time = 0

    def _get_features_for_task_type(self):
        """根据任务类型返回相应的特征列表"""
        # common_features = [
        #     "input_size",          # 输入数据大小
        #     "output_size",         # 输出数据大小
        #     "cpu_cores",           # CPU核心数
        #     "memory_usage",        # 内存使用
        #     "gpu_usage",           # GPU使用
        #     "system_load",         # 系统负载
        # ]
        
        task_specific_features = {
            TaskType.OCR: [
                "image_width",         # 图像宽度
                "image_height",        # 图像高度
                "image_entropy",       # 图像熵
                "edge_density",        # 边缘密度
                "text_contrast",       # 文本对比度
                "text_block_count",    # 文本区块数
                "char_count_estimate", # 估计字符数
                "avg_brightness",      # 平均亮度
            ],
            TaskType.SPEECH: [
                "duration",            # 音频时长
                "audio_entropy",       # 音频熵
                "audio_energy",        # 音频能量
                "sample_rate",         # 采样率
                "channels",            # 声道数
            ],
            TaskType.IMAGE: [
                "image_width",         # 图像宽度
                "image_height",        # 图像高度
                "image_entropy",       # 图像熵
                "edge_density",        # 边缘密度
                "avg_brightness",      # 平均亮度
            ],
            TaskType.TEXT: [
                "text_length",         # 文本长度
                "word_count",          # 词数
                "language",            # 语言
                "complexity",          # 复杂度
            ],
            TaskType.AUDIO: [
                "duration",            # 音频时长
                "audio_entropy",       # 音频熵
                "audio_energy",        # 音频能量
                "sample_rate",         # 采样率
            ]
        }
        
        return task_specific_features.get(self.task_type, [])

    def calculate_features(self, input_data):
        """计算任务特征"""
        try:
            features = {}
            
            # 1. 通用特征
            # features["input_size"] = self._get_input_size(input_data)
            # features["cpu_cores"] = os.cpu_count()
            # features["memory_usage"] = self._get_memory_usage()
            # features["gpu_usage"] = self._get_gpu_usage()
            # features["system_load"] = self._get_system_load()
            
            # 2. 任务特定特征
            if self.task_type == TaskType.OCR:
                features.update(self._calculate_ocr_features(input_data))
            elif self.task_type == TaskType.SPEECH:
                features.update(self._calculate_speech_features(input_data))
            elif self.task_type == TaskType.IMAGE:
                features.update(self._calculate_image_features(input_data))
            elif self.task_type == TaskType.TEXT:
                features.update(self._calculate_text_features(input_data))
            elif self.task_type == TaskType.AUDIO:
                features.update(self._calculate_audio_features(input_data))
            
            return features
            
        except Exception as e:
            print(f"计算特征时出错: {str(e)}")
            return None

    def _get_input_size(self, input_data):
        """获取输入数据大小"""
        if isinstance(input_data, str):
            return os.path.getsize(input_data) if os.path.exists(input_data) else len(input_data)
        elif isinstance(input_data, (bytes, bytearray)):
            return len(input_data)
        elif isinstance(input_data, (list, dict)):
            return len(str(input_data))
        return 0

    def _get_memory_usage(self):
        """获取内存使用情况"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0

    def _get_gpu_usage(self):
        """获取GPU使用情况"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return 0
        except:
            return 0

    def _get_system_load(self):
        """获取系统负载"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0

    def _calculate_ocr_features(self, image_path):
        """计算OCR任务特征"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            width, height = image.size
            
            # 转换为灰度图
            if len(img_array.shape) == 3:
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img_array
            
            # 计算图像熵
            image_entropy = self._calculate_entropy(gray_img)
            
            # 计算边缘密度
            edges = cv2.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # 文本检测
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_blocks = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 5 and h > 5:
                        text_blocks.append((x, y, w, h))
            
            # 计算文本对比度
            text_mask = np.zeros_like(gray_img)
            cv2.drawContours(text_mask, contours, -1, 255, -1)
            text_pixels = gray_img[text_mask > 0]
            background_pixels = gray_img[text_mask == 0]
            
            text_contrast = 0
            if len(text_pixels) > 0 and len(background_pixels) > 0:
                text_contrast = abs(np.mean(text_pixels) - np.mean(background_pixels))
            
            return {
                "image_width": width,
                "image_height": height,
                "image_entropy": image_entropy,
                "edge_density": edge_density,
                "text_contrast": text_contrast,
                "text_block_count": len(text_blocks),
                "char_count_estimate": len(contours),
                "avg_brightness": np.mean(gray_img)
            }
        except Exception as e:
            print(f"计算OCR特征时出错: {str(e)}")
            return {}

    def _calculate_speech_features(self, audio_path):
        """计算语音识别任务特征"""
        try:
            # 使用pydub获取基本信息
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000  # 秒
            
            # 使用librosa计算音频特征
            y, sr = librosa.load(audio_path)
            S = np.abs(librosa.stft(y))
            p = S / np.sum(S)
            audio_entropy = -np.sum(p * np.log2(p + 1e-10))
            audio_energy = np.sum(y**2)
            
            return {
                "duration": duration,
                "audio_entropy": audio_entropy,
                "audio_energy": audio_energy,
                "sample_rate": sr,
                "channels": audio.channels
            }
        except Exception as e:
            print(f"计算语音特征时出错: {str(e)}")
            return {}

    def _calculate_image_features(self, image_path):
        """计算图像处理任务特征"""
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            width, height = image.size
            
            if len(img_array.shape) == 3:
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img_array
            
            image_entropy = self._calculate_entropy(gray_img)
            edges = cv2.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0) / (width * height)
            
            return {
                "image_width": width,
                "image_height": height,
                "image_entropy": image_entropy,
                "edge_density": edge_density,
                "avg_brightness": np.mean(gray_img)
            }
        except Exception as e:
            print(f"计算图像特征时出错: {str(e)}")
            return {}

    def _calculate_text_features(self, text):
        """计算文本处理任务特征"""
        try:
            words = text.split()
            return {
                "text_length": len(text),
                "word_count": len(words),
                "language": self._detect_language(text),
                "complexity": self._calculate_text_complexity(text)
            }
        except Exception as e:
            print(f"计算文本特征时出错: {str(e)}")
            return {}

    def _calculate_audio_features(self, audio_path):
        """计算音频处理任务特征"""
        try:
            audio = AudioSegment.from_file(audio_path)
            y, sr = librosa.load(audio_path)
            S = np.abs(librosa.stft(y))
            p = S / np.sum(S)
            audio_entropy = -np.sum(p * np.log2(p + 1e-10))
            audio_energy = np.sum(y**2)
            
            return {
                "duration": len(audio) / 1000,
                "audio_entropy": audio_entropy,
                "audio_energy": audio_energy,
                "sample_rate": sr
            }
        except Exception as e:
            print(f"计算音频特征时出错: {str(e)}")
            return {}

    def _calculate_entropy(self, data):
        """计算熵"""
        histogram = np.histogram(data, bins=256, range=(0, 256))[0]
        histogram = histogram / np.sum(histogram)
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))

    def _detect_language(self, text):
        """检测文本语言（简化版）"""
        # 这里可以实现更复杂的语言检测
        return "unknown"

    def _calculate_text_complexity(self, text):
        """计算文本复杂度（简化版）"""
        # 这里可以实现更复杂的文本复杂度计算
        return len(text) * 0.1

    def save_execution_data(self):
        """保存执行数据"""
        df = pd.DataFrame(self.execution_data)
        df.to_csv(self.data_path, index=False)

    def load_execution_data(self):
        """加载执行数据"""
        if os.path.exists(self.data_path):
            return pd.read_csv(self.data_path).to_dict("records")
        return []

    def train_model(self):
        """训练预测模型"""
        df = pd.DataFrame(self.execution_data)
        if len(df) < 5:
            print("数据量不足，无法训练模型")
            return False
            
        # 训练执行时间预测模型
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

    def predict_execution_time(self, input_data):
        """预测执行时间"""
        if not self.model_trained:
            print("模型未训练，请先训练模型")
            return None
            
        # 计算特征
        features = self.calculate_features(input_data)
        if features is None:
            return None
            
        # 按照特征列表的顺序创建特征数组
        feature_values = [features[feature] for feature in self.features]
        return round(self.model_trainer.predict(feature_values), 2)

    def predict_load_time(self, input_data):
        """预测加载时间"""
        if not self.load_model_trained:
            print("加载时间模型未训练，请先训练模型")
            return None
            
        # 计算特征
        features = self.calculate_features(input_data)
        if features is None:
            return None
            
        # 按照特征列表的顺序创建特征数组
        feature_values = [features[feature] for feature in self.features]
        return round(self.load_model_trainer.predict(feature_values), 2)

    def predict_total_time(self, input_data):
        """预测总时间（加载+执行）"""
        exec_time = self.predict_execution_time(input_data)
        if exec_time is None:
            return None
            
        if self.load_model_trained:
            load_time = self.predict_load_time(input_data)
            return round(exec_time + load_time, 2)
        else:
            return round(exec_time + self.model_load_time, 2)

def main():
    # 示例：OCR任务预测
    ocr_predictor = TaskPredictor(TaskType.OCR)
    image_path = "/path/to/image.jpg"
    if os.path.exists(image_path):
        predicted_time = ocr_predictor.predict_total_time(image_path)
        print(f"OCR任务预测总时间: {predicted_time} 秒")
    
    # 示例：语音识别任务预测
    speech_predictor = TaskPredictor(TaskType.SPEECH)
    audio_path = "/path/to/audio.wav"
    if os.path.exists(audio_path):
        predicted_time = speech_predictor.predict_total_time(audio_path)
        print(f"语音识别任务预测总时间: {predicted_time} 秒")

if __name__ == "__main__":
    main() 