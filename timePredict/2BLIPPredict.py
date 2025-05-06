import pandas as pd
import time
import torch
from PIL import Image
from io import BytesIO
import warnings
import numpy as np
from transformers import BlipProcessor, BlipForQuestionAnswering
from ModelTrainer2 import ModelTrainer
import os
warnings.filterwarnings("ignore")

class BLIPPredictionTool:
    def __init__(self, measure_load_time=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载模型和处理器
        load_start_time = time.perf_counter() if measure_load_time else None
        
        print("Loading model and processor...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.model.to(self.device)
        
        # 计算加载时间
        self.model_load_time = time.perf_counter() - load_start_time if measure_load_time else 0
        print(f"Model loaded. Load time: {self.model_load_time:.2f} seconds")

        # 数据和模型的路径
        self.execution_data_path = "data/800blip_data.csv"
        self.process_model_path = "model/blip_process_model.pkl"
        self.load_model_path = "model/blip_load_model.pkl"

        # 加载执行数据
        self.execution_data = self.load_execution_data()

        # 简化特征 - 专注于最重要的特征
        self.features = [
            "image_height",      # 图像高度 
            "image_width",       # 图像宽度
            "image_entropy",     # 图像复杂度
            "question_length"    # 问题长度
        ]
        
        # 处理时间预测模型
        self.process_model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="process_time",
            model_path=self.process_model_path
        )
        
        # 加载时间预测模型
        self.load_model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="load_time",
            model_path=self.load_model_path
        )
        
        self.process_model_trained = False
        self.load_model_trained = False

    def calculate_image_entropy(self, image):
        """计算图像熵作为复杂度指标"""
        hist = np.histogram(image, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def measure_model_load_time(self, repeat=3):
        """测量模型加载时间，多次测量取平均值"""
        load_times = []
        for _ in range(repeat):
            # 清除之前的模型
            del self.model
            del self.processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 测量加载时间
            start_time = time.perf_counter()
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            self.model.to(self.device)
            load_time = time.perf_counter() - start_time
            load_times.append(load_time)
            
            print(f"Model load time: {load_time:.2f} seconds")
            
        # 返回平均加载时间
        avg_load_time = sum(load_times) / len(load_times)
        print(f"Average model load time: {avg_load_time:.2f} seconds")
        return avg_load_time

    def image_to_answer(self, image_path: str, question: str, measure_load=False):
        """处理图像并返回带有时间测量的答案的主要方法"""
        try:
            print(f"Processing: {image_path}")
            
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # 计算特征
            features = {
                "image_height": image.height,
                "image_width": image.width,
                "image_entropy": self.calculate_image_entropy(image_np),
                "question_length": len(question)
            }
            
            # 测量加载时间（如需要）
            load_time = 0
            if measure_load:
                load_time = self.measure_model_load_time()
                
            # 处理图像和问题
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            # 生成答案 - 测量处理时间
            process_start_time = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(**inputs)
            process_time = time.perf_counter() - process_start_time
            
            # 计算总执行时间
            execution_time = load_time + process_time
            
            # 解码答案
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Process time: {process_time:.2f} seconds")
            print(f"Total execution time: {execution_time:.2f} seconds")

            # 记录执行数据
            self.execution_data.append({
                **features,
                "load_time": load_time,
                "process_time": process_time,
                "execution_time": execution_time
            })
            self.save_execution_data()
            
            return answer, process_time, execution_time
        except Exception as e:
            print(f"[Error] {image_path}: {str(e)}")
            return None, 0, 0

    def save_execution_data(self):
        """将执行数据保存到CSV文件"""
        df = pd.DataFrame(self.execution_data)
        df.to_csv(self.execution_data_path, index=False)
        
    def load_execution_data(self):
        """从CSV文件加载执行数据"""
        if os.path.exists(self.execution_data_path):
            return pd.read_csv(self.execution_data_path).to_dict("records")
        return []
        
    def train_model(self):
        """训练处理和加载时间预测模型"""
        df = pd.DataFrame(self.execution_data)
        if len(df) < 5:
            print("Insufficient data for model training (need at least 5 samples)")
            return False
            
        # 训练处理时间预测模型
        print("Training processing time prediction model...")
        process_success = self.process_model_trainer.train_model(df)
        self.process_model_trained = process_success
        print(f"Processing time model training {'successful' if process_success else 'failed'}")
        
        # 检查是否有足够的加载时间数据
        if 'load_time' in df.columns and df['load_time'].sum() > 0:
            # 训练加载时间预测模型
            print("Training load time prediction model...")
            load_success = self.load_model_trainer.train_model(df)
            self.load_model_trained = load_success
            print(f"Load time model training {'successful' if load_success else 'failed'}")
        else:
            print("Not enough load time data, skipping load time model training")
            
        return process_success

    def predict_process_time(self, image_height, image_width, image_entropy, question_length):
        """预测处理时间（不包括模型加载）"""
        if not self.process_model_trained:
            print("Processing time model not trained. Train model first.")
            return None
            
        # 创建特征数组
        features = np.array([[
            image_height, 
            image_width, 
            image_entropy,
            question_length
        ]])
            
        # 预测处理时间
        return round(self.process_model_trainer.predict(features), 2)
    
    def predict_load_time(self, image_height, image_width, image_entropy, question_length):
        """预测模型加载时间"""
        if not self.load_model_trained:
            print("Load time model not trained. Train model first.")
            return None
            
        # 创建特征数组
        features = np.array([[
            image_height, 
            image_width, 
            image_entropy,
            question_length
        ]])
            
        # 预测加载时间
        return round(self.load_model_trainer.predict(features), 2)
        
    def predict_total_time(self, image_height, image_width, image_entropy, question_length):
        """预测总时间（模型加载+处理）"""
        process_time = self.predict_process_time(
            image_height, image_width, image_entropy, question_length
        )
        
        if process_time is None:
            return None
            
        # 如果加载时间模型已训练，添加预测的加载时间
        if self.load_model_trained:
            load_time = self.predict_load_time(
                image_height, image_width, image_entropy, question_length
            )
            return round(process_time + load_time, 2)
        # 否则使用实际测量的加载时间
        else:
            return round(process_time + self.model_load_time, 2)

def main():
    tool = BLIPPredictionTool(measure_load_time=True)

    # 测试问题列表
    questions = [
        "What is in the image?",
        "How many objects are there?",
        "What color is the main object?",
        "Is there a person in the image?",
        "What is the background?",
        "Are there any animals?",
        "What is the main focus?",
        "Is it day or night?",
        "What is the setting?",
        "Are there any vehicles?"
    ]

    # 处理图像数据集
    print("Processing images...")
    for i in range(50):  # 限制为50张图像
        image_path = f"/home/xingzhuang/workplace/yyh/GAIA_TOOL/picture/gaia_{i}.jpg"
        if os.path.exists(image_path):
            # 选择一个问题
            question = questions[i % len(questions)]
            # 每5张图像测量一次加载时间
            measure_load = (i % 5 == 0)  # 每5张图像测量一次加载时间
            result, process_time, total_time = tool.image_to_answer(image_path, question, measure_load=measure_load)
            print(f"Result: {result}")
            print(f"Processing time: {process_time:.2f}s, Total time: {total_time:.2f}s")

    # 训练模型
    print("Training models...")
    if tool.train_model():
        print("Model training successful")
    else:
        print("Model training failed")
        return

    # 执行预测
    print("Predicting execution times...")
    try:
        # 使用示例图像
        example_image_path = "/home/xingzhuang/workplace/yyh/GAIA_TOOL/picture/gaia_0.jpg"
        if not os.path.exists(example_image_path):
            print("Example image not found, using default parameters")
            # 使用默认参数
            image_height = 512
            image_width = 512
            image_entropy = 7.5
            question_length = 20
        else:
            # 计算实际特征
            example_image = Image.open(example_image_path).convert('RGB')
            image_np = np.array(example_image)
            image_height = example_image.height
            image_width = example_image.width
            image_entropy = tool.calculate_image_entropy(image_np)
            question_length = len("What is in the image?")
            
        # 预测处理时间（不包含加载）
        predicted_process_time = tool.predict_process_time(
            image_height=image_height,
            image_width=image_width,
            image_entropy=image_entropy,
            question_length=question_length
        )
        
        # 预测总时间（包含加载）
        predicted_total_time = tool.predict_total_time(
            image_height=image_height,
            image_width=image_width,
            image_entropy=image_entropy,
            question_length=question_length
        )
        
        if predicted_process_time is not None:
            print(f"Predicted Process Time (without loading): {predicted_process_time} seconds")
        if predicted_total_time is not None:
            print(f"Predicted Total Time (with loading): {predicted_total_time} seconds")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main() 