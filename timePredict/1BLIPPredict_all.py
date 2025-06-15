import pandas as pd
import time
import torch
from PIL import Image
from io import BytesIO
import warnings
import numpy as np
from transformers import BlipProcessor, BlipForQuestionAnswering
from ModelTrainer1 import ModelTrainer
import os
warnings.filterwarnings("ignore")

class BLIPPredictionTool:
    def __init__(self):
        # 指定使用 GPU 1
        if torch.cuda.is_available():
            torch.cuda.set_device(1)  # 使用 GPU 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载模型和处理器
        print("Loading model and processor...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.model.to(self.device)
        print("Model loaded.")

        # 数据和模型的路径
        self.data_path = "data/800blip_data.csv"
        self.model_path = "/home/xingzhuang/workplace/yyh/Data/timePredict/model/blip_final.pkl"  # 修改为合并后的模型路径

        # 确保模型目录存在
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # 加载执行数据
        self.execution_data = self.load_execution_data()

        # 简化特征 - 专注于最重要的特征
        self.features = [
            "image_height",      # 图像高度 
            "image_width",       # 图像宽度
            "image_entropy",     # 图像复杂度
            "question_length"    # 问题长度
        ]
        
        # 执行时间预测模型（合并后的总时间模型）
        self.model_trainer = ModelTrainer(
            feature_columns=self.features,
            target_column="total_time",  # 改为预测总执行时间
            model_path=self.model_path
        )
        
        self.model_trained = False

    def calculate_image_entropy(self, image):
        """计算图像熵作为复杂度指标"""
        hist = np.histogram(image, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def image_to_answer(self, image_path: str, question: str):
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
            
            # 固定的加载时间
            load_time = 8.0
                
            # 处理图像和问题
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            # 生成答案 - 测量处理时间
            process_start_time = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(**inputs)
            execution_time = time.perf_counter() - process_start_time
            
            # 计算总执行时间
            total_time = load_time + execution_time
            
            # 解码答案
            answer = self.processor.decode(out[0], skip_special_tokens=True)
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Total execution time: {total_time:.2f} seconds")

            # 记录执行数据
            self.execution_data.append({
                **features,
                "load_time": load_time,
                "execution_time": execution_time,
                "total_time": total_time
            })
            self.save_execution_data()
            
            return answer, total_time
        except Exception as e:
            print(f"[Error] {image_path}: {str(e)}")
            return None, 0

    def save_execution_data(self):
        """将执行数据保存到CSV文件"""
        df = pd.DataFrame(self.execution_data)
        df.to_csv(self.data_path, index=False)
        
    def load_execution_data(self):
        """从CSV文件加载执行数据"""
        if os.path.exists(self.data_path):
            return pd.read_csv(self.data_path).to_dict("records")
        return []
        
    def train_model(self):
        """训练执行时间预测模型"""
        df = pd.DataFrame(self.execution_data)
        if len(df) < 5:
            print("Insufficient data for model training (need at least 5 samples)")
            return False
            
        # 如果没有load_time列，添加固定的8秒加载时间
        if 'load_time' not in df.columns:
            df['load_time'] = 8.0
            
        # 计算总时间 (execution_time + load_time)
        df['total_time'] = df['execution_time'] + df['load_time']
            
        # 训练合并后的总时间预测模型
        print("Training total time prediction model...")
        success = self.model_trainer.train_model(df)
        self.model_trained = success
        print(f"Model training {'successful' if success else 'failed'}")
        return success

    def predict_total_time(self, image_height, image_width, image_entropy, question_length):
        """预测总执行时间（process_time + load_time）"""
        if not self.model_trained:
            print("Model not trained. Train model first.")
            return None
            
        # 创建特征数组
        features = np.array([[
            image_height, 
            image_width, 
            image_entropy,
            question_length
        ]])
            
        # 预测总执行时间
        return round(self.model_trainer.predict(features), 2)


def main():
    tool = BLIPPredictionTool()

    # 训练模型
    print("Training model...")
    if tool.train_model():
        print("Model training successful")
    else:
        print("Model training failed")
        return

    # 执行预测
    print("Predicting total time (process_time + load_time)...")
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
            
        # 预测总执行时间
        predicted_time = tool.predict_total_time(
            image_height=image_height,
            image_width=image_width,
            image_entropy=image_entropy,
            question_length=question_length
        )
        
        if predicted_time is not None:
            print(f"Predicted Total Time (process_time + load_time): {predicted_time} seconds")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    main() 