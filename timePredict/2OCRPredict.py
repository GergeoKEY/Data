import pandas as pd
import time
from pydub import AudioSegment
import easyocr
from ModelTrainer1 import ModelTrainer
import os
import numpy as np
import torch
from PIL import Image
import cv2


class OCRPredictionTool:
    def __init__(self, languages=['en', 'ch_sim'], measure_load_time=True):
        # 开始测量加载时间
        load_start_time = time.perf_counter() if measure_load_time else None
        
        print(f"加载 EasyOCR 模型（languages: {languages}）中...")
        self.model = easyocr.Reader(languages)
        
        # 计算加载时间
        self.model_load_time = time.perf_counter() - load_start_time if measure_load_time else 0
        print(f"模型加载完成。加载时间: {self.model_load_time:.2f} 秒")
        
        self.data_path = "data/1ocr_data.csv"
        self.model_path = "model/1ocr_model.pkl"
        self.load_model_path = "model/1ocr_load_model.pkl"
        self.execution_data = self.load_execution_data()
        self.languages = languages
        
        # 扩展特征列表，包含更多有用的特征
        self.features = [
            # 基本几何特征
            "image_width",         # 图像宽度
            "image_height",        # 图像高度
            # "image_area",          # 图像面积（宽x高）
            # "image_aspect_ratio",  # 宽高比
            # "image_dpi",           # 图像DPI
            
            # 图像复杂度特征
            "image_entropy",       # 图像熵（复杂度）
            "edge_density",        # 边缘密度
            
            # 文本相关特征
            # "text_area_ratio",     # 文本区域占比
            "text_contrast",       # 文本与背景对比度
            "text_block_count",    # 文本区块数量
            "char_count_estimate", # 估计的字符数量
            
            # 图像质量特征
            # "blur_measure",        # 模糊度测量
            "avg_brightness",      # 平均亮度
            # "brightness_variance", # 亮度方差
        ]
        
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

    def calculate_image_features(self, image_path):
        """计算图像特征"""
        try:
            # 读取图像
            image = Image.open(image_path)
            img_array = np.array(image)
            
            # 基本图像信息
            width, height = image.size
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # 获取DPI信息
            try:
                dpi = image.info.get('dpi', (72, 72))[0]  # 默认72dpi
            except:
                dpi = 72
            
            # 准备灰度图像用于后续分析
            if len(img_array.shape) == 3:
                # 转换为灰度图
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img_array
            
            # 1. 图像复杂度特征
            # 计算图像熵
            image_entropy = self._calculate_entropy(gray_img)
            
            # 计算边缘密度
            edges = cv2.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0) / area
            
            # 2. 文本相关特征提取
            # 使用OTSU自适应阈值进行二值化
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 形态学操作以改善文本区域
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 寻找潜在的文本区域
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 统计和处理文本区域
            text_area = 0
            text_blocks = []
            valid_contours = []
            
            min_contour_area = 5  # 最小区域阈值
            for cnt in contours:
                area_cnt = cv2.contourArea(cnt)
                if area_cnt > min_contour_area:
                    valid_contours.append(cnt)
                    text_area += area_cnt
                    # 创建文本区块的边界矩形
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 5 and h > 5:  # 过滤太小的矩形
                        text_blocks.append((x, y, w, h))
            
            # 估计字符数量
            char_count_estimate = len(valid_contours)
            
            # 文本区域占比
            text_area_ratio = text_area / area if area > 0 else 0
            
            # 文本区块数量（合并很接近的区块）
            text_block_count = len(text_blocks)
            
            # 计算文本与背景的对比度
            # 使用文本区域内外像素值的标准差作为对比度度量
            text_mask = np.zeros_like(gray_img)
            cv2.drawContours(text_mask, valid_contours, -1, 255, -1)
            text_pixels = gray_img[text_mask > 0]
            background_pixels = gray_img[text_mask == 0]
            
            text_contrast = 0
            if len(text_pixels) > 0 and len(background_pixels) > 0:
                text_contrast = abs(np.mean(text_pixels) - np.mean(background_pixels))
            
            # 3. 图像质量特征
            # 平均亮度
            avg_brightness = np.mean(gray_img)
            
            # 亮度方差(作为一种质量指标)
            brightness_variance = np.var(gray_img)
            
            # 模糊度度量 - 使用拉普拉斯算子
            laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
            blur_measure = 1 / (laplacian_var + 1e-10)  # 避免除零
            
            return {
                # 基本几何特征
                "image_width": width,
                "image_height": height,
                "image_area": area,
                "image_aspect_ratio": aspect_ratio,
                "image_dpi": dpi,
                
                # 图像复杂度特征
                "image_entropy": image_entropy,
                "edge_density": edge_density,
                
                # 文本相关特征
                "text_area_ratio": text_area_ratio,
                "text_contrast": text_contrast,
                "text_block_count": text_block_count,
                "char_count_estimate": char_count_estimate,
                
                # 图像质量特征
                "blur_measure": blur_measure,
                "avg_brightness": avg_brightness,
                "brightness_variance": brightness_variance,
            }
        except Exception as e:
            print(f"计算图像特征时出错: {str(e)}")
            return None

    def _calculate_entropy(self, img_array):
        """计算图像熵（信息量）"""
        histogram = np.histogram(img_array, bins=256, range=(0, 256))[0]
        histogram = histogram / np.sum(histogram)
        # 过滤掉零概率值（避免log(0)）
        histogram = histogram[histogram > 0]
        return -np.sum(histogram * np.log2(histogram))

    def get_system_features(self):
        """获取系统特征"""
        # 由于不再使用系统特征，返回空字典
        return {}
        
    def measure_model_load_time(self, repeat=3):
        """测量模型加载时间"""
        load_times = []
        for _ in range(repeat):
            # 清除之前的模型
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 测量加载时间
            start_time = time.perf_counter()
            self.model = easyocr.Reader(self.languages)
            load_time = time.perf_counter() - start_time
            load_times.append(load_time)
            
            print(f"模型加载时间: {load_time:.2f} 秒")
            
        # 返回平均加载时间
        avg_load_time = sum(load_times) / len(load_times)
        print(f"平均模型加载时间: {avg_load_time:.2f} 秒")
        return avg_load_time

    def ocr_process(self, image_path: str, measure_load=False):
        try:
            print(f"Processing: {image_path}")
            
            # 计算图像特征
            image_features = self.calculate_image_features(image_path)
            if image_features is None:
                return None
                
            # 测量加载时间（如果需要）
            load_time = 0
            if measure_load:
                load_time = self.measure_model_load_time()
            
            # 开始计时
            start_time = time.perf_counter()
            
            # OCR处理
            result = self.model.readtext(image_path)
            
            execution_time = time.perf_counter() - start_time
            text_res = ' '.join([text[1] for text in result])
            print(f"识别内容：{text_res}")
            print(f"执行时间: {execution_time:.2f} 秒")
            
            # 记录特征和执行时间
            data_entry = {
                **image_features,  # 图像特征
                "execution_time": execution_time,  # 执行时间
                "load_time": load_time  # 加载时间
            }
            self.execution_data.append(data_entry)
            
            self.save_execution_data()
            return text_res
            
        except Exception as e:
            print(f"[Error] {image_path}: {str(e)}")
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

    def predict_execution_time(self, width: int, height: int, dpi: int = 72, 
                              brightness: float = 128.0, char_count_estimate: int = 100,
                              **kwargs) -> float:
        """预测仅OCR执行时间（不包括模型加载）"""
        if not self.model_trained:
            print("模型未训练，请先训练模型")
            return None
            
        # 创建特征字典，包含所有必要特征
        features = {
            "image_width": width,
            "image_height": height,
            "image_area": width * height,
            "image_aspect_ratio": width / height if height > 0 else 0,
            "image_dpi": dpi,
            "avg_brightness": brightness,
            "char_count_estimate": char_count_estimate,
            
            # 设置默认值，如果没有提供
            "image_entropy": kwargs.get("image_entropy", 5.0),
            "edge_density": kwargs.get("edge_density", 0.1),
            "text_area_ratio": kwargs.get("text_area_ratio", 0.2),
            "text_contrast": kwargs.get("text_contrast", 50.0),
            "text_block_count": kwargs.get("text_block_count", 10),
            "blur_measure": kwargs.get("blur_measure", 0.5),
            "brightness_variance": kwargs.get("brightness_variance", 2000.0),
        }
        
        # 预测执行时间
        # 按照特征列表的顺序创建特征数组
        feature_values = [features[feature] for feature in self.features]
        return round(self.model_trainer.predict(feature_values), 2)
        
    def predict_load_time(self, width: int, height: int, dpi: int = 72, 
                         brightness: float = 128.0, char_count_estimate: int = 100,
                         **kwargs) -> float:
        """预测模型加载时间"""
        if not self.load_model_trained:
            print("加载时间模型未训练，请先训练模型")
            return None
            
        # 创建特征字典，包含所有必要特征
        features = {
            "image_width": width,
            "image_height": height,
            "image_area": width * height,
            "image_aspect_ratio": width / height if height > 0 else 0,
            "image_dpi": dpi,
            "avg_brightness": brightness,
            "char_count_estimate": char_count_estimate,
            
            # 设置默认值，如果没有提供
            "image_entropy": kwargs.get("image_entropy", 5.0),
            "edge_density": kwargs.get("edge_density", 0.1),
            "text_area_ratio": kwargs.get("text_area_ratio", 0.2),
            "text_contrast": kwargs.get("text_contrast", 50.0),
            "text_block_count": kwargs.get("text_block_count", 10),
            "blur_measure": kwargs.get("blur_measure", 0.5),
            "brightness_variance": kwargs.get("brightness_variance", 2000.0),
        }
        
        # 预测加载时间
        # 按照特征列表的顺序创建特征数组
        feature_values = [features[feature] for feature in self.features]
        return round(self.load_model_trainer.predict(feature_values), 2)
        
    def predict_total_time(self, width: int, height: int, dpi: int = 72, 
                          brightness: float = 128.0, char_count_estimate: int = 100,
                          **kwargs) -> float:
        """预测总时间（模型加载+执行）"""
        exec_time = self.predict_execution_time(width, height, dpi, brightness, 
                                              char_count_estimate, **kwargs)
        
        # 如果加载时间模型已训练，则加上预测的加载时间
        if self.load_model_trained:
            load_time = self.predict_load_time(width, height, dpi, brightness, 
                                             char_count_estimate, **kwargs)
            return round(exec_time + load_time, 2)
        # 否则使用实际测量的加载时间
        else:
            return round(exec_time + self.model_load_time, 2)


def main():
    # 创建工具时测量加载时间
    tool = OCRPredictionTool(measure_load_time=True)
    possible_extensions = ['.png', '.jpg', '.jpeg']
    # # 处理与BLIP相同的图像数据集
    # print("Processing images...")
    # for i in range(68):
    #     # 检查不同扩展名
    #     possible_extensions = ['.png', '.jpg', '.jpeg']
    #     image_path = None
        
    #     for ext in possible_extensions:
    #         temp_path = f"picture/gaia_{i}{ext}"
    #         if os.path.exists(temp_path):
    #             image_path = temp_path
    #             break
                
    #     if image_path and os.path.exists(image_path):
    #         # 每次都重新加载模型
    #         measure_load = True
    #         result = tool.ocr_process(image_path, measure_load=measure_load)
    #         print(f"Text: {result}")

    # # 训练模型
    print("Training model...")
    if tool.train_model():
        print("模型训练成功")
    else:
        print("模型训练失败")
        return

    # 执行预测
    print("Predicting execution time...")
    try:
        # 使用指定的图片
        example_img = "picture/gaia_67.jpg"
        
        if not os.path.exists(example_img):
            print(f"找不到示例图片: {example_img}")
            return
            
        # 使用找到的图片作为示例
        features = tool.calculate_image_features(example_img)
        if features:
            print("图像特征：")
            for key, value in features.items():
                print(f"  {key}: {value}")
                
            # 预测执行时间（不含加载）
            predicted_exec_time = tool.predict_execution_time(
                width=features["image_width"],
                height=features["image_height"],
                dpi=features["image_dpi"],
                brightness=features["avg_brightness"],
                char_count_estimate=features["char_count_estimate"],
                image_entropy=features["image_entropy"],
                edge_density=features["edge_density"],
                text_area_ratio=features["text_area_ratio"],
                text_contrast=features["text_contrast"],
                text_block_count=features["text_block_count"],
                blur_measure=features["blur_measure"],
                brightness_variance=features["brightness_variance"]
            )
            
            # 预测总时间（含加载）
            predicted_total_time = tool.predict_total_time(
                width=features["image_width"],
                height=features["image_height"],
                dpi=features["image_dpi"],
                brightness=features["avg_brightness"],
                char_count_estimate=features["char_count_estimate"],
                image_entropy=features["image_entropy"],
                edge_density=features["edge_density"],
                text_area_ratio=features["text_area_ratio"],
                text_contrast=features["text_contrast"],
                text_block_count=features["text_block_count"],
                blur_measure=features["blur_measure"],
                brightness_variance=features["brightness_variance"]
            )
            
            if predicted_exec_time is not None:
                print(f"Predicted Execution Time (without loading): {predicted_exec_time} seconds")
            if predicted_total_time is not None:
                print(f"Predicted Total Time (with loading): {predicted_total_time} seconds")
        else:
            print("无法计算图像特征")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    main() 