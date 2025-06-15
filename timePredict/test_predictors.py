import sys
import os
import redis
import json
from tp import task2_blip_process_predictor, task3_ocr_process_predictor

def test_blip_predictor():
    print("\n=== Testing BLIP Predictor ===")
    # 初始化预测器
    predictor = task2_blip_process_predictor(redis_ip="localhost", redis_port=6379)
    
    # 测试数据
    test_features = {
        "image_height": 512,
        "image_width": 512,
        "image_entropy": 7.5,
        "question_length": 20
    }
    
    # 将测试数据存入Redis
    redis_client = redis.Redis(host="localhost", port=6379)
    test_task_id = "test_blip_task"
    redis_client.set(test_task_id, str(test_features))
    
    # 测试预测
    try:
        predicted_time = predictor.predict(test_task_id)
        print(f"Predicted BLIP execution time: {predicted_time:.2f}s")
    except Exception as e:
        print(f"BLIP prediction error: {str(e)}")
    
    # 清理测试数据
    redis_client.delete(test_task_id)

def test_ocr_predictor():
    print("\n=== Testing OCR Predictor ===")
    # 初始化预测器
    predictor = task3_ocr_process_predictor(redis_ip="localhost", redis_port=6379)
    
    # 测试数据
    test_features = {
        "image_width": 512,
        "image_height": 512,
        "image_entropy": 7.5,
        "edge_density": 0.1,
        "text_contrast": 50.0,
        "text_block_count": 10,
        "char_count_estimate": 100,
        "avg_brightness": 128.0
    }
    
    # 将测试数据存入Redis
    redis_client = redis.Redis(host="localhost", port=6379)
    test_task_id = "test_ocr_task"
    redis_client.set(test_task_id, str(test_features))
    
    # 测试预测
    try:
        predicted_time = predictor.predict(test_task_id)
        print(f"Predicted OCR execution time: {predicted_time:.2f}s")
    except Exception as e:
        print(f"OCR prediction error: {str(e)}")
    
    # 清理测试数据
    redis_client.delete(test_task_id)

def main():
    # 确保Redis服务器正在运行
    try:
        redis_client = redis.Redis(host="localhost", port=6379)
        redis_client.ping()
    except redis.ConnectionError:
        print("Error: Could not connect to Redis server. Please make sure Redis is running.")
        return
    
    # 运行测试
    test_blip_predictor()
    test_ocr_predictor()

if __name__ == "__main__":
    main() 