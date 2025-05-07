import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

class ModelTrainer:
    def __init__(self, feature_columns, target_column, model_path):
        """
        使用XGBoost的增量学习模型训练工具
        :param feature_columns: list，特征列名
        :param target_column: str，目标列名
        :param model_path: str，保存模型的路径
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model_path = model_path
        self.best_model = None
        self.scaler = None
        self.best_mae = float('inf')
        self.best_r2 = 0
        
        # 创建模型目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 尝试加载已有模型
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.best_model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"已加载现有模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {str(e)}")

    def train_model(self, data, incremental=False):
        """
        训练模型
        :param data: pandas.DataFrame, 包含特征和目标变量的数据
        :param incremental: bool, 是否增量训练
        :return: 是否训练成功
        """
        if len(data) < 5:
            print("数据量不足，无法训练模型")
            return False

        # 准备数据
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # 如果是增量训练且已有模型，直接使用新数据
        if incremental and self.best_model is not None:
            print("使用新数据进行增量训练...")
            X_new = X
            y_new = y
        else:
            # 首次训练，划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_new = X_train
            y_new = y_train
            self.val_data = (X_val, y_val)  # 保存验证集用于评估

        # 标准化数据
        if self.scaler is None:
            print("创建新的标准化器...")
            self.scaler = StandardScaler()
            X_new_scaled = self.scaler.fit_transform(X_new)
        else:
            print("使用现有标准化器...")
            X_new_scaled = self.scaler.transform(X_new)

        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(X_new_scaled, label=y_new, feature_names=self.feature_columns)
        
        # 准备验证集
        if hasattr(self, 'val_data'):
            X_val, y_val = self.val_data
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_columns)
            eval_list = [(dval, "eval")]
        else:
            eval_list = []

        # XGBoost参数
        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cuda",
        }

        # 训练模型
        if incremental and self.best_model is not None:
            print("增量训练现有模型...")
            self.best_model = xgb.train(
                params, dtrain,
                num_boost_round=50,
                evals=eval_list,
                early_stopping_rounds=10,
                xgb_model=self.best_model
            )
        else:
            print("开始新模型训练...")
            self.best_model = xgb.train(
                params, dtrain,
                num_boost_round=100,
                evals=eval_list,
                early_stopping_rounds=20
            )

        # 评估模型
        if hasattr(self, 'val_data'):
            X_val, y_val = self.val_data
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, feature_names=self.feature_columns)
            y_pred_val = self.best_model.predict(dval)
            current_mae, current_r2 = self._print_evaluation_metrics(y_val, y_pred_val, "验证集")
            
            # 更新最佳模型
            if current_mae < self.best_mae:
                self.best_mae = current_mae
                self.best_r2 = current_r2
                self._save_model()

        print("\n📊 训练完成！")
        if hasattr(self, 'val_data'):
            print(f"最佳验证集 MAE: {self.best_mae:.4f}")
            print(f"最佳验证集 R²: {self.best_r2:.4f}")
        return True

    def _print_evaluation_metrics(self, y_true, y_pred, stage_name=""):
        """打印评估指标"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{stage_name}评估结果:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        return mae, r2

    def _save_model(self):
        """保存模型和标准化器"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, self.model_path)

    def predict(self, X_new):
        """使用最佳模型进行预测"""
        if self.best_model is None:
            if not os.path.exists(self.model_path):
                raise ValueError("模型未训练或加载失败")
            model_data = joblib.load(self.model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
        
        # 检查输入类型并相应处理
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new[self.feature_columns]
        elif isinstance(X_new, np.ndarray):
            if X_new.ndim == 1:
                X_new = pd.DataFrame([X_new], columns=self.feature_columns)
            else:
                X_new = pd.DataFrame(X_new, columns=self.feature_columns)
        else:
            X_new = pd.DataFrame([X_new], columns=self.feature_columns)
            
        # 应用标准化
        X_new_scaled = self.scaler.transform(X_new)
        dtest = xgb.DMatrix(X_new_scaled, feature_names=self.feature_columns)
        return self.best_model.predict(dtest)[0]
