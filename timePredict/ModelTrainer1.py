import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
# import xgboost as xgb
# import lightgbm as lgb

class ModelTrainer:
    def __init__(self, feature_columns, target_column, model_path):
        """
        通用模型训练工具
        :param feature_columns: list，特征列名
        :param target_column: str，目标列名
        :param model_path: str，保存模型的路径
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.model_path = model_path
        self.best_model = None
        self.scaler = None  # 用于保存归一化器

    def train_model(self, data, incremental=False):
        """
        训练模型
        :param data: pandas.DataFrame, 包含特征和目标变量的数据
        :param incremental: bool, 是否增量训练
        :return: 是否训练成功
        """
        if len(data) < 5:
            print("Not enough data to train the model.")
            return False

        X = data[self.feature_columns]
        y = data[self.target_column]

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 创建归一化器
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        models = {
            "DecisionTree": DecisionTreeRegressor(max_depth=5),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),
            "KNeighbors": KNeighborsRegressor(n_neighbors=5),
            "Bagging": BaggingRegressor(n_estimators=50, random_state=42),
        }

        self.best_model = self._train_and_evaluate(models, X_train_scaled, y_train, X_val_scaled, y_val)

        if self.best_model:
            # 保存模型和归一化器
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, self.model_path)
            return True
        return False

    def _train_and_evaluate(self, models, X_train, y_train, X_val, y_val):
        """训练并评估多个模型，返回最优模型"""
        best_model = None
        best_mae = float("inf")

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            print(f"{name} 模型验证 MAE: {mae:.4f} s, R2: {r2:.4f}")

            if mae < best_mae:
                best_mae = mae
                best_model = model
        return best_model

    def predict(self, X_new):
        """使用最佳模型进行预测"""
        if self.best_model is None:
            model_data = joblib.load(self.model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
        
        # 对输入特征进行归一化
        # 检查输入类型并相应处理
        if isinstance(X_new, pd.DataFrame):
            # 如果已经是DataFrame，确保列顺序正确
            X_new = X_new[self.feature_columns]
        elif isinstance(X_new, np.ndarray):
            # 如果是numpy数组，转换为DataFrame并设置列名
            if X_new.ndim == 1:
                X_new = pd.DataFrame([X_new], columns=self.feature_columns)
            else:
                X_new = pd.DataFrame(X_new, columns=self.feature_columns)
        else:
            # 其他类型转换为DataFrame
            X_new = pd.DataFrame([X_new], columns=self.feature_columns)
            
        # 应用归一化
        X_new_scaled = self.scaler.transform(X_new)
        return self.best_model.predict(X_new_scaled)[0]
