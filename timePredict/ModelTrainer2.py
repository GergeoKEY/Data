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
        self.checkpoint_dir = os.path.join(os.path.dirname(model_path), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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

        # 首先划分训练集和测试集
        X_full, X_test, y_full, y_test = train_test_split(
            data[self.feature_columns], data[self.target_column], test_size=0.2, random_state=42
        )
        
        # 从训练集中再划分出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

        # 标准化数据
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=self.feature_columns)
        dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_columns)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=self.feature_columns)

        # XGBoost参数 - 针对小数据集优化
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

        # 训练参数
        batch_size = 10
        initial_batch_size = 20
        num_batches = (len(X_train_scaled) - initial_batch_size) // batch_size

        # 检查是否存在检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.json')
        if os.path.exists(checkpoint_path) and incremental:
            print("Loading existing model checkpoint...")
            self.best_model = xgb.Booster()
            self.best_model.load_model(checkpoint_path)
        else:
            # 初始训练
            X_init = X_train_scaled[:initial_batch_size]
            y_init = y_train.iloc[:initial_batch_size]
            dtrain_init = xgb.DMatrix(X_init, label=y_init, feature_names=self.feature_columns)

            print(f"\n🆕 训练初始模型 ({initial_batch_size} samples)...")
            self.best_model = xgb.train(
                params, dtrain_init, 
                num_boost_round=100,
                evals=[(dval, "eval")], 
                early_stopping_rounds=20
            )

        # 评估初始模型
        y_pred_val = self.best_model.predict(dval)
        y_pred_test = self.best_model.predict(dtest)
        self._print_evaluation_metrics(y_val, y_pred_val, "初始模型验证集")
        self._print_evaluation_metrics(y_test, y_pred_test, "初始模型测试集")

        # 增量训练
        for i in range(1, num_batches + 1):
            current_size = initial_batch_size + i * batch_size
            X_batch = X_train_scaled[:current_size]
            y_batch = y_train.iloc[:current_size]
            dtrain_batch = xgb.DMatrix(X_batch, label=y_batch, feature_names=self.feature_columns)

            # 调整学习率
            learning_rate = 0.05 * (0.95 ** i)
            params["learning_rate"] = learning_rate
            
            print(f"\n🔄 增量训练 batch {i} ({current_size} samples, lr={learning_rate:.4f})...")
            self.best_model = xgb.train(
                params, dtrain_batch, 
                num_boost_round=50,  # 减少每轮的训练轮数
                evals=[(dval, "eval")], 
                early_stopping_rounds=10,
                xgb_model=self.best_model
            )
            
            # 评估当前模型
            y_pred_val = self.best_model.predict(dval)
            y_pred_test = self.best_model.predict(dtest)
            current_mae_val, current_r2_val = self._print_evaluation_metrics(y_val, y_pred_val, f"Batch {i} 验证集")
            current_mae_test, current_r2_test = self._print_evaluation_metrics(y_test, y_pred_test, f"Batch {i} 测试集")
            
            # 更新最佳模型
            if current_mae_val < self.best_mae:
                self.best_mae = current_mae_val
                self.best_r2 = current_r2_val
                self._save_model()
                # 保存检查点
                self.best_model.save_model(checkpoint_path)

        print("\n📊 训练完成！")
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
