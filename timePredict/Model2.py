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

        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        #已经有模型了
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.best_model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"已经加载了现有的模型:{model_path}")
            except Exception as e:
                print(f"加载模型失败: {str(e)}")
        
    def train_model(self,data,incremental= False):
        """
        训练模型
        :param data: pandas.DataFrame, 包含特征和目标变量的数据
        :param incremental: bool, 是否增量训练
        :return: 是否训练成功
        """
        if len(data)<5:
            print("数据量不足，无法进行训练")
            return False
        
        X = data[self.feature_columns]
        y = data[self.target_column]

        # 增量训练分支
        if incremental and self.best_model is not None:
            print("进行一轮增量训练……")
            X_new = X
            y_new = y
        else:
            #首次进行训练
            X_train,X_val,y_train,y_val = train_test_split(
                X,y,test_size=0.2,random_state=42
            )
            X_new = X_train
            y_new = y_train
            self.val_data = (X_val, y_val)  # 保存验证集用于评估
        
        if self.scaler is None:
            print("创建新的标准化器...")
            self.scaler = StandardScaler()
            X_new_scaled = self.scaler.fit_transform(X_new)
        else:
            print("使用现有标准化器...")
            X_new_scaled = self.scaler.transform(X_new)
        
        #转换数据格式
        dtrain = xgb.DMatrix(X_new_scaled, label=y_new, feature_names=self.feature_columns)
        # 准备验证集 dval
        if hasattr(self, 'val_data'):
            X_val, y_val = self.val_data
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_columns)
            eval_list = [(dval, "eval")]
        else:
            eval_list = []
        
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
