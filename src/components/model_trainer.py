# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
# from sklearn.tree import DecisionTreeRegressor
# from dataclasses import dataclass
# from src.utils import save_object,evaluate_models
# from sklearn.metrics import r2_score
# from xgboost import XGBRegressor


# @dataclass
# class ModelTrainerConfig:
#     model_file_path:str=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()

#     def initiate_model_trainer(self,train_array,test_array):
#         try:
            
#             logging.info("take the x_train,x_test,y_train,y_test data")
#             x_train,y_train,x_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             logging.info("take the all models")
#             models={
#                 # "LinearRegression":LinearRegression(),
#                 'KNeighborsRegressor':KNeighborsRegressor(),
#                 "RandomForestRegressor":RandomForestRegressor(),
#                  "XGBRegressor": XGBRegressor(),
#                 "GradientBoostingRegressor":GradientBoostingRegressor(),
#                 "AdaBoostRegressor":AdaBoostRegressor(),
#                 "DecisionTreeRegressor":DecisionTreeRegressor()
#             }

#             logging.info("apply evlaute model for train and test data")
#             model_report:dict=evaluate_models(x_train,y_train,x_test,y_test,models)
            
            
#             logging.info("best model score is:")
#             best_model_score=max(sorted(model_report.values()))

#             print(f"best model score is {best_model_score}")

#             logging.info("best model name is")
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             logging.info("best model from all is")

#             best_model=models[best_model_name]


#             print(best_model)

#             # if best_model_score<0.6:
#             #     raise CustomException("Not found a good model",sys)
#             # logging.info("Best model found on both train and test data")

#             logging.info("predoction for tets data")

#             save_object(
#                 file_path=self.model_trainer_config.model_file_path,
#                 obj=best_model
#                 )
#             prediction=best_model.predict(x_train)

#             scoring=r2_score(y_train,prediction)

#             print("scoring is",scoring)

#         except Exception as e:
#             raise CustomException(e,sys)

import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
               
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None],
    },
    
    "Random Forest": {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512],
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
    },
    
    "Gradient Boosting": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['auto', 'sqrt', 'log2'],
    },
    
    "Linear Regression": {},  # No hyperparameters to tune

    "XGBRegressor": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0],
    },

    # "CatBoosting Regressor": {
    #     'depth': [4, 6, 8, 10],
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'iterations': [30, 50, 100, 200],
    #     'l2_leaf_reg': [1, 3, 5, 7],
    #     'border_count': [32, 50, 100],
    # },

    "AdaBoost Regressor": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512],
        'loss': ['linear', 'square', 'exponential'],
    }
}

            

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(best_model)

            if best_model_score<0.5:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)