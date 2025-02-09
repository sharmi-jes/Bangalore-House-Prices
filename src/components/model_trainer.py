import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from src.utils import save_object,evaluate_models
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("take the x_train,x_test,y_train,y_test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("take the all models")
            models={
                "LinearRegression":LinearRegression(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor()
            }

            logging.info("apply evlaute model for train and test data")
            model_report:dict=evaluate_models(x_train,y_train,x_test,y_test,models)
            
            
            logging.info("best model score is:")
            best_model_score=max(sorted(model_report.values()))

            print(f"best model score is {best_model_score}")

            logging.info("best model name is")
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            logging.info("best model from all is")

            best_model=models[best_model_name]


            print(best_model)

            # if best_model_score<0.6:
            #     raise CustomException("Not found a good model",sys)
            # logging.info("Best model found on both train and test data")

            logging.info("predoction for tets data")

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model
                )
            prediction=best_model.predict(x_train)

            scoring=r2_score(y_train,prediction)

            print("scoring is",scoring)

        except Exception as e:
            raise CustomException(e,sys)