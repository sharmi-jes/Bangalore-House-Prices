import os 
import sys
import pandas as pd
import logging
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


logging.info("create a dataingestion config class using datclass(without using init method)")
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts",'train.csv')
    test_data_path:str=os.path.join("artifacts","test.csv")
    raw_data_path:str=os.path.join("artifacts","raw.csv")


logging.info("create a data ingestion class")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Data ingestion is started")
        try:
            logging.info("read the dataset")
            df=pd.read_csv(r"E:\Resume Machine Learning Projects\Bangalore House Prices\notebook\cleaned.csv")

            logging.info("create a directory for data path(train,test,raw)")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("read the dataset to raw path")
            df.to_csv(self.ingestion_config.raw_data_path,header=False)

            logging.info("train test split is started")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path)

            test_set.to_csv(self.ingestion_config.test_data_path)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()


    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)