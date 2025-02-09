import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)

            return pred
        except Exception as e:
            raise CustomException(e,sys)

        


# cat_onehot_features = ['area_type']
        # cat_ordinal_features = ['size']
        # num_features = ['bath', 'balcony', 'total_sqft']
class CustomData:
        
    def __init__(self,area_type,size,bath,balcony,total_sqft):
        self.area_type=area_type
        self.size=size
        self.bath=bath
        self.balcony=balcony
        self.total_sqft=total_sqft

    def get_data_as_data_frame(self):
        try:
            
            custom_data_input={
                "area_type":[self.area_type],
                "size":[self.size],
                "bath":[self.bath],
                "balcony":[self.balcony],
                "total_sqft":[self.total_sqft],

            }

            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e,sys)

