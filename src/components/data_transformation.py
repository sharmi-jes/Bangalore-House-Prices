import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts",'preprocesssor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    
    
   
    
    
    

    def get_data_transformation(self):
     try:
        logging.info("Data Transformation is started.")

        # Define feature types
        cat_onehot_features = ['area_type']
        cat_ordinal_features = ['size']
        num_features = ['bath', 'balcony', 'total_sqft']

        # # Convert total_sqft to numerical values
        # train_df["total_sqft"] = train_df["total_sqft"].apply(self.convert_sqft_to_num)
        # test_df["total_sqft"] = test_df["total_sqft"].apply(self.convert_sqft_to_num)

        # # Fill missing values for total_sqft
        # train_df["total_sqft"].fillna(train_df["total_sqft"].median(), inplace=True)
        # test_df["total_sqft"].fillna(test_df["total_sqft"].median(), inplace=True)

        # Define numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Define OneHotEncoder pipeline (NO SCALING needed)
        onehot_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot_encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Define OrdinalEncoder pipeline (WITH SCALING)
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), 
                ("scaler", StandardScaler())  # Scaling applied here
            ]
        )

        # Define ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, num_features),
                ("onehot_pipeline", onehot_pipeline, cat_onehot_features),
                ("ordinal_pipeline", ordinal_pipeline, cat_ordinal_features)
            ]
        )

        logging.info("Data transformation is completed.")
        return preprocessor

     except Exception as e:
        raise CustomException(e, sys)



    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            logging.info("read the train and test data")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            

            logging.info("take the preprocessor obj for do data transformation")

            preprocessor_obj = self.get_data_transformation()


            print(f"training data is {train_df.head(2)}")
            print(f"tesitng data {test_df.head(2)}")

            target_col_name="price"
            # num_features = ['bath', 'balcony',  'total_sqft'] 

            logging.info("from the train  and test input data i have to drop target col")
            input_train_df=train_df.drop(columns=[target_col_name],axis=1)
            target_train_df=train_df[target_col_name]

            input_test_df=test_df.drop(columns=[target_col_name],axis=1)
            target_test_df=test_df[target_col_name]

            logging.info("apply preprocesssor obj to the input train and test data")

            input_train_array=preprocessor_obj.fit_transform(input_train_df)
            input_test_array=preprocessor_obj.transform(input_test_df)

            logging.info("combine the input data along with target col")

            train_arr=np.c_[
                input_train_array,np.array(target_train_df)

            ]

            test_arr=np.c_[
                input_test_array,np.array(target_test_df)
            ]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info("return train and test data and preprocessor file path") 
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            