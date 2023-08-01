import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config = DataTransformationConfig):
        self.config = config

    ## Note: You can add different data transformation techniques such as scaler , PCA and all
    # You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train test split because the data is already cleaned up


    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into train and test sets. (0.75 , 0.25) split
        train , test = train_test_split(data,random_state=0)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info(f"Splited data into training and testing sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)