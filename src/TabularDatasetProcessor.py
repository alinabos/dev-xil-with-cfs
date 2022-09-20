import logging as log
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class TabularDatasetProcessor():

    def __init__(self, raw_data: pd.DataFrame):
        """Initialize processor for tabular datasets, start preprocessing of the input raw_data and fit encoder for transforming categorical features

        Args:
            raw_data (pandas.DataFrame): unpreprocessed data that was read from e.g. a csv file
        """

        self.data = None
        # assume last column is target
        self.target_name = raw_data.columns[-1]

        self.initial_features = raw_data.columns
        # list of numerical features for counterfactual explanations
        self.numerical_features = [column for column in raw_data.columns if (raw_data[column].dtypes == "int" or raw_data[column].dtypes == "float")] 
        # initial (before preprocessing) list of categorical features for inverse transform OHE
        self.categorical_features = [column for column in raw_data.columns[raw_data.dtypes == "object"]][:-1]
        # feature count after preprocessing
        self.preprocessed_feature_count = None # for helper model initialization
        
        # encoder for categorical values
        self.ohc_encoder = OneHotEncoder(categories="auto", sparse=False)
        self.label_encoder = LabelEncoder()



        # drop NaN values to ensure encoders are fitted correctly
        raw_data_tmp = self.drop_nan(raw_data)
        
        # fit OneHotEncoder and LabelEncoder
        self.ohc_encoder.fit(raw_data_tmp[self.categorical_features])
        self.label_encoder.fit(raw_data_tmp[self.target_name])

        # transform categorical features of the data
        self.data = self.preprocess_data(raw_data)
        self.preprocessed_feature_count = len(self.data.columns)

    def drop_nan(self, data):
        data_tmp = data.dropna(axis=0, how="any")
        data_tmp.reset_index(drop=True, inplace=True)
        return data_tmp


    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """basic preprocessing of the training data: drop rows with NaN, OneHotEncoding of categorical data

        Args:
            raw_data (pandas.DataFrame): unpreprocessed data that was read from e.g. a csv file

        Returns:
            pandas.DataFrame: preprocessed data
        """
        
        log.debug("Start preprocessing")
        log.info(f"Shape of the dataset before preprocessing: {raw_data.shape}")

        # drop NaN values
        log.debug("Dropping NaN values")
        old_row_count = raw_data.shape[0]
        raw_data = self.drop_nan(raw_data)
        new_row_count = raw_data.shape[0]
        log.debug(f"{old_row_count - new_row_count} of {old_row_count} rows were dropped. New count: {new_row_count}")
            
        log.debug(f"Transform the following categorical variables with one-hot encoding: {self.categorical_features}")
        data = self.encode_features(raw_data)
        log.debug(f"Encoded target variable. Number of classes found: {len(self.label_encoder.classes_)}. Mapping of classes: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        log.info(f"Finished preprocessing: Shape of the dataset after the preprocessing (including target): {data.shape}")

        return data

    def encode_features(self, raw_data):
        # encode categorical data and target column in One Hot Encoding
        cat_data = self.ohc_encoder.transform(raw_data[self.categorical_features])
        
        # put categorical features and numerical features together in one DataFrame
        data = pd.concat([raw_data[self.numerical_features], 
                    pd.DataFrame(cat_data, columns=self.ohc_encoder.get_feature_names_out(self.categorical_features))], axis=1)

        # encode target variable with label encoding
        data[self.target_name] = self.label_encoder.transform(y=raw_data[self.target_name])
        
        # reorder columns so the target variable is the last column again
        new_order = data.columns.tolist()
        new_order.remove(self.target_name)
        new_order += [self.target_name]
        data = data.reindex(columns=new_order)
        return data
    

    def inverse_transform_data_and_target(self, data: pd.DataFrame):
        # inverse_transform feature data
        inv_feature_data = self.ohc_encoder.inverse_transform(data[self.ohc_encoder.get_feature_names_out(self.categorical_features)])
        inv_feature_df = pd.DataFrame(data=inv_feature_data, columns=self.categorical_features)

        # inverse_transform target column
        inv_target_data = self.label_encoder.inverse_transform(data[self.target_name])
        inv_target_df = pd.DataFrame(data=inv_target_data, columns=[self.target_name])

        # put different DataFrames together: numerical data, categorical features, target column
        inv_data = pd.concat([data[self.numerical_features], inv_feature_df, inv_target_df], axis=1)
        # reindex DataFrame to restore initial column order
        inv_data = inv_data.reindex(columns=self.initial_features)

        return inv_data