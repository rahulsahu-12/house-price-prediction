import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attributes,cat_attributes):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaling",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline=ColumnTransformer([
        ("numerical",num_pipeline,num_attributes),
        ("categorical",cat_pipeline,cat_attributes)
    ])

    return full_pipeline

#if the model is not exists then we train the model
if not os.path.exists(MODEL_FILE):
    housing=pd.read_csv("housing.csv")

    housing['income_cat']=pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=(1,2,3,4,5))
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].copy()
        test_set = housing.loc[test_index].copy()

# Drop 'income_cat' from both sets
        train_set.drop("income_cat", axis=1, inplace=True)
        test_set.drop("income_cat", axis=1, inplace=True)

        # Save test_set as input.csv
        test_set.drop("median_house_value", axis=1).to_csv("input.csv", index=False)

        # Use train_set to train model
        housing_features = train_set.drop("median_house_value", axis=1)
        housing_labels = train_set["median_house_value"].copy()
        

        num_attributes =housing_features.drop("ocean_proximity",axis=1).columns.tolist()
        cat_attributes =["ocean_proximity"]

        pipeline=build_pipeline(num_attributes,cat_attributes)

        cleaned_housing=pipeline.fit_transform(housing_features)

        model=RandomForestRegressor(random_state=42)
        model.fit(cleaned_housing,housing_labels)

        joblib.dump(model,MODEL_FILE)
        joblib.dump(pipeline,PIPELINE_FILE)

        print("model is trained and saved")

# else:

#     model=joblib.load(MODEL_FILE)
#     pipeline=joblib.load(PIPELINE_FILE)

#     input_data=pd.read_csv("input.csv")
#     transformed_input=pipeline.transform(input_data)
#     prediction=model.predict(transformed_input)
#     input_data["median_house_value"]=prediction

#     input_data.to_csv("output.csv",index=False)

#     print("inference is complete and saved to output.csv")

#if you want to deploy preediction will be in the app.py part
