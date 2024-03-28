import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def read_data(data_dir):
    '''
    Prepare training data
    Input:
        data_dir <str> : path to data file (csv), can be train set or test set
    Returns:
        Dataframe after clean
    '''
    # Load data
    df = pd.read_csv(data_dir)
    DT_bmi_pipe = Pipeline(steps=[('scale', StandardScaler()),
                                  ('dtr', DecisionTreeRegressor(random_state=42))])
    X_BMI = df[['age', 'gender', 'bmi']].copy()
    X_BMI.gender = X_BMI.gender.replace(
        {'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)

    # Handling NA data
    na_data = X_BMI[X_BMI.bmi.isna()]
    X_BMI = X_BMI[~X_BMI.bmi.isna()]
    Y_BMI = X_BMI.pop('bmi')
    DT_bmi_pipe.fit(X_BMI, Y_BMI)
    predicted_bmi = pd.Series(DT_bmi_pipe.predict(
        na_data[['age', 'gender']]), index=na_data.index)
    df.loc[na_data.index, 'bmi'] = predicted_bmi
    df.drop(df[df['gender'] == "Other"].index, inplace=True)
    return df


def label_encoder(X):
    X["gender"] = X["gender"].replace({"Male": 0, "Female": 1}).astype(float)
    X["hypertension"] = X["hypertension"].replace(
        {"No": 0, "Yes": 1}).astype(float)
    X["heart_disease"] = X["heart_disease"].replace(
        {"No": 0, "Yes": 1}).astype(float)
    X["Residence_type"] = X["Residence_type"].replace(
        {"Rural": 0, "Urban": 1}).astype(float)
    X["smoking_status"] = X["smoking_status"].replace({"Unknown": 0, "formerly smoked": 1,
                                                       "never smoked": 2, "smokes": 3}).astype(float)


def scaler_data(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)


if __name__ == '__main__':
    df = read_data('./dataset/stroke-data.csv')
    df = df.drop(["id", "ever_married", "work_type"], 1)
    label_encoder(df)
    new_header = ['gender', 'age', 'hypertension', 'heart_disease',
                  'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'smoke']
    df_new = pd.DataFrame(df)
    df_new.to_csv('./dataset/stroke-data-processed.csv',
                  header=new_header, index=False)
    print('DONE')
