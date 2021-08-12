from flask import Flask, render_template, request
import joblib
import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html', loading=False)

@app.route('/predict', methods=['POST'])
def model_predict():
    longitude = request.form['longitude']
    latitude = request.form['latitude']
    housing_median_age = request.form['housing_median_age']
    total_rooms = request.form['total_rooms']
    total_bedrooms = request.form['total_bedrooms']
    population = request.form['population']
    households = request.form['households']
    median_income = request.form['median_income']
    ocean_proximity = request.form['ocean_proximity']

    
    new_data = [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]]
    new_df = pd.DataFrame(new_data, columns=num_attribs + cat_attribs)

    new_data_prepared = full_pipeline.transform(new_df)

    final_model = joblib.load('./model/model.pkl')

    predicted_value = final_model.predict(new_data_prepared)
    return render_template('result.html', predicted_value=predicted_value[0])
    print(predicted_value[0])  

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def fetch_data(housing_url, housing_path):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_data(housing_path):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


room_ix, bedroom_ix, population_ix, households_ix = 3, 4, 5, 6

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

num_attribs = ['longitude',
            'latitude',
            'housing_median_age',
            'total_rooms',
            'total_bedrooms',
            'population',
            'households',
            'median_income']

cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

def model_process():
    DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
    HOUSING_PATH = os.path.join('datasets/', 'housing')
    HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'

    if not os.path.exists('./datasets/housing/housing.csv'):
        fetch_data(HOUSING_URL, HOUSING_PATH)

    housing = load_data(HOUSING_PATH)

    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]


    strat_train_set.drop('income_cat', axis=1, inplace=True)
    strat_test_set.drop('income_cat', axis=1, inplace=True)

    housing = strat_train_set.copy()

    full_pipeline.fit_transform(housing.drop('median_house_value', axis=1))


if __name__=='__main__':
    model_process()
    app.run(debug=True)
