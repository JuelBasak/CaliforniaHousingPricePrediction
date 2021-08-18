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


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def model_predict():
    longitude = request.form['longitude']
    if longitude:
        longitude = float(longitude)
    else:
        longitude = 0

    latitude = request.form['latitude']
    if latitude:
        latitude = request.form['latitude']
    else:
        latitude = 0

    housing_median_age = request.form['housing_median_age']
    if housing_median_age:
        housing_median_age = float(housing_median_age)
    else:
        housing_median_age = 0

    total_rooms = request.form['total_rooms']
    if total_rooms:
        total_rooms = float(total_rooms)
    else:
        total_rooms = 0

    total_bedrooms = request.form['total_bedrooms']
    if total_bedrooms:
        total_bedrooms = float(total_bedrooms)
    else:
        total_bedrooms = 0

    population = request.form['population']
    if population:
        population = float(population)
    else:
        population = 0

    households = request.form['households']
    if households:
        households = float(households)
    else:
        households = 0

    median_income = request.form['median_income']
    if median_income:
        median_income = float(median_income)
    else:
        median_income = 0

    ocean_proximity = request.form['ocean_proximity']

    new_data = [
        [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income,
         ocean_proximity]]
    # print(new_data)
    new_df = pd.DataFrame(new_data, columns=num_attribs + cat_attribs)

    try:
        print('Working 1')
        new_data_prepared = full_pipeline.transform(new_df)
        print('Working 2')
        final_model = joblib.load('./model/model.pkl')
        print('Working 3')
        predicted_value = final_model.predict(new_data_prepared)
        print('Working 4')
    except Exception:
        return render_template('index.html', flag=True)
    return render_template('index.html', predicted_value=predicted_value[0], flag=False)

    
if __name__ == '__main__':
    model_process()
    app.run(debug=False)
