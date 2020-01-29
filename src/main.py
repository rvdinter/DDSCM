# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:33:16 2020

@author: Raymon van Dinter
Assignment - Forecasting weather-dependent demand main script
"""

import os
import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

from src.demand_forecast_simulator import DemandForecastSimulator


def ignore_warnings():
    """
    Ignore warnings for this script
    :return: None
    """
    warnings.filterwarnings("ignore")


def set_cwd(cwd='M:\WUR\Year_1\DDSCM'):
    """
    Set current working directory, used in-class to enable easy data loading
    :param cwd: string of current working directory
    :return: None
    """
    os.path.abspath(os.getcwd())
    os.chdir(cwd)


def run_sim_datasets(dataframes, plot=False):
    """
    Run the DemandForecastSimulator with MA for all dataframes and report scores
    :param dataframes: list of dataframes
    :param plot: boolean
    :return: None
    """
    # Set warmup time and cooldown time
    TW = 30
    TK = 7
    dataframe_names = ['Dataframe 14-17', 'Dataframe 14-16', 'Dataframe 17']
    # Initialize ForecastSimulator model
    fs = DemandForecastSimulator('MA')

    for df, names in zip(dataframes, dataframe_names):
        TO = df.shape[0] - TW - TK
        fs.set_D(df.demand, 1, TO)
        D, S, Q, I, short, waste, ShortRun, WasteRun = fs.simulate(1, TO)
        print('\n--- %s ---'%names)
        report_rmse_fillrun_shortrun(D[0], S[0], ShortRun, WasteRun)

        if plot:
            plt.plot(D[0], label='D')
            plt.plot(S[0], label='S')
            plt.plot(np.sum(I[0], axis=1), label='I')
            plt.plot(np.sum(short), label='short')
            plt.plot(np.sum(waste), label='waste')
            plt.legend()
            plt.show()


def report_rmse_fillrun_shortrun(D, S, ShortRun, WasteRun):
    """
    Report the RMSE, ShortRun and WasteRun
    :param D: demand
    :param S: predicted demand
    :param ShortRun: fill rate
    :param WasteRun: waste rate
    :return: None
    """
    print('RMSE\t\t: \t%0.3f' % sqrt(mean_squared_error(D, S)))
    print('Fill rate\t: \t%0.3f' % ShortRun)
    print('Waste rate\t: \t%0.3f' % WasteRun)


def preprocess_train_test(train_df, test_df):
    """
    Preprocess the dataframes by enriching, splitting, and scaling
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :return: X and y values for train and test
    """
    # Enrich the dataframe
    train_df = enrich_df(train_df)
    test_df = enrich_df(test_df)

    # Remove demand from X set, keep demand at y
    X_train = train_df.drop(columns=['demand']).values
    X_test = test_df.drop(columns=['demand']).values
    y_train = train_df.demand.values
    y_test = test_df.demand.values

    # Scale input values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


def preprocess_train(train_df):
    """
    Preprocess the dataframe by enriching, splitting, and scaling
    :param train_df: train dataframe
    :return: X and y values for train
    """
    # Enrich the dataframe
    train_df = enrich_df(train_df)

    # Remove demand from X set, keep demand at y
    X_train = train_df.drop(columns=['demand']).values
    y_train = train_df.demand.values

    # Scale input values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    return X_train_scaled, y_train


def enrich_df(df):
    """
    Enrich the dataframe by one-hot-encoding of the weekday and adding polynominal features.
    :param df: dataframe
    :return: dataframe
    """
    df['Date'] = pd.to_datetime(df.Date, format='%Y%m%d', errors='ignore')
    df['dayofweek'] = df.Date.dt.dayofweek
    df = pd.get_dummies(df, columns=['dayofweek'])
    df = df.drop(columns='Date')
    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    df = pd.DataFrame(poly_transformer.fit_transform(df))
    df = df.rename(columns={0: 'demand', 1: 'TempTimes10', 2: 'RainfallTimes10'})
    return df.dropna().replace([np.inf, -np.inf], 0)


def evaluate_models(train_df, test_df):
    """
    Evaluate multiple models using the GridSearchCV object, print scores
    :param train_df: train dataframe
    :param test_df: test dataframe
    :return: None
    """
    X_train_scaled, y_train, X_test_scaled, y_test = preprocess_train_test(train_df, test_df)

    # Apply Gridsearch method to find the best parameters and their RMSE
    models = [SVR(), RandomForestRegressor(), GradientBoostingRegressor(), MLPRegressor()]
    parameters = [{'C': [0.001, 0.1, 10, 100], 'gamma': [0.1, 0.01, 10, 100]},
                  {'n_estimators': [100, 200, 500], 'max_features': ['auto', 'sqrt', 'log2'],
                   'max_depth': [2, 3, 4, 5, 6, 7, 8], 'random_state': [0]},
                  {'learning_rate': [0.001, 0.01, 0.1, 1, 10], 'max_depth': [1, 3, 5], 'random_state': [0]},
                  {'solver': ['adam', 'lbfgs', 'sgd'], 'hidden_layer_sizes': [[10], [10, 10], [10, 10, 10]],
                   'alpha': [0.000001, 0.00001, 0.0001], 'random_state': [0], 'max_iter':[1000000]}]

    for model, params in zip(models, parameters):
        grid = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        y_preds = np.rint(grid.predict(X_test_scaled)).clip(0)

        print('\n--- %s ---' % model.__class__.__name__)
        print("Train score: \t%3.2f" % (grid.score(X_train_scaled, y_train)))
        print("Test score: \t%3.2f" % (grid.score(X_test_scaled, y_test)))
        print('RMSE\t\t: \t%0.3f' % sqrt(mean_squared_error(y_test, y_preds)))
        print(grid.best_params_)


def run_sim_ml_dataset(dataframe, plot=False):
    """
    Run the simulation using the ML model
    :param dataframe: training dataframe
    :param plot: boolean
    :return: None
    """
    # Set warmup time and cooldown time
    TW = 30
    TK = 7
    TO = dataframe.shape[0] - TW - TK

    X, y = preprocess_train(dataframe)

    mlp = MLPRegressor(alpha=0.000001, hidden_layer_sizes=[10], random_state=0, solver='lbfgs')
    mlp.fit(X, y)
    demand = mlp.predict(X)

    # Initialize ForecastSimulator model
    fs = DemandForecastSimulator('ML')
    fs.set_D(demand, 1, TO)
    D, S, Q, I, short, waste, ShortRun, WasteRun = fs.simulate(1, TO)
    report_rmse_fillrun_shortrun(D[0], demand[TW:-TK], ShortRun, WasteRun)

    if plot:
        plt.plot(D[0], label='D')
        plt.plot(S[0], label='S')
        plt.plot(np.sum(I[0], axis=1), label='I')
        plt.plot(np.sum(short), label='short')
        plt.plot(np.sum(waste), label='waste')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ignore_warnings()

    # Adjust current working directory if needed
    # set_cwd()

    # Load data into dataframes
    demand_weather_14_17 = pd.read_excel('data/Data2014-2017DemandWeather.xlsx')
    demand_weather_14_16 = pd.read_excel('data/Data2014-2016DemandWeather.xlsx')
    demand_weather_17 = pd.read_excel('data/Data2017DemandWeather.xlsx')

    # A) Report RMSE, as well as fill rate and waste rate and run two subsets
    print('\n---- Run the Simulator ----')
    dataframes = [demand_weather_14_17.drop(columns=['Date']), demand_weather_14_16.drop(columns=['Date']),
                  demand_weather_17.drop(columns=['Date'])]
    run_sim_datasets(dataframes, plot=False)

    print('\n\n---- Model Evaluation ----')
    evaluate_models(demand_weather_14_16, demand_weather_17)

    print('\n\n---- Run the Simulator ----')
    run_sim_ml_dataset(demand_weather_14_17, plot=False)
