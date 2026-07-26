# Standard library
import os  # OS module for interacting with the operating system
import numpy as np

# Data manipulation and visualization
import pandas as pd  # Pandas for data handling and manipulation
import matplotlib.pyplot as plt  # Matplotlib for visualization
import seaborn as sns  # Seaborn for enhanced plotting

# Time series processing and transformation
import darts  # Darts library for time series modeling
from darts import TimeSeries  # TimeSeries object for handling time series data
from darts.dataprocessing.transformers import Scaler, Diff  # Scaling transformer for normalization

# Custom preprocessing classes
from classes_preprocessing import SplitByCodeBss, AnomalyDetector, StationarityTester

# Time series forecasting models
from darts.models import (
    TCNModel, BlockRNNModel,
    NBEATSModel, NHiTSModel, RandomForest, LinearRegressionModel, XGBModel
)

# Error metrics and evaluation
from darts.metrics import rmse, mape  # RMSE and MAPE error metrics
import hydroeval as he  # Hydrological evaluation metrics


# Stationary data
def stationary_data(df, unique_codes, bss_split):
    # preprocessing data
    for code in unique_codes:

        print(f"Traitement du code_bss : {code}")
        # Filter data for current code_bss

        df_code = df[df['code_bss'] == code]
        # Convert the 'time' column to datetime and set it as an index
        df_code['time'] = pd.to_datetime(df_code['time'])
        df_code.set_index('time', inplace=True)

        # Apply the preprocessing pipeline
        # 1. Split data by code_bss
        list_of_dfs = bss_split.transform(df_code)

        # Select the first piezometer
        # index_piezometre = 0

        # Anomaly detection
        detector = AnomalyDetector(method='hampel', window_length=50, n_sigma=4)
        anomalies_dict, cleaned_dfs = detector.transform(list_of_dfs)

        # Resampling
        resampled_df_lvl = [df.resample('W').mean() for df in cleaned_dfs[0]]
        resampled_df_etp = [df.resample('W').mean() for df in cleaned_dfs[1]]
        resampled_df_pluie = [df.resample('W').mean() for df in cleaned_dfs[2]]

        # Imputation: Interpolation of missing values
        imput_dfs_lvl = [df.interpolate(method='polynomial', order=2) for df in resampled_df_lvl]
        imput_dfs_etp = [df.interpolate(method='polynomial', order=2) for df in resampled_df_etp]
        imput_dfs_pluie = [df.interpolate(method='polynomial', order=2) for df in resampled_df_pluie]

        # Stationarity test
        tester = StationarityTester(significance_level=0.05)
        stationary_series_lvl, non_stationary_series_lvl = tester.test_stationarity(imput_dfs_lvl)
        stationary_series_etp, non_stationary_series_etp = tester.test_stationarity(imput_dfs_etp)
        stationary_series_pluie, non_stationary_series_pluie = tester.test_stationarity(imput_dfs_pluie)

        # Converting to TimeSeries Darts
        lvl_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                           stationary_series_lvl]
        etp_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                           stationary_series_etp]
        pluie_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                             stationary_series_pluie]

        # Combination of stationary series
        liste_piezo_lvl_final = lvl_dtimeseries
        liste_piezo_etp_final = etp_dtimeseries
        liste_piezo_pluie_final = pluie_dtimeseries

        # Combination of covariates
        combined_covariates = [liste_piezo_etp_final[i].concatenate(liste_piezo_pluie_final[i], axis=1) for i in
                               range(len(liste_piezo_pluie_final))]

        # Add data scaling
        scaler_lvl, scaler_cov = Scaler(), Scaler()
        scaled_lvl = [scaler_lvl.fit_transform(ts) for ts in liste_piezo_lvl_final]
        scaled_cov = [scaler_cov.fit_transform(ts) for ts in combined_covariates]

        # Split data into train/test
        lvl_train, lvl_test = [], []
        for ts in liste_piezo_lvl_final:
            train_test_split_date = pd.Timestamp('20180101')
            train_val, test = ts.split_before(train_test_split_date)
            lvl_train.append(train_val)
            lvl_test.append(test)

        cov_train, cov_test = [], []
        for ts in combined_covariates:
            train_test_split_date_cov = pd.Timestamp('20180101')
            train_val_cov, test_cov = ts.split_before(train_test_split_date_cov)
            cov_train.append(train_val_cov)
            cov_test.append(test_cov)

        # Data scaling
        scaler_lvl, scaler_cov = Scaler(), Scaler()
        scaled_lvl_train = [scaler_lvl.fit_transform(ts) for ts in lvl_train]
        scaled_cov_train = [scaler_cov.fit_transform(ts) for ts in cov_train]
        scaled_lvl_test = [scaler_lvl.transform(ts) for ts in lvl_test]
        scaled_cov_test = [scaler_cov.transform(ts) for ts in cov_test]

        # Convert to dataframe
        df_scaled_lvl_train = [scaled_lvl_train[i].to_dataframe() for i in range(len(scaled_lvl_train))]
        df_scaled_lvl_test = [scaled_lvl_test[i].to_dataframe() for i in range(len(scaled_lvl_test))]

        data_level_train = [df_scaled_lvl_train[i].reset_index() for i in range(len(scaled_lvl_train))]
        data_level_test = [df_scaled_lvl_test[i].reset_index() for i in range(len(scaled_lvl_test))]

        data_level = [pd.concat([data_level_train[i], data_level_test[i]]) for i in range(len(data_level_train))]
        # df_data_level = [data_level[i].rename(columns={data_level[i].columns[1]:'y'}, inplace=True) for i in range(len(data_level))]

        df_scaled_cov_train = [scaled_cov_train[i].to_dataframe() for i in range(len(scaled_cov_train))]
        df_scaled_cov_test = [scaled_cov_test[i].to_dataframe() for i in range(len(scaled_cov_test))]

        data_cov_train = [df_scaled_cov_train[i].reset_index() for i in range(len(scaled_cov_train))]
        data_cov_test = [df_scaled_cov_test[i].reset_index() for i in range(len(scaled_cov_test))]

        data_cov = [pd.concat([data_cov_train[i], data_cov_test[i]]) for i in range(len(data_cov_test))]
        # df_data_cov = [data_cov[i].rename(columns={data_cov[i].columns[1]:'etp',
        #               data_cov[i].columns[2]:'pluie'},inplace=True) for i in range(len(data_cov))]

        # Merge data
        final_data = [pd.merge(data_level[i], data_cov[i], how='left', on='time') for i in range(len(data_level))]

        [final_data[i].rename(columns={final_data[i].columns[1]: 'y',
                                       final_data[i].columns[2]: 'etp',
                                       final_data[i].columns[3]: 'pluie'},
                              inplace=True) for i in range(len(final_data))]

        print(f"Write data to file for : {code}")
        for i in range(len(final_data)):
            final_data[i].insert(1, "unique_id", code)
            final_data[i].to_csv(f"../data/{code}.csv")

        # Nonstationary data


def nonstationary_data(df, unique_codes, bss_split):
    # preprocessing stationary data
    for code in unique_codes:

        print(f"Traitement du code_bss : {code}")
        # Filter data for current code_bss
        df_code = df[df['code_bss'] == code]
        # Convert the 'time' column to datetime and set it as an index
        df_code['time'] = pd.to_datetime(df_code['time'])
        df_code.set_index('time', inplace=True)

        # Apply the preprocessing pipeline

        # 1. Split data by code_bss
        list_of_dfs = bss_split.transform(df_code)

        # Select the first piezometer
        # index_piezometre = 0

        # Anomaly detection
        detector = AnomalyDetector(method='hampel', window_length=50, n_sigma=4)
        anomalies_dict, cleaned_dfs = detector.transform(list_of_dfs)

        # Resampling
        resampled_df_lvl = [df.resample('W').mean() for df in cleaned_dfs[0]]
        resampled_df_etp = [df.resample('W').mean() for df in cleaned_dfs[1]]
        resampled_df_pluie = [df.resample('W').mean() for df in cleaned_dfs[2]]

        # Imputation : Interpolation of missing values
        imput_dfs_lvl = [df.interpolate(method='polynomial', order=2) for df in resampled_df_lvl]
        imput_dfs_etp = [df.interpolate(method='polynomial', order=2) for df in resampled_df_etp]
        imput_dfs_pluie = [df.interpolate(method='polynomial', order=2) for df in resampled_df_pluie]

        # Stationarity test
        tester = StationarityTester(significance_level=0.05)
        stationary_series_lvl, non_stationary_series_lvl = tester.test_stationarity(imput_dfs_lvl)
        stationary_series_etp, non_stationary_series_etp = tester.test_stationarity(imput_dfs_etp)
        stationary_series_pluie, non_stationary_series_pluie = tester.test_stationarity(imput_dfs_pluie)

        # Conversion en TimeSeries Darts
        lvl_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                           non_stationary_series_lvl]
        etp_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                           stationary_series_etp]
        pluie_dtimeseries = [TimeSeries.from_dataframe(df.to_frame(), value_cols=df.name, freq='W') for df in
                             stationary_series_pluie]

        # Combination of stationary series
        liste_piezo_lvl_final = lvl_dtimeseries
        liste_piezo_etp_final = etp_dtimeseries
        liste_piezo_pluie_final = pluie_dtimeseries

        # Combination of covariates
        combined_covariates = [liste_piezo_etp_final[i].concatenate(liste_piezo_pluie_final[i], axis=1) for i in
                               range(len(liste_piezo_pluie_final))]

        # Differencer transformer - Store the transformer for the inverse_transform later
        differencer = Diff(lags=1)
        diff_lvl = []
        # Keep the last values before differencing for later reconstruction
        last_values_before_diff = []

        for i, ts in enumerate(liste_piezo_lvl_final):
            # Keep the last value before differentiation
            last_values_before_diff.append(ts.values()[-1][0])
            # Apply differentiation
            differencer.fit(ts)
            diff_lvl.append(differencer.transform(ts))

        # Split des données en train/test
        lvl_train, lvl_test = [], []
        original_lvl_train, original_lvl_test = [], []  # Keep a copy of the original series
        train_test_split_date = pd.Timestamp('20180101')

        for i, ts in enumerate(diff_lvl):
            train_val, test = ts.split_before(train_test_split_date)
            lvl_train.append(train_val)
            lvl_test.append(test)

            # Keep the original (undifferentiated) parts
            original_train, original_test = liste_piezo_lvl_final[i].split_before(train_test_split_date)
            original_lvl_train.append(original_train)
            original_lvl_test.append(original_test)

        cov_train, cov_test = [], []
        for ts in combined_covariates:
            train_val_cov, test_cov = ts.split_before(train_test_split_date)
            cov_train.append(train_val_cov)
            cov_test.append(test_cov)

        # Scaling des données
        scaler_lvl, scaler_cov = Scaler(), Scaler()
        scaled_lvl_train = [scaler_lvl.fit_transform(ts) for ts in lvl_train]
        scaled_cov_train = [scaler_cov.fit_transform(ts) for ts in cov_train]
        scaled_lvl_test = [scaler_lvl.transform(ts) for ts in lvl_test]
        scaled_cov_test = [scaler_cov.transform(ts) for ts in cov_test]

        # Convert to dataframe
        df_scaled_lvl_train = [scaled_lvl_train[i].to_dataframe() for i in range(len(scaled_lvl_train))]
        df_scaled_lvl_test = [scaled_lvl_test[i].to_dataframe() for i in range(len(scaled_lvl_test))]

        data_level_train = [df_scaled_lvl_train[i].reset_index() for i in range(len(scaled_lvl_train))]
        data_level_test = [df_scaled_lvl_test[i].reset_index() for i in range(len(scaled_lvl_test))]

        data_level = [pd.concat([data_level_train[i], data_level_test[i]]) for i in range(len(data_level_train))]
        # df_data_level = [data_level[i].rename(columns={data_level[i].columns[1]:'y'}, inplace=True) for i in range(len(data_level))]

        df_scaled_cov_train = [scaled_cov_train[i].to_dataframe() for i in range(len(scaled_cov_train))]
        df_scaled_cov_test = [scaled_cov_test[i].to_dataframe() for i in range(len(scaled_cov_test))]

        data_cov_train = [df_scaled_cov_train[i].reset_index() for i in range(len(scaled_cov_train))]
        data_cov_test = [df_scaled_cov_test[i].reset_index() for i in range(len(scaled_cov_test))]

        data_cov = [pd.concat([data_cov_train[i], data_cov_test[i]]) for i in range(len(data_cov_test))]
        # df_data_cov = [data_cov[i].rename(columns={data_cov[i].columns[1]:'etp',
        #               data_cov[i].columns[2]:'pluie'},inplace=True) for i in range(len(data_cov))]

        # Merge data
        final_data = [pd.merge(data_level[i], data_cov[i], how='left', on='time') for i in range(len(data_level))]

        [final_data[i].rename(columns={final_data[i].columns[1]: 'y',
                                       final_data[i].columns[2]: 'etp',
                                       final_data[i].columns[3]: 'pluie'},
                              inplace=True) for i in range(len(final_data))]

        for i in range(len(final_data)):
            final_data[i].insert(1, "unique_id", code)
            final_data[i].to_csv(f"../data/{code}.csv")


# Process start
if __name__ == "__main__":

    # 1. Prepare the input data
    df = pd.read_csv('../data/backup/core_data.csv', index_col=0)

    ## Convert column 'time' to datetime and set 'time' as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    ## Filter data for the period of interest
    df = df[(df.index >= '1995-01-01') & (df.index <= '2020-01-01')]
    df = df.reset_index()

    ## Iterate over each code_bss = Iterate over each code_bss
    unique_codes = df['code_bss'].unique()

    ## Spit data columns by code_bss
    bss_split = SplitByCodeBss(columns=['niveau_nappe_eau', 'ETP_Q', 'PRELIQ_Q'])

    # 2. Start the preprocessing step

    ## Stationary data
    stationary_data(df, unique_codes, bss_split)

    ## Nonstationary data
    nonstationary_data(df, unique_codes, bss_split)

    # 3. End
    print("The preprocessing step finishes!")