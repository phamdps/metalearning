
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split
from darts import TimeSeries, concatenate
from sktime.transformations.series.outlier_detection import HampelFilter
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.series.impute import Imputer
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from darts.models import  XGBModel,RandomForest,LinearRegressionModel
from statsmodels.tsa.stattools import adfuller

class SplitByCodeBss(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not all(col in X.columns for col in self.columns):
            raise ValueError("Certaines colonnes spécifiées n'existent pas dans le DataFrame")
        
        list_of_dfs = {col: [] for col in self.columns}
        
        unique_code_bss = X['code_bss'].unique()
        
        for code_bss in unique_code_bss:
            df_filtered = X[X['code_bss'] == code_bss].copy()
            
            for col in self.columns:
                df_col = df_filtered[[col]].copy()
                df_col.rename(columns={col: code_bss}, inplace=True)
                list_of_dfs[col].append(df_col)
        
        return [list_of_dfs[col] for col in self.columns]
    


class AnomalyDetector(BaseEstimator, TransformerMixin):
    def __init__(self, method='hampel', window_length=36, n_sigma=4, k=1.4826, n_estimators=60, contamination=0.01, random_state=42):
        self.method = method
        self.window_length = window_length
        self.n_sigma = n_sigma
        self.k = k
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X,y=None):
        if not isinstance(X, list) or not all(isinstance(sublist, list) for sublist in X):
            raise ValueError("X doit être une liste de listes de DataFrames")
        
        all_anomalies_dict = []
        all_cleaned_dfs = []
        
        for sublist in X:
            if not all(isinstance(df, pd.DataFrame) for df in sublist):
                raise ValueError("Chaque sous-liste doit contenir des DataFrames")
            
            anomalies_dict = {}
            cleaned_dfs = []
            
            for df in sublist:
                code_bss = df.columns[0]
                series = df[code_bss]
                
                if self.method == 'hampel':
                    # Initialiser le HampelFilter
                    detector = HampelFilter(window_length=self.window_length, n_sigma=self.n_sigma, k=self.k, return_bool=True)
                    anomalies = detector.fit_transform(series)
                    anomaly_dates = series.index[anomalies]
                    
                    # Nettoyer les outliers
                    df_cleaned = df[~anomalies]
                    
                elif self.method == 'isolation_forest':
                    # Préparer les données pour IsolationForest
                    data = series.values.reshape(-1, 1)
                    
                    # Appliquer l'Isolation Forest
                    model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, random_state=self.random_state)
                    model.fit(data)
                    anomalies = model.predict(data) == -1
                    anomaly_dates = series.index[anomalies]
                    
                    # Nettoyer les outliers
                    df_cleaned = df[~anomalies.flatten()]
                    
                else:
                    raise ValueError("Méthode non reconnue. Utilisez 'hampel' ou 'isolation_forest'.")
                
                # Stocker les résultats
                anomalies_dict[code_bss] = {
                    'series': series,
                    'anomalies': anomaly_dates,
                    'anomaly_scores': None  # Les scores ne sont pas fournis pour HampelFilter, mais peuvent être ajoutés pour IsolationForest
                }
                
                # Ajouter le DataFrame nettoyé à la liste
                cleaned_dfs.append(df_cleaned)
            
            all_anomalies_dict.append(anomalies_dict)
            all_cleaned_dfs.append(cleaned_dfs)
        
        return all_anomalies_dict,all_cleaned_dfs


class Resampler(BaseEstimator, TransformerMixin):
    def __init__(self, frequency='D'):
        self.frequency = frequency

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X doit être un DataFrame")

        code_bss = X.columns[0]
        series = X[code_bss]

        # Échantillonner en fonction de la fréquence
        if self.frequency == 'D':
            df_resampled = series.resample('D').mean()  # Moyenne quotidienne
        elif self.frequency == 'W':
            df_resampled = series.resample('W').mean()  # Moyenne hebdomadaire
        else:
            raise ValueError("Fréquence non reconnue. Utilisez 'D' pour quotidien ou 'W' pour hebdomadaire.")

        df_resampled = df_resampled.to_frame()
        df_resampled.rename(columns={code_bss: f'{code_bss}_{self.frequency}'}, inplace=True)

        return df_resampled



class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=10, weights='distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputers = []

    def fit(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError("X doit être une liste de DataFrames")

        # Ajuster un KNNImputer sur chaque DataFrame
        for df in X:
            imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
            imputer.fit(df)
            self.imputers.append(imputer)

        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError("X doit être une liste de DataFrames")

        # Appliquer l'imputation avec chaque KNNImputer
        imputed_dfs = []
        for df, imputer in zip(X, self.imputers):
            imputed_df = pd.DataFrame(index=df.index, columns=df.columns, data=imputer.transform(df))
            imputed_dfs.append(imputed_df)

        # Retourner les DataFrames imputés
        return imputed_dfs


### Inverse Scaling Transformer for DataFrames
### DataFrame to Darts TimeSeries Converter
class ToDartsTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("X doit être un tuple avec 2 éléments : anomalies_dict, all_unscaled_dfs")
        anomalies_dict, imputed_dfs = X
        if not isinstance(imputed_dfs, list) or len(imputed_dfs) != 3:
            raise ValueError("Le deuxième élément de X doit être une liste contenant exactement trois DataFrames")
        
        # Vérification des DataFrames
        for df in imputed_dfs:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Les éléments de all_unscaled_dfs  doivent être des DataFrames")

        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("X doit être un tuple avec 2 éléments : anomalies_dict, all_unscaled_dfs")
        anomalies_dict, imputed_dfs = X
        if not isinstance(imputed_dfs, list) or len(imputed_dfs) != 3:
            raise ValueError("Le deuxième élément de X doit être une liste contenant exactement trois DataFrames")

        # Convertir chaque DataFrame en séries temporelles Darts
        niveau_nappe_df, pe_df, etp_df = imputed_dfs
        
        # Convertir les séries temporelles en objets TimeSeries de Darts
        niveau_nappe_timeseries = []
        pe_timeseries = []
        etp_timeseries = []

        # Convertir niveau_nappe_df
        niveau_nappe_df['DATE'] = niveau_nappe_df.index
        for col in niveau_nappe_df.columns:
            if col != 'DATE':  # On ne prend pas la colonne 'DATE' elle-même comme valeur
                ts = TimeSeries.from_dataframe(niveau_nappe_df, time_col='DATE', value_cols=col)
                niveau_nappe_timeseries.append(ts)

        # Convertir pe_df
        pe_df['DATE'] = pe_df.index
        for col in pe_df.columns:
            if col != 'DATE':  # On ne prend pas la colonne 'DATE' elle-même comme valeur
                ts = TimeSeries.from_dataframe(pe_df, time_col='DATE', value_cols=col)
                pe_timeseries.append(ts)

        # Convertir etp_df
        etp_df['DATE'] = etp_df.index
        for col in etp_df.columns:
            if col != 'DATE':  # On ne prend pas la colonne 'DATE' elle-même comme valeur
                ts = TimeSeries.from_dataframe(etp_df, time_col='DATE', value_cols=col)
                etp_timeseries.append(ts)

        return anomalies_dict,niveau_nappe_timeseries, pe_timeseries, etp_timeseries

### TimeSeries Split for Training and Testing
class SplitTimeSeries(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.15):
        self.test_size = test_size

    def fit(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 4:
            raise ValueError("X doit être un tuple avec 4 éléments : anomalies_dict,niveau_nappe_timeseries, pe_timeseries, etp_timeseries")
        anomalies_dict,niveau_nappe_timeseries, pe_timeseries, etp_timeseries = X
        
        if not all(isinstance(ts_list, list) for ts_list in X):
            raise ValueError("Les éléments de X doivent être des listes de séries temporelles")

        return self

    def transform(self, X, y=None):
        if not isinstance(X, tuple) or len(X) != 4:
            raise ValueError("X doit être un tuple avec 4 éléments : anomalies_dict,niveau_nappe_timeseries, pe_timeseries, etp_timeseries")
        anomalies_dict, niveau_nappe_timeseries, pe_timeseries, etp_timeseries = X
        
        # Fractionner les données pour chaque série
        train_niveau_nappe = []
        test_niveau_nappe = []
        for ts in niveau_nappe_timeseries:
            train_size = int((1 - self.test_size) * len(ts))
            train_ts, val_ts = ts.split_after(train_size)
            train_niveau_nappe.append(train_ts)
            test_niveau_nappe.append(val_ts)

        train_pe = []
        test_pe = []
        for ts in pe_timeseries:
            train_size = int((1 - self.test_size) * len(ts))
            train_ts, val_ts = ts.split_after(train_size)
            train_pe.append(train_ts)
            test_pe.append(val_ts)

        train_etp = []
        test_etp = []
        for ts in etp_timeseries:
            train_size = int((1 - self.test_size) * len(ts))
            train_ts, val_ts = ts.split_after(train_size)
            train_etp.append(train_ts)
            test_etp.append(val_ts)

        return (
            train_niveau_nappe, test_niveau_nappe,
            train_pe, test_pe,
            train_etp, test_etp,
            niveau_nappe_timeseries, pe_timeseries, etp_timeseries
        )

class InterpolatorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='polynomial', order=2):
        self.method = method
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, list):
            raise ValueError("X doit être une liste de DataFrames")

        # Appliquer l'interpolation sur chaque DataFrame
        interpolated_dfs = []
        for df in X:
            interpolated_df = df.interpolate(method=self.method, order=self.order)
            interpolated_dfs.append(interpolated_df)

        return interpolated_dfs


class StationarityTester:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level

    def test_stationarity(self, dfs):
        # Initialize empty lists to store stationary and non-stationary series
        stationary_series = []
        non_stationary_series = []

        # Perform an ADF test on each time series in the list of DataFrames
        for df in dfs:
            print(df)
            for col in df.columns:
                result = adfuller(df[col])
                if result[1] < self.significance_level:
                    # If the p-value is less than the significance level, the series is stationary
                    stationary_series.append(df[col])
                else:
                    # If the p-value is greater than the significance level, the series is non-stationary
                    non_stationary_series.append(df[col])

        return stationary_series, non_stationary_series



class StationarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level

    def fit(self, X, y=None):
        # Aucun apprentissage nécessaire
        return self

    def transform(self, X):
        # X: liste de DataFrames imputés
        stationary = []
        non_stationary = []
        for df in X:
            for col in df.columns:
                pvalue = adfuller(df[col])[1]
                if pvalue < self.significance_level:
                    stationary.append(df[[col]])
                else:
                    non_stationary.append(df[[col]])
        return stationary, non_stationary




