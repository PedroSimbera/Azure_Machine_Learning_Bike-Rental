# Importando bibliotecas
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
import sys
from azureml.core import Run
from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
from azureml.training.tabular._diagnostics import logging_utilities
from azureml.training.tabular.preprocessing import data_cleaning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from lightgbm.sklearn import LGBMRegressor
from azureml.training.tabular.models.voting_ensemble import PreFittedSoftVotingRegressor
import mlflow

# Função para configurar o logger
def setup_instrumentation(automl_run_id):
    logger = logging.getLogger("azureml.training.tabular")
    try:
        logger.setLevel(logging.INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        logger.addHandler(telemetry_handler)

        run = Run.get_context()
        return logging.LoggerAdapter(logger, extra={
            "properties": {
                "codegen_run_id": run.id,
                "automl_run_id": automl_run_id
            }
        })
    except Exception:
        return logger

# Configuração do logger
automl_run_id = 'mslearn-bike_2'
logger = setup_instrumentation(automl_run_id)

# Função para carregar o dataset
def get_training_dataset(file_path):
    logger.info("Carregando o dataset")
    return pd.read_csv(file_path)

# Função para preparar os dados
def prepare_data(dataframe):
    logger.info("Preparando os dados")
    label_column_name = 'rentals'
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    X, y, _ = data_cleaning._remove_nan_rows_in_X_y(X, y, None, is_timeseries=False, target_column=label_column_name)
    return X, y

# Função para dividir o dataset
def split_dataset(X, y, split_ratio=0.1):
    return train_test_split(X, y, test_size=split_ratio, random_state=42)

# Função para construir o pipeline do modelo
def build_model_pipeline():
    logger.info("Construindo o pipeline do modelo")
    pipeline = Pipeline(steps=[
        ('preproc', MaxAbsScaler()),
        ('model', LGBMRegressor())
    ])
    return pipeline

# Função para treinar o modelo
def train_model(X, y):
    logger.info("Treinando o modelo")
    model_pipeline = build_model_pipeline()
    model = model_pipeline.fit(X, y)
    return model

# Função principal
def main(file_path):
    df = get_training_dataset(file_path)
    X, y = prepare_data(df)
    X_train, X_valid, y_train, y_valid = split_dataset(X, y)
    model = train_model(X_train, y_train)

    # Avaliação do modelo
    logger.info("Avaliação do modelo")
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(np.mean((y_pred - y_valid) ** 2))
    logger.info(f"RMSE: {rmse}")

    # Registrando o modelo
    signature = mlflow.models.signature.infer_signature(X, y)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='outputs/',
        signature=signature
    )

    logger.info("Modelo registrado com sucesso")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='C:/TECNOLOGIA/DIO/AI-900/ml bike/daily-bike-share.csv', help='Caminho do dataset')
    args = parser.parse_args()

    try:
        main(args.file_path)
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise
