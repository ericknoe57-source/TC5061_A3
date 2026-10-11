import argparse
import os
import tempfile

# Importa los datos en el notebook
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Análisis EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Entrenamiento del modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Evaluación del modelo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

# Registrar el experimento en MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Cargar los datos
def load_data():
    # Dataset: https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits

    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "samartalwar/sleep-debt-and-screen-time-late-night-phone-habits",
        "bedtime_screentime_sleep_debt.csv"
    )

# Preprocesamiento de los datos
def preprocess_data(df):
    # Eliminar columnas innecesarias
    df = df.drop(columns=["user_id"])

    # Tratamiento de variables categóricas
    categorical_columns = [
        "primary_bedtime_app",
        "chronotype",
        "occupation_type",
        "gender"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )
    
    return df

# Dividir los datos en conjuntos de entrenamiento y prueba
def division_datos(df, test_size=0.20, random_state=1):
    X = df.drop(columns=["sleep_debt_category"])
    y = df["sleep_debt_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

# Entrenar el modelo
def entrenar_modelo(X_train, y_train, max_iter=1000, C=1.0):
    modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                solver="lbfgs",
                max_iter=max_iter,
                C=C
            ))
        ])
    
    modelo.fit(X_train, y_train)

    return modelo

# Evaluación del modelo
def evaluar_modelo(modelo, X_test, y_test):
    # Evaluación del modelo
    y_pred = modelo.predict(X_test)
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    return y_pred

def mlflow_logging(modelo, X_train, y_train, y_test, y_pred, args=None):

    mlflow.set_tracking_uri(uri="http://localhost:5000")
    mlflow.set_experiment("Clasificacion_Estado_Sueno")

    with mlflow.start_run():
        # Registrar hiperparámetros
        mlflow.log_params({
            "max_iter": args.max_iter,
            "C": args.C,
            "test_size": args.test_size,
            "seed": args.seed,
            "solver": "lbfgs",
            "modelo": "LogisticRegression"
        })

        # Registrar métricas
        mlflow.log_metric("Accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average="weighted"))

        # Registrar la matriz de confusión como artefacto
        with tempfile.TemporaryDirectory() as temp_dir:

            # Matriz de confusión
            fig, ax = plt.subplots(figsize=(9, 7))

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                xticks_rotation=45,
                cmap="Blues",
                ax=ax
            )

            ax.set_title("Matriz de confusión")
            fig.tight_layout()

            confusion_path = os.path.join(
                temp_dir, "matriz_confusion.png"
            )
            fig.savefig(confusion_path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(confusion_path)

        # Registrar el modelo
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo_clasificacion_estado_sueno",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name="Clasificacion_Estado_Sueno"
        )



def main():
    # Definir los argumentos de ejecución
    parser = argparse.ArgumentParser(
        description="Entrenamiento de un modelo de clasificación"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Proporción de datos destinada a pruebas"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Semilla aleatoria para reproducibilidad"
    )

    args = parser.parse_args()

    # Cargar los datos
    df = load_data()

    # Preprocesamiento de los datos
    df = preprocess_data(df)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = division_datos(df, test_size=args.test_size, random_state=args.seed)

    # Entrenar el modelo
    modelo = entrenar_modelo(X_train, y_train, max_iter=args.max_iter, C=args.C)

    # Evaluación del modelo
    y_pred = evaluar_modelo(modelo, X_test, y_test)

    # Registrar el experimento en MLflow
    mlflow_logging(modelo, X_train, y_train, y_test, y_pred, args=args)

if __name__ == "__main__":
    main()