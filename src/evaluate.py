import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def evaluate_classification(model_name, y_true, y_pred, y_proba=None):
    """
    Calcula métricas de clasificación binaria.

    Parámetros:
        model_name: nombre del modelo.
        y_true: valores reales.
        y_pred: clases predichas.
        y_proba: probabilidad estimada para la clase positiva.

    Retorna:
        Diccionario con accuracy, precision, recall, f1 y roc_auc.
    """

    result = {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        result["roc_auc"] = np.nan

    return result


def plot_confusion_matrix(y_true, y_pred, output_path, title):
    """
    Genera y guarda una matriz de confusión.
    """

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Sin enfermedad", "Con enfermedad"],
        cmap="Blues"
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_proba, output_path, title):
    """
    Genera y guarda una curva ROC.
    """

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_comparison(results_df, metric, output_path, title):
    """
    Genera gráfico de barras para comparar modelos según una métrica.
    """

    plt.figure(figsize=(8, 5))

    ordered_df = results_df.sort_values(by=metric, ascending=False)

    sns.barplot(
        data=ordered_df,
        x=metric,
        y="modelo"
    )

    plt.title(title)
    plt.xlabel(metric)
    plt.ylabel("Modelo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_results_table(results, output_path):
    """
    Guarda una lista de resultados como CSV.
    """

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    return df_results