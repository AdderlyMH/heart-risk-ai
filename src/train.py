import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .data_loader import (
    PROJECT_ROOT,
    load_cleveland,
    load_external_uci_datasets,
    save_processed
)

from .preprocessing import build_preprocessor
from .evaluate import (
    evaluate_classification,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metric_comparison,
    save_results_table
)


def baseline_rule(X):
    """
    Regla heurística simple para comparar contra modelos de ML.

    La regla suma puntos de riesgo según variables clínicas básicas:
        - Edad >= 55
        - Presión arterial >= 140
        - Colesterol >= 240
        - Frecuencia cardíaca máxima < 130
        - Angina inducida por ejercicio

    Si el puntaje es >= 2, predice presencia de enfermedad.
    """

    score = (
        (X["age"] >= 55).astype(int)
        + (X["trestbps"] >= 140).astype(int)
        + (X["chol"] >= 240).astype(int)
        + (X["thalach"] < 130).astype(int)
        + (X["exang"].fillna(0) == 1).astype(int)
    )

    return (score >= 2).astype(int)


def build_models():
    """
    Define los modelos que serán evaluados.
    """

    models = {
        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42,
            max_depth=4
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=200
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    return models


def main():
    # Carpetas de salida
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    results_dir = PROJECT_ROOT / "results"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Cargar dataset principal
    # =========================

    df = load_cleveland()

    # La columna source sirve para identificar origen, pero no se entrena con ella
    df_model = df.drop(columns=["source"])

    # Guardar versión procesada básica
    save_processed(df_model, "cleveland_clean.csv")

    X = df_model.drop(columns=["target"])
    y = df_model["target"]

    # =========================
    # 2. División train/test
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # =========================
    # 3. Experimento 1:
    #    regla base
    # =========================

    test_results = []

    y_pred_base = baseline_rule(X_test)

    test_results.append(
        evaluate_classification(
            "Regla base",
            y_test,
            y_pred_base
        )
    )

    # =========================
    # 4. Experimento 2:
    #    comparación de modelos
    # =========================

    preprocessor = build_preprocessor()
    models = build_models()

    trained_models = {}

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        test_results.append(
            evaluate_classification(
                model_name,
                y_test,
                y_pred,
                y_proba
            )
        )

        trained_models[model_name] = pipeline

    df_test_results = save_results_table(
        test_results,
        tables_dir / "resultados_test.csv"
    )

    # Gráfico comparativo por recall
    plot_metric_comparison(
        df_test_results,
        metric="recall",
        output_path=figures_dir / "comparacion_recall.png",
        title="Comparación de modelos según recall"
    )

    # =========================
    # 5. Experimento 3:
    #    validación cruzada
    # =========================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    cv_results = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        cv_results.append({
            "modelo": model_name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "precision_mean": scores["test_precision"].mean(),
            "precision_std": scores["test_precision"].std(),
            "recall_mean": scores["test_recall"].mean(),
            "recall_std": scores["test_recall"].std(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std()
        })

    df_cv_results = pd.DataFrame(cv_results)
    df_cv_results.to_csv(
        tables_dir / "resultados_validacion_cruzada.csv",
        index=False
    )

    # =========================
    # 6. Experimento 4:
    #    ajuste de hiperparámetros
    # =========================

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid_rf = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7, None],
        "model__min_samples_split": [2, 5, 10],
        "model__class_weight": [None, "balanced"]
    }

    grid_rf = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid_rf,
        cv=cv,
        scoring="recall",
        n_jobs=-1
    )

    # Entrenar búsqueda de hiperparámetros para Random Forest
    grid_rf.fit(X_train, y_train)

    # Guardamos Random Forest optimizado como modelo auxiliar de interpretabilidad,
    # no como modelo final predictivo.
    rf_interpretability_model = grid_rf.best_estimator_

    # =========================
    # Modelo final predictivo:
    # KNN
    # =========================

    # final_model = grid_rf.best_estimator_
    final_model = trained_models["KNN"]

    y_pred_final = final_model.predict(X_test)
    y_proba_final = final_model.predict_proba(X_test)[:, 1]

    final_result = evaluate_classification(
        "KNN",
        y_test,
        y_pred_final,
        y_proba_final
    )

    pd.DataFrame([final_result]).to_csv(
        tables_dir / "resultado_modelo_final.csv",
        index=False
    )

    # Guardar modelo final predictivo: KNN
    joblib.dump(
        final_model,
        results_dir / "modelo_final_heart_risk.pkl"
    )

    # Guardar modelo auxiliar de interpretabilidad: Random Forest optimizado
    joblib.dump(
        rf_interpretability_model,
        results_dir / "modelo_interpretabilidad_random_forest.pkl"
    )

    # Guardar matriz de confusión
    plot_confusion_matrix(
        y_test,
        y_pred_final,
        output_path=figures_dir / "matriz_confusion_modelo_final.png",
        title="Matriz de confusión - Modelo final KNN"
    )

    # Guardar curva ROC
    plot_roc_curve(
        y_test,
        y_proba_final,
        output_path=figures_dir / "curva_roc_modelo_final.png",
        title="Curva ROC - Modelo final KNN"
    )

    # =========================
    # 7. Importancia de variables
    # =========================

    preprocessor_fitted = rf_interpretability_model.named_steps["preprocessor"]
    rf_model = rf_interpretability_model.named_steps["model"]

    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = rf_model.feature_importances_

    df_importances = pd.DataFrame({
        "variable": feature_names,
        "importancia": importances
    }).sort_values(by="importancia", ascending=False)

    df_importances.to_csv(
        tables_dir / "importancia_variables.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df_importances.head(15),
        x="importancia",
        y="variable"
    )

    plt.title("Top 15 variables más importantes")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "importancia_variables.png",
        dpi=300
    )
    plt.close()

    # =========================
    # 8. Experimento opcional:
    #    evaluación externa
    # =========================

    try:
        df_external = load_external_uci_datasets()
        save_processed(df_external, "uci_external_clean.csv")

        external_results = []

        for source_name, df_source in df_external.groupby("source"):
            X_ext = df_source.drop(columns=["target", "source"])
            y_ext = df_source["target"]

            y_pred_ext = final_model.predict(X_ext)
            y_proba_ext = final_model.predict_proba(X_ext)[:, 1]

            external_results.append(
                evaluate_classification(
                    f"Evaluación externa - {source_name}",
                    y_ext,
                    y_pred_ext,
                    y_proba_ext
                )
            )

        pd.DataFrame(external_results).to_csv(
            tables_dir / "resultados_evaluacion_externa.csv",
            index=False
        )

    except FileNotFoundError as error:
        print("No se ejecutó evaluación externa.")
        print(error)

    # =========================
    # 9. Resumen en consola
    # =========================

    print("\nResultados en test:")
    print(df_test_results)

    print("\nResultados de validación cruzada:")
    print(df_cv_results)

    print("\nMejores parámetros Random Forest auxiliar:")
    print(grid_rf.best_params_)

    print("\nResultado modelo final predictivo KNN:")
    print(final_result)

    print("\nArchivos guardados en results/.")


if __name__ == "__main__":
    main()