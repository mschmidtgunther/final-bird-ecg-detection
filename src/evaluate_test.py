import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

# Importamos las rutas centralizadas
from src.config import MODEL_PATH, TEST_FEAT_PATH


def evaluate_holdout():
    print(f"📂 Cargando set de prueba desde: {TEST_FEAT_PATH}")
    df_test = pd.read_csv(TEST_FEAT_PATH)

    # 🟢 Detectamos si la columna objetivo es 'clase' o 'target'
    target_col = 'clase' if 'clase' in df_test.columns else 'target'

    # SeparamOS características y objetivo limpiando identificadores
    X_test = df_test.drop(
        columns=['id_registro', 'record_id', 'clase', 'target', 'dataset'], 
        errors='ignore'
    )
    y_test = df_test[target_col]

    print(f"🤖 Cargando modelo entrenado desde: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Alineación explícita de características
    if hasattr(model, "feature_names_in_"):
        X_test = X_test[model.feature_names_in_]

    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n" + "=" * 40)
    print("   RESULTADOS FINALES (HOLDOUT TEST)")
    print("=" * 40)
    print(classification_report(y_test, y_pred, digits=4))

    # 🟢 Pasamos 'labels' explícitamente: si el holdout llegara a no contener
    # alguna clase, evita el ValueError de roc_auc_score por descalce entre
    # las columnas de y_proba (todas las clases del modelo) y las clases
    # presentes en y_test.
    print(
        f"AUC-ROC (Macro): "
        f"{roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro', labels=model.classes_):.4f}"
    )


if __name__ == "__main__":
    evaluate_holdout()