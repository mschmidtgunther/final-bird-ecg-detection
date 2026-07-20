import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 🟢 Importamos las rutas centralizadas
from src.config import TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR

def entrenar_evaluar_mlp(data_path: Path, output_dir: Path, reports_dir: Path):
    # Paso 1: Crear directorios de salida si no existen
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Paso 2: Cargar dataset TrainVal
    print(f"📂 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, dtype={'id_registro': str})

    X = df.drop(columns=['id_registro', 'clase', 'dataset'], errors='ignore')
    y_raw = df['clase']

    print(f"📊 Dimensiones de entrada: {X.shape[0]} registros, {X.shape[1]} características iniciales.")

    # Paso 3: Codificar las etiquetas categóricas a numéricas
    le = LabelEncoder()
    y_num = le.fit_transform(y_raw)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"🏷️ Mapeo de clases: {class_mapping}")

    # Paso 4: Definir el Pipeline (Estandarización + Selección k=18 + MLP)
    pipeline_mlp = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("sc", StandardScaler()),
        ("select", SelectKBest(f_classif, k=18)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),  # Arquitectura acotada (2 capas ocultas)
            activation='relu',
            alpha=0.01,                  # Regularización L2
            early_stopping=True,         # Control de overfitting
            max_iter=500,
            random_state=42
        ))
    ])

    # Paso 5: Validación Cruzada Estratificada (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 50)
    print("--- INICIANDO VALIDACIÓN CRUZADA MLP (5-FOLD) ---")
    print("=" * 50)

    scores_f1 = cross_val_score(pipeline_mlp, X, y_num, cv=skf, scoring='f1_macro', n_jobs=-1)
    print(f"F1-Macro por fold: {[round(s, 4) for s in scores_f1]}")
    print(f"🎯 F1-Macro promedio MLP: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")

    # Paso 6: Predicciones Out-Of-Fold y Métricas por Clase
    y_pred_num = cross_val_predict(pipeline_mlp, X, y_num, cv=skf, n_jobs=-1)
    y_pred_text = le.inverse_transform(y_pred_num)

    print("\n--- REPORTE DE CLASIFICACIÓN MLP (OOF) ---")
    print(classification_report(y_raw, y_pred_text))

    # Paso 7: Matriz de Confusión y Guardado de Reporte Gráfico
    labels = ['NORM', 'LBBB', 'RBBB']
    cm = confusion_matrix(y_raw, y_pred_text, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(7, 6))
    disp.plot(cmap='Purples', values_format='d')
    plt.title("Matriz de Confusión - MLP Baseline (CV)")
    
    cm_path = reports_dir / 'cm_mlp.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Matriz de confusión guardada en: {cm_path}")

    # Paso 8: Entrenar el modelo final sobre todo TrainVal y guardar artefacto (.joblib)
    print("\n💾 Entrenando modelo MLP final sobre todo TrainVal...")
    pipeline_mlp.fit(X, y_num)
    
    model_path = output_dir / 'mlp_baseline.joblib'
    joblib.dump({
        'pipeline': pipeline_mlp,
        'label_encoder': le
    }, model_path)
    print(f"✅ Modelo y LabelEncoder guardados exitosamente en: {model_path}\n")


if __name__ == '__main__':
    # Usamos las rutas directas desde nuestro archivo de configuración
    if TRAIN_FEAT_PATH.exists():
        entrenar_evaluar_mlp(TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR)
    else:
        print(f"❌ Error: No se encuentra el archivo {TRAIN_FEAT_PATH}")