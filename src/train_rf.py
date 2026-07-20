import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 🟢 Importamos las rutas centralizadas
from src.config import TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR

def entrenar_evaluar_rf(data_path: Path, output_dir: Path, reports_dir: Path):
    # 1. Crear directorios de salida si no existen
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Cargar datos
    print(f"📂 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, dtype={'id_registro': str})

    # Separar características (X) y etiqueta (y)
    X = df.drop(columns=['id_registro', 'clase', 'dataset'], errors='ignore')
    y = df['clase']

    print(f"📊 Dimensiones de entrada: {X.shape[0]} registros, {X.shape[1]} características iniciales.")

    # 3. Definir Pipeline
    pipeline_rf = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("sc", StandardScaler()),
        ("select", SelectKBest(f_classif, k=18)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4. Validación Cruzada Estratificada (5 Folds)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 50)
    print("--- INICIANDO VALIDACIÓN CRUZADA (5-FOLD) ---")
    print("=" * 50)

    scores_f1 = cross_val_score(pipeline_rf, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    print(f"F1-Macro por fold: {[round(s, 4) for s in scores_f1]}")
    print(f"🎯 F1-Macro promedio: {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")

    # 5. Predicciones fuera de bolsa (Out-Of-Fold)
    y_pred_cv = cross_val_predict(pipeline_rf, X, y, cv=skf, n_jobs=-1)

    print("\n--- REPORTE DE CLASIFICACIÓN (OOF) ---")
    print(classification_report(y, y_pred_cv))

    # 6. Matriz de Confusión y guardado de gráfico
    labels = ['NORM', 'LBBB', 'RBBB']
    cm = confusion_matrix(y, y_pred_cv, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(7, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Matriz de Confusión - Random Forest (Baseline CV)")
    
    cm_path = reports_dir / 'cm_random_forest.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Matriz de confusión guardada en: {cm_path}")

    # 7. Re-entrenar modelo final con todo trainval y guardar artefacto
    print("\n💾 Entrenando modelo final sobre todo TrainVal...")
    pipeline_rf.fit(X, y)
    
    model_path = output_dir / 'random_forest_baseline.joblib'
    joblib.dump(pipeline_rf, model_path)
    print(f"✅ Modelo guardado exitosamente en: {model_path}\n")


if __name__ == '__main__':
    # Usamos las rutas directas desde nuestro archivo de configuración
    if TRAIN_FEAT_PATH.exists():
        entrenar_evaluar_rf(TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR)
    else:
        print(f"❌ Error: No se encuentra el archivo {TRAIN_FEAT_PATH}")