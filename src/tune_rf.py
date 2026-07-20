from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# 🟢 Importamos las rutas centralizadas desde nuestro config
from src.config import TRAIN_FEAT_PATH, MODELS_DIR, MODEL_PATH


def optimizar_hiperparametros_conservador(
    data_path: Path = TRAIN_FEAT_PATH, 
    output_dir: Path = MODELS_DIR
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar conjunto TrainVal
    print(f"📂 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path, dtype={'id_registro': str})

    # Detectamos la columna objetivo dinámicamente ('clase' o 'target')
    target_col = 'clase' if 'clase' in df.columns else 'target'
    
    X = df.drop(
        columns=['id_registro', 'record_id', 'clase', 'target', 'dataset'], 
        errors='ignore'
    )
    y = df[target_col]

    print(f"📊 Dataset cargado: {X.shape[0]} registros, {X.shape[1]} características iniciales.")

    # 2. Definir Pipeline base
    pipeline_rf = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("sc", StandardScaler()),
        ("select", SelectKBest(f_classif)),
        ("clf", RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 3. Grilla Conservadora (evitando overfitting: max_depth acotado, min_samples_leaf >= 3)
    param_grid_conservador = {
        'select__k': [15, 18, 20],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [6, 8, 10, 12],        # Sin opción None
        'clf__min_samples_leaf': [3, 5, 10]      # Sin opción 1
    }

    # 4. Validación Cruzada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid_conservador,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    print("\n" + "=" * 55)
    print("--- INICIANDO BÚSQUEDA EN GRILLA (RF CONSERVADOR) ---")
    print("=" * 55)

    grid_search.fit(X, y)

    print("\n" + "=" * 55)
    print("🏆 ¡OPTIMIZACIÓN CONSERVADORA FINALIZADA!")
    print("=" * 55)
    print(f"🎯 Mejor F1-Macro CV: {grid_search.best_score_:.4f}")
    print("\n🔧 Mejores hiperparámetros elegidos:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")

    # 5. Guardar modelo optimizado usando la ruta centralizada
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    print(f"\n✅ Modelo optimizado conservador guardado en: {MODEL_PATH}\n")


if __name__ == '__main__':
    if TRAIN_FEAT_PATH.exists():
        optimizar_hiperparametros_conservador()
    else:
        print(f"❌ Error: No se encuentra el archivo en {TRAIN_FEAT_PATH}")