from pathlib import Path

# 1. Ruta raíz del repositorio
ROOT_DIR = Path(__file__).resolve().parent.parent

# 2. Rutas de carpetas principales
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 3. Subcarpetas organizadas
METADATA_DIR = PROCESSED_DIR / "metadata"
SIGNALS_DIR = PROCESSED_DIR / "signals"
INTERMEDIATE_DIR = PROCESSED_DIR / "intermediate"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 4. Archivos clave para el entrenamiento y evaluación
TRAIN_FEAT_PATH = PROCESSED_DIR / "features_trainval.csv"
TEST_FEAT_PATH = PROCESSED_DIR / "features_test.csv"
MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"