from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Importamos las carpetas centralizadas
from src.config import INTERMEDIATE_DIR, PROCESSED_DIR


def balancear_dataset(
    df: pd.DataFrame,
    conteo_objetivo: dict[str, int | None],
    random_state: int = 42,
) -> pd.DataFrame:
    """Aplica un undersampling parcial por clase según las cantidades indicadas en conteo_objetivo.

    Si una clase mapea a None, se conserva intacta sin submuestreo.
    """
    dfs_submuestreados = []

    print("\n" + "=" * 50)
    print("--- APLICANDO UNDERSAMPLING PARCIAL ---")
    print("=" * 50)
    print("Distribución original:")
    print(df["clase"].value_counts().to_string())

    for clase, n_samples in conteo_objetivo.items():
        df_clase = df[df["clase"] == clase]

        if n_samples is not None and len(df_clase) > n_samples:
            df_sub = resample(
                df_clase,
                replace=False,
                n_samples=n_samples,
                random_state=random_state,
            )
        else:
            df_sub = df_clase.copy()

        dfs_submuestreados.append(df_sub)

    df_balanceado = pd.concat(dfs_submuestreados, ignore_index=True)
    df_balanceado = df_balanceado.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    print("\nDistribución posterior al balanceo:")
    print(df_balanceado["clase"].value_counts().to_string())
    print("=" * 50 + "\n")

    return df_balanceado


def crear_splits_train_test(
    df: pd.DataFrame,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en conjuntos Train+Val y Test (Holdout) de forma estratificada

    y guarda los archivos CSV en el directorio destino.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["clase"])
    y = df["clase"]

    # División estratificada respetando las proporciones NORM:LBBB:RBBB
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Recomponer DataFrames completos
    df_trainval = X_trainval.copy()
    df_trainval["clase"] = y_trainval

    df_test = X_test.copy()
    df_test["clase"] = y_test

    # Rutas de guardado
    path_trainval = output_dir / "features_trainval.csv"
    path_test = output_dir / "features_test.csv"

    df_trainval.to_csv(path_trainval, index=False)
    df_test.to_csv(path_test, index=False)

    print("=" * 50)
    print("--- DIVISION STRATIFIED TRAIN / TEST COMPLETADA ---")
    print("=" * 50)
    print(
        f"Conjunto Train+Val ({int((1-test_size)*100)}%): {len(df_trainval)} registros"
    )
    print(df_trainval["clase"].value_counts().to_string())
    print(
        f"\nConjunto Test Holdout ({int(test_size*100)}%): {len(df_test)} registros"
    )
    print(df_test["clase"].value_counts().to_string())
    print("=" * 50)
    print(
        f"Archivos guardados exitosamente en:\n - {path_trainval}\n - {path_test}\n"
    )

    return df_trainval, df_test


if __name__ == "__main__":
    # 🎯 Usamos las constantes centralizadas
    RUTA_ENTRADA = INTERMEDIATE_DIR / "df_features.csv"
    RUTA_BALANCEADO_OUT = INTERMEDIATE_DIR / "df_features_balanceado.csv"
    DIR_SALIDA = PROCESSED_DIR

    # Metas de undersampling
    CONTEO_OBJETIVO = {
        "NORM": 2800,
        "RBBB": 2200,
        "LBBB": None,  # Se conservan todos los registros (~1041)
    }

    if RUTA_ENTRADA.exists():
        df_master = pd.read_csv(RUTA_ENTRADA)

        # Paso 1: Balanceo parcial
        df_bal = balancear_dataset(
            df_master, conteo_objetivo=CONTEO_OBJETIVO, random_state=42
        )
        df_bal.to_csv(RUTA_BALANCEADO_OUT, index=False)

        # Paso 2: Partición Stratified Train+Val (80%) y Test Holdout (20%)
        df_trainval, df_test = crear_splits_train_test(
            df=df_bal, output_dir=DIR_SALIDA, test_size=0.2, random_state=42
        )
    else:
        print(
            f"❌ Error: No se encontró el archivo de características en {RUTA_ENTRADA}"
        )
        print(
            "💡 Recuerda ejecutar 'python -m src.extract_features' primero para generar la matriz final."
        )
        