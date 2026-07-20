from pathlib import Path
import pandas as pd

# Importamos las carpetas centralizadas desde nuestro config
from src.config import METADATA_DIR


def procesar_etiquetas_ptbxl(csv_path: str | Path) -> pd.DataFrame:
    """Carga el CSV de etiquetas de PTB-XL y colapsa las columnas de

    bloqueos completos/incompletos en LBBB y RBBB.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de PTB-XL en {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df["dataset"] = "PTB-XL"

    # Fusión de bloqueos completos e incompletos (LBBB)
    if "CLBBB" in df.columns or "ILBBB" in df.columns:
        clbbb = (df["CLBBB"].fillna(0) > 0) if "CLBBB" in df.columns else False
        ilbbb = (df["ILBBB"].fillna(0) > 0) if "ILBBB" in df.columns else False
        df["LBBB"] = (clbbb | ilbbb).astype(int)

    # Fusión de bloqueos completos e incompletos (RBBB)
    if "CRBBB" in df.columns or "IRBBB" in df.columns:
        crbbb = (df["CRBBB"].fillna(0) > 0) if "CRBBB" in df.columns else False
        irbbb = (df["IRBBB"].fillna(0) > 0) if "IRBBB" in df.columns else False
        df["RBBB"] = (crbbb | irbbb).astype(int)

    # Eliminar granularidad original para evitar solapamientos
    df = df.drop(columns=["CLBBB", "ILBBB", "CRBBB", "IRBBB"], errors="ignore")
    return df


def cargar_etiquetas_simples(
    csv_path: str | Path, nombre_dataset: str
) -> pd.DataFrame:
    """Carga un CSV de etiquetas estándar (Chapman, CPSC2018, etc.)

    y añade la columna identificadora del dataset.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de {nombre_dataset} en {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df["dataset"] = nombre_dataset
    return df


def generar_tabla_maestra(
    rutas_datasets: dict[str, str | Path],
    output_csv_path: str | Path,
    clases_objetivo: list[str] = ["NORM", "LBBB", "RBBB"],
) -> pd.DataFrame:
    """Unifica los DataFrames de etiquetas de múltiples fuentes, filtra registros

    multietiqueta/sin etiqueta y genera la columna de clasificación binaria/multiclase.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dfs = []

    # 1. Cargar cada dataset según su origen
    for nombre, ruta in rutas_datasets.items():
        if not Path(ruta).exists():
            print(
                f"Advertencia: Omitiendo {nombre}, el archivo no existe en {ruta}"
            )
            continue

        if nombre == "PTB-XL":
            df_curr = procesar_etiquetas_ptbxl(ruta)
        else:
            df_curr = cargar_etiquetas_simples(ruta, nombre_dataset=nombre)

        dfs.append(df_curr)

    if not dfs:
        raise ValueError(
            "No se pudo cargar ningún dataset. Revisa las rutas especificadas."
        )

    # 2. Fusionar y estandarizar
    df_master = pd.concat(dfs, ignore_index=True).fillna(0)

    # Asegurar tipos enteros para las columnas numéricas
    cols_existentes = [c for c in clases_objetivo if c in df_master.columns]
    df_master[cols_existentes] = df_master[cols_existentes].astype(int)

    # Seleccionar únicamente columnas requeridas
    cols_finales = ["id_registro", "dataset"] + cols_existentes
    df_master = df_master[cols_finales]

    # 3. Filtrar ambiguos y registros de otras clases
    n_etiquetas_activas = df_master[cols_existentes].sum(axis=1)

    n_ambiguos = (n_etiquetas_activas > 1).sum()
    n_otros_dx = (n_etiquetas_activas == 0).sum()

    # Nos quedamos solo con los registros mutuamente exclusivos (exactamente 1 etiqueta activa)
    df_master = df_master[n_etiquetas_activas == 1].copy()

    # 4. Asignación vectorizada de la columna categórica 'clase'
    df_master["clase"] = df_master[cols_existentes].idxmax(axis=1)

    # 5. Guardar en disco
    df_master.to_csv(output_csv_path, index=False)

    # 6. Reporte de consolidación
    print("=" * 50)
    print("--- TABLA MAESTRA CREADA CON ÉXITO ---")
    print("=" * 50)
    print(f"Registros ambiguos descartados (>1 clase): {n_ambiguos}")
    print(
        f"Registros con otros diagnósticos descartados (0 clases): {n_otros_dx}"
    )
    print(f"Total de registros válidos combinados: {len(df_master)}")
    print("\nDesglose por Dataset:")
    print(df_master["dataset"].value_counts().to_string())
    print("\nResumen Global por Clase:")
    print(df_master["clase"].value_counts().to_string())
    print("=" * 50)
    print(f"Archivo maestro guardado en: {output_csv_path}\n")

    return df_master


if __name__ == "__main__":
    # 🎯 Apuntamos a la subcarpeta de metadatos usando METADATA_DIR de config.py
    RUTAS_INPUT = {
        "Chapman": METADATA_DIR / "chapman_etiquetas.csv",
        "CPSC2018": METADATA_DIR / "cpsc2018_etiquetas.csv",
        "PTB-XL": METADATA_DIR / "ptbxl_etiquetas.csv",
    }

    RUTA_OUTPUT = METADATA_DIR / "etiquetas_maestras.csv"

    # Generar archivo maestro
    df_maestro = generar_tabla_maestra(
        rutas_datasets=RUTAS_INPUT,
        output_csv_path=RUTA_OUTPUT,
        clases_objetivo=["NORM", "LBBB", "RBBB"],
    )
