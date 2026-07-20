import argparse
import concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importación unificada desde el módulo reestructurado
from src.features import extraer_todas_las_caracteristicas
# Importamos las rutas centralizadas
from src.config import METADATA_DIR, INTERMEDIATE_DIR


def worker_extraer(args: tuple) -> dict | None:
    """Procesa un registro individual extrayendo sus 51 características en 1 sola pasada."""
    idx, row, fs = args
    path_npy = Path(row.get('path_npy', ''))
    id_reg = row.get('id_registro', row.get('id_registro_str', f'reg_{idx}'))

    try:
        if not path_npy.exists():
            return None

        # Cargar la señal
        senal = np.load(path_npy)

        # Extracción consolidada de las 51 características
        feats = extraer_todas_las_caracteristicas(senal, fs=fs)

        # Agregar metadatos del registro al inicio
        meta = {
            'id_registro': id_reg,
            'clase': row.get('clase', np.nan),
            'dataset': row.get('dataset', np.nan)
        }
        
        return {**meta, **feats}

    except Exception as e:
        # Silencia o registra errores de archivos corruptos/incompletos
        return None


def procesar_dataset_features(df_input: pd.DataFrame, fs: float = 100.0, num_workers: int = None) -> pd.DataFrame:
    """Recorre el DataFrame y extrae todas las características utilizando paralelismo de procesos."""
    # Mantenemos la lógica de workers paralelos
    if num_workers is None:
        import os
        num_workers = os.cpu_count() or 4

    print(f"[*] Iniciando extracción con {num_workers} procesos en paralelo...")

    # Preparar la lista de tareas para el pool de procesos
    tareas = [(idx, row, fs) for idx, row in df_input.iterrows()]
    resultados_features = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(worker_extraer, tareas), total=len(tareas), desc="Extrayendo features"):
            if res is not None:
                resultados_features.append(res)

    df_features = pd.DataFrame(resultados_features)

    # Asegurar que las columnas de metadatos estén al principio
    cols_meta = ['id_registro', 'clase', 'dataset']
    cols_existentes = [c for c in cols_meta if c in df_features.columns]
    cols_resto = [c for c in df_features.columns if c not in cols_existentes]
    df_features = df_features[cols_existentes + cols_resto]

    print(f"\n[+] Procesamiento finalizado con éxito.")
    print(f"    - Registros procesados: {len(df_features)} / {len(df_input)}")
    print(f"    - Matriz resultante (registros, columnas): {df_features.shape}")

    return df_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracción masiva de características ECG.")
    
    # 🎯 Configuramos los valores por defecto usando nuestro archivo config.py
    default_input = METADATA_DIR / 'etiquetas_maestras_limpias.csv'
    default_output = INTERMEDIATE_DIR / 'df_features.csv'

    # Cambiamos type=str a type=Path para manejar rutas dinámicas fácilmente
    parser.add_argument('--input', type=Path, default=default_input, help='Ruta al CSV de entrada.')
    parser.add_argument('--output', type=Path, default=default_output, help='Ruta al CSV de salida.')
    parser.add_argument('--fs', type=float, default=100.0, help='Frecuencia de muestreo (default: 100 Hz).')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.exists():
        df_in = pd.read_csv(input_path)
        df_out = procesar_dataset_features(df_in, fs=args.fs)
        
        # 🎯 Reemplazamos os.makedirs por pathlib
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)
        print(f"💾 Guardado archivo final en: {output_path}")
    else:
        print(f"❌ Error: El archivo {input_path} no existe.")