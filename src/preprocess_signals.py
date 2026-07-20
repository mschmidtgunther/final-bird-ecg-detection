import os
import wfdb
import numpy as np
import pandas as pd
import concurrent.futures
from pathlib import Path
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

from src.config import RAW_DIR, SIGNALS_DIR, METADATA_DIR

# =============================================================================
# 1. FUNCIONES CORE (Filtro y Control de Calidad)
# =============================================================================
def aplicar_filtro_bandpass(
    senal_multicanal: np.ndarray, 
    fs: float, 
    lowcut: float = 0.5, 
    highcut: float = 40.0, 
    orden: int = 4
) -> np.ndarray:
    """Aplica un filtro pasa-banda Butterworth a lo largo del eje temporal."""
    nyquist = 0.5 * fs
    b, a = butter(orden, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, senal_multicanal, axis=0)


def tiene_linea_plana(
    senal_4_canales: np.ndarray, 
    fs: float, 
    ventana_segundos: float = 1.5, 
    tol_std: float = 1e-4
) -> bool:
    """Detecta tramos planos (desconexión de electrodos) si la desviación estándar cae por debajo de la tolerancia."""
    largo_ventana = int(ventana_segundos * fs)
    n_muestras = len(senal_4_canales)
    
    if n_muestras < largo_ventana:
        return True  # Señal demasiado corta

    paso = largo_ventana // 2
    for i in range(0, n_muestras - largo_ventana + 1, paso):
        if np.any(np.std(senal_4_canales[i:i + largo_ventana, :], axis=0) < tol_std):
            return True
    return False


# =============================================================================
# 2. WORKER DE PROCESAMIENTO POR REGISTRO
# =============================================================================
def procesar_paciente(args: tuple) -> tuple[str, str, str | None]:
    """
    Extrae DI, DII, V1, V6, aplica resampleo a 100Hz, evalúa QC y guarda el array limpio en .npy
    """
    id_registro, dataset, ruta_archivo, ruta_destino = args
    archivo_salida = ruta_destino / f"{id_registro}_filtrado.npy"

    if archivo_salida.exists():
        return (id_registro, 'ok', None)

    try:
        if ruta_archivo is None:
            return (id_registro, 'error', 'Archivo .hea no encontrado en disco')

        ruta_base = str(ruta_archivo.with_suffix(''))
        registro = wfdb.rdrecord(ruta_base)

        fs_original = registro.fs
        fs_objetivo = 100

        # Mapear nombres de derivaciones
        nombres_canales = [n.strip().upper() for n in registro.sig_name]
        idx_di = next((i for i, n in enumerate(nombres_canales) if n in ['I', 'DI']), None)
        idx_dii = next((i for i, n in enumerate(nombres_canales) if n in ['II', 'DII']), None)
        idx_v1 = next((i for i, n in enumerate(nombres_canales) if n in ['V1']), None)
        idx_v6 = next((i for i, n in enumerate(nombres_canales) if n in ['V6']), None)

        if None in [idx_di, idx_dii, idx_v1, idx_v6]:
            return (id_registro, 'error', 'Faltan derivaciones críticas (DI, DII, V1 o V6)')

        # Orden estricto: [DI, DII, V1, V6]
        senal_4_canales = registro.p_signal[:, [idx_di, idx_dii, idx_v1, idx_v6]]

        # Resampleo temporal si difiere de 100 Hz
        if fs_original != fs_objetivo:
            num_muestras_nuevo = int(senal_4_canales.shape[0] * (fs_objetivo / fs_original))
            senal_4_canales = resample(senal_4_canales, num_muestras_nuevo, axis=0)

        # Control de Calidad
        if tiene_linea_plana(senal_4_canales, fs=fs_objetivo):
            if archivo_salida.exists():
                archivo_salida.unlink()
            return (id_registro, 'qc_fail', 'Línea plana detectada')

        # Filtrado Pasa-Banda (0.5 - 40 Hz) y exportación a float32
        senal_limpia = aplicar_filtro_bandpass(senal_4_canales, fs_objetivo)
        np.save(archivo_salida, senal_limpia.astype(np.float32))

        return (id_registro, 'ok', None)

    except Exception as e:
        return (id_registro, 'error', str(e))


# =============================================================================
# 3. CONTROLADOR PRINCIPAL
# =============================================================================
def ejecutar_preprocesamiento_masivo(
    ruta_raw: Path,
    ruta_destino: Path,
    ruta_csv_master: Path,
    ruta_csv_limpio: Path
) -> None:
    ruta_destino.mkdir(parents=True, exist_ok=True)

    # 1. Indexación rápida de cabeceras .hea
    print("Creando índice de archivos .hea en disco...")
    todos_archivos = list(ruta_raw.rglob('*.hea'))
    mapa_archivos = {}

    for f in todos_archivos:
        nombre_base = f.stem
        if '_lr' in nombre_base:
            nombre_normalizado = nombre_base.replace('_lr', '')
        elif '_hr' in nombre_base:
            nombre_normalizado = nombre_base.replace('_hr', '')
        else:
            nombre_normalizado = nombre_base

        if nombre_normalizado not in mapa_archivos or '_lr' in f.name:
            mapa_archivos[nombre_normalizado] = f

    # 2. Cargar Master CSV
    if not ruta_csv_master.exists():
        raise FileNotFoundError(f"No se encontró el archivo maestro en {ruta_csv_master}")

    df_master = pd.read_csv(ruta_csv_master)
    total_original = len(df_master)

    # Filtrar solo clases válidas
    df_master = df_master[
        (df_master['LBBB'] == 1) | 
        (df_master['RBBB'] == 1) | 
        (df_master['NORM'] == 1)
    ].copy()

    print("\n--- FILTRO DE PATOLOGÍAS ---")
    print(f"Pacientes totales en CSV original: {total_original}")
    print(f"Pacientes a procesar (NORM, LBBB, RBBB): {len(df_master)}")

    df_master['id_registro_str'] = df_master['id_registro'].astype(str).str.replace('.0', '', regex=False)

    # 3. Preparación de tareas
    tareas = []
    for _, row in df_master.iterrows():
        id_str = row['id_registro_str']
        dataset = row['dataset']
        clave_busqueda = id_str.zfill(5) if dataset == 'PTB-XL' else id_str
        ruta_archivo = mapa_archivos.get(clave_busqueda)

        tareas.append((id_str, dataset, ruta_archivo, ruta_destino))

    # 4. Paralelización
    nucleos = os.cpu_count() or 4
    print(f"\nIniciando multi-hilos con {nucleos} workers...")

    resultados = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=nucleos) as executor:
        for res in tqdm(executor.map(procesar_paciente, tareas), total=len(tareas), desc="Filtrando"):
            resultados.append(res)

    # 5. Generación del CSV Limpio final
    ids_validos = [res[0] for res in resultados if res[1] == 'ok']
    descartados_qc = sum(1 for res in resultados if res[1] == 'qc_fail')
    errores = sum(1 for res in resultados if res[1] == 'error')

    df_master_limpio = df_master[df_master['id_registro_str'].isin(ids_validos)].copy()
    # 2. Creamos la columna 'path_npy' apuntando a los archivos filtrados
    # (Nota: usa 'ruta_destino' o la carpeta donde guardaste los .npy)
    df_master_limpio['path_npy'] = df_master_limpio['id_registro_str'].apply(
    lambda id_reg: str(ruta_destino / f"{id_reg}_filtrado.npy"))

    # 3. Guardamos el CSV final (sin borrar id_registro_str)
    df_master_limpio.to_csv(ruta_csv_limpio, index=False)

    print("\n" + "=" * 50)
    print("--- PREPROCESAMIENTO MASIVO FINALIZADO ---")
    print("=" * 50)
    print(f"✓ Señales guardadas (.npy): {len(ids_validos)}")
    print(f"🚨 Descartadas por QC (línea plana): {descartados_qc}")
    print(f"❌ Errores técnicos / Faltantes: {errores}")
    print(f"💾 Tabla maestra limpia: {ruta_csv_limpio}\n")


if __name__ == '__main__':
    # 🎯 Rutas dinámicas desde src.config
    CSV_MASTER = METADATA_DIR / 'etiquetas_maestras.csv'
    CSV_LIMPIO_OUT = METADATA_DIR / 'etiquetas_maestras_limpias.csv'

    ejecutar_preprocesamiento_masivo(
        ruta_raw=RAW_DIR,
        ruta_destino=SIGNALS_DIR,  # <--- Ahora guardará en data/processed/signals/
        ruta_csv_master=CSV_MASTER,
        ruta_csv_limpio=CSV_LIMPIO_OUT
    )