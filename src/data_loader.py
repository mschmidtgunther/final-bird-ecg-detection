"""
data_loader.py — Carga y normalización de registros individuales para la GUI.

A diferencia de preprocess_signals.py (que corre en batch sobre todo el
dataset y siempre persiste el resultado a disco), este módulo expone una
única función de entrada, `cargar_registro`, que acepta las 3 formas de
entrada que soporta la interfaz:

  - 'hea': archivo crudo WFDB (.hea/.dat), típicamente de PTB-XL/CPSC2018/
           Chapman-Shaoxing tal como vienen de PhysioNet.
  - 'npy': archivo ya preprocesado por el pipeline batch (formato
           [muestras, 4] = [DI, DII, V1, V6] a 100 Hz).
  - 'id':  un id_registro ya presente en el catálogo
           etiquetas_maestras_limpias.csv, del cual se busca el path_npy.

Todas devuelven el mismo contrato: un array [muestras, 4] en el orden
[DI, DII, V1, V6] a 100 Hz, filtrado, listo para pasar directo a
segmentar_qrs_registro / extraer_todas_las_caracteristicas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

from src.config import METADATA_DIR
from src.preprocess_signals import aplicar_filtro_bandpass, tiene_linea_plana

FS_OBJETIVO = 100.0


class RegistroInvalido(Exception):
    """Se lanza cuando un registro no puede cargarse o no pasa control de calidad."""


def cargar_desde_hea(ruta_hea: str | Path) -> np.ndarray:
    """Lee un archivo crudo .hea/.dat (WFDB), mapea DI/DII/V1/V6 por nombre
    de derivación (no por posición), remuestrea a 100 Hz si hace falta y
    aplica el filtro pasa-banda 0.5-40 Hz. Misma lógica que
    preprocess_signals.procesar_paciente, pero sin persistir a disco.
    """
    ruta_hea = Path(ruta_hea)
    if not ruta_hea.exists():
        raise RegistroInvalido(f"No se encontró el archivo: {ruta_hea}")

    ruta_base = str(ruta_hea.with_suffix(""))
    try:
        registro = wfdb.rdrecord(ruta_base)
    except Exception as e:
        raise RegistroInvalido(f"No se pudo leer el registro WFDB: {e}") from e

    nombres_canales = [n.strip().upper() for n in registro.sig_name]
    idx_di = next((i for i, n in enumerate(nombres_canales) if n in ["I", "DI"]), None)
    idx_dii = next((i for i, n in enumerate(nombres_canales) if n in ["II", "DII"]), None)
    idx_v1 = next((i for i, n in enumerate(nombres_canales) if n in ["V1"]), None)
    idx_v6 = next((i for i, n in enumerate(nombres_canales) if n in ["V6"]), None)

    if None in (idx_di, idx_dii, idx_v1, idx_v6):
        raise RegistroInvalido(
            "Faltan derivaciones críticas (DI, DII, V1 o V6) en el archivo."
        )

    senal = registro.p_signal[:, [idx_di, idx_dii, idx_v1, idx_v6]]

    if registro.fs != FS_OBJETIVO:
        n_nuevo = int(senal.shape[0] * (FS_OBJETIVO / registro.fs))
        senal = resample(senal, n_nuevo, axis=0)

    if tiene_linea_plana(senal, fs=FS_OBJETIVO):
        raise RegistroInvalido(
            "Se detectó línea plana (posible desconexión de electrodo) en el registro."
        )

    return aplicar_filtro_bandpass(senal, FS_OBJETIVO).astype(np.float32)


def cargar_desde_npy(ruta_npy: str | Path) -> np.ndarray:
    """Carga un .npy ya preprocesado (formato [muestras, 4] = [DI, DII, V1, V6])."""
    ruta_npy = Path(ruta_npy)
    if not ruta_npy.exists():
        raise RegistroInvalido(f"No se encontró el archivo: {ruta_npy}")

    senal = np.load(ruta_npy)
    if senal.ndim != 2 or senal.shape[1] != 4:
        raise RegistroInvalido(
            f"Formato inesperado en {ruta_npy}: se esperaba [muestras, 4], "
            f"se recibió {senal.shape}"
        )
    return senal


def cargar_catalogo() -> pd.DataFrame:
    """Carga el catálogo de etiquetas maestras limpias (id_registro -> path_npy)."""
    ruta_csv = METADATA_DIR / "etiquetas_maestras_limpias.csv"
    if not ruta_csv.exists():
        raise RegistroInvalido(f"No se encontró el catálogo en {ruta_csv}")
    return pd.read_csv(ruta_csv, dtype={"id_registro_str": str})


def cargar_desde_id(
    id_registro: str, catalogo: pd.DataFrame | None = None
) -> tuple[np.ndarray, dict]:
    """Busca un id_registro en el catálogo y carga su .npy correspondiente.
    Devuelve la señal y sus metadatos (clase real y dataset de origen, si existen)."""
    if catalogo is None:
        catalogo = cargar_catalogo()

    id_registro = str(id_registro).strip()
    fila = catalogo[catalogo["id_registro_str"] == id_registro]
    if fila.empty:
        raise RegistroInvalido(f"El ID '{id_registro}' no está en el catálogo.")

    fila = fila.iloc[0]
    senal = cargar_desde_npy(fila["path_npy"])
    meta = {
        "clase_real": fila["clase"] if "clase" in fila and pd.notna(fila["clase"]) else None,
        "dataset": fila["dataset"] if "dataset" in fila and pd.notna(fila["dataset"]) else None,
    }
    return senal, meta


def cargar_registro(
    entrada: str | Path, tipo: str, catalogo: pd.DataFrame | None = None
) -> tuple[np.ndarray, dict]:
    """Punto de entrada único para la GUI.

    Parameters
    ----------
    entrada : ruta al archivo (.hea/.npy) o id_registro (str), según `tipo`.
    tipo : uno de {'hea', 'npy', 'id'}.
    catalogo : DataFrame de etiquetas_maestras_limpias.csv precargado
        (opcional, para no releerlo en cada llamada dentro de un batch).

    Returns
    -------
    (senal, meta) : señal [muestras, 4] y dict con 'clase_real'/'dataset'
        (None si no aplica, p. ej. para 'hea'/'npy' sueltos sin catálogo).
    """
    if tipo == "hea":
        return cargar_desde_hea(entrada), {"clase_real": None, "dataset": None}
    elif tipo == "npy":
        return cargar_desde_npy(entrada), {"clase_real": None, "dataset": None}
    elif tipo == "id":
        return cargar_desde_id(entrada, catalogo)
    else:
        raise ValueError(f"Tipo de entrada desconocido: '{tipo}' (use 'hea', 'npy' o 'id')")
