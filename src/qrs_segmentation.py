import numpy as np
import pandas as pd
from pathlib import Path 
from tqdm import tqdm
from src.pan_tompkins import bandpass_filter, detect_qrs_multilead


def limites_por_amplitud(
    filtrado: np.ndarray, 
    picos_r: np.ndarray, 
    fs: float, 
    nivel_relativo: float = 0.05, 
    ventana_max_ms: float = 200.0
) -> list[tuple[int, int, float, float]]:
    """
    Delimita el inicio y fin del QRS midiendo la caída de amplitud respecto al pico.
    Calcula el umbral localmente mediante la relación:
    Umbral = Base + Nivel_Relativo * (Amplitud_Pico - Base)
    """
    n = len(filtrado)
    radio_max = int(round(fs * ventana_max_ms / 1000.0))
    limites = []

    for i, pico_r in enumerate(picos_r):
        # Evitar traslape con latidos adyacentes
        izquierda = 0 if i == 0 else (picos_r[i - 1] + pico_r) // 2
        derecha = n - 1 if i == len(picos_r) - 1 else (pico_r + picos_r[i + 1]) // 2
        
        izquierda = max(izquierda, pico_r - radio_max)
        derecha = min(derecha, pico_r + radio_max)

        contexto = np.abs(filtrado[izquierda:derecha + 1])
        base = float(np.percentile(contexto, 10))
        amplitud_pico = float(np.abs(filtrado[pico_r]))
        umbral = base + nivel_relativo * (amplitud_pico - base)

        antes = np.flatnonzero(np.abs(filtrado[izquierda:pico_r + 1]) <= umbral)
        despues = np.flatnonzero(np.abs(filtrado[pico_r:derecha + 1]) <= umbral)

        inicio = izquierda if len(antes) == 0 else izquierda + antes[-1] + 1
        fin = derecha if len(despues) == 0 else pico_r + despues[0] - 1
        
        limites.append((int(inicio), int(fin), base, float(umbral)))

    return limites


def segmentar_qrs_registro(
    senal: np.ndarray, 
    fs: float = 100.0, 
    band: tuple[float, float] = (8.0, 20.0),
    nivel_relativo: float = 0.05,
    min_leads_consenso: int = 2
) -> list[dict]:
    """
    Segmenta los complejos QRS de un registro 4-canales [DI, DII, V1, V6]
    utilizando consenso multi-derivación para los picos R y envolvente de amplitud para los límites.
    """
    # Mapeo de derivaciones
    idx_di, idx_v1, idx_v6 = 0, 2, 3
    
    filtrado_di = bandpass_filter(senal[:, idx_di], fs, *band)
    filtrado_v1 = bandpass_filter(senal[:, idx_v1], fs, *band)
    filtrado_v6 = bandpass_filter(senal[:, idx_v6], fs, *band)

    # 1. Detección de R-peaks por Consenso Multi-derivación
    leads_dict = {"DI": senal[:, idx_di], "V1": senal[:, idx_v1], "V6": senal[:, idx_v6]}
    picos_r, _ = detect_qrs_multilead(leads_dict, fs=fs, band=band, min_leads=min_leads_consenso)

    if len(picos_r) == 0:
        return []

    # 2. Delimitación de fronteras por derivación
    limites_di = limites_por_amplitud(filtrado_di, picos_r, fs, nivel_relativo=nivel_relativo)
    limites_v1 = limites_por_amplitud(filtrado_v1, picos_r, fs, nivel_relativo=nivel_relativo)
    limites_v6 = limites_por_amplitud(filtrado_v6, picos_r, fs, nivel_relativo=nivel_relativo)

    segmentos = []
    for pico_r, l_di, l_v1, l_v6 in zip(picos_r, limites_di, limites_v1, limites_v6):
        # Envolvente = Unión amplia para conservar detalles de todas las derivaciones
        inicio = min(l_di[0], l_v1[0], l_v6[0])
        fin = max(l_di[1], l_v1[1], l_v6[1])
        
        inicio = max(0, min(inicio, int(pico_r)))
        fin = min(len(senal) - 1, max(fin, int(pico_r)))
        
        ancho_env_ms = (fin - inicio + 1) * 1000.0 / fs
        ancho_di_ms = (l_di[1] - l_di[0] + 1) * 1000.0 / fs
        ancho_v1_ms = (l_v1[1] - l_v1[0] + 1) * 1000.0 / fs
        ancho_v6_ms = (l_v6[1] - l_v6[0] + 1) * 1000.0 / fs

        segmentos.append({
            "senal_qrs": senal[inicio:fin + 1, [idx_di, idx_v1, idx_v6]],
            "inicio": inicio,
            "fin": fin,
            "indice_R": int(pico_r),
            "indice_R_segmento": int(pico_r - inicio),
            "ancho_envolvente_ms": ancho_env_ms,
            "ancho_max_ms": max(ancho_di_ms, ancho_v1_ms, ancho_v6_ms),
            "ancho_di_ms": ancho_di_ms,
            "ancho_v1_ms": ancho_v1_ms,
            "ancho_v6_ms": ancho_v6_ms,
        })
        
    return segmentos

