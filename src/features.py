import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal
from scipy.signal import welch
from scipy.integrate import trapezoid
import pywt

from src.qrs_segmentation import segmentar_qrs_registro

# =============================================================================
# CONSTANTES GLOBALES Y CONFIGURACIÓN
# =============================================================================
# Índices en la señal original (4 canales: DI, DII, V1, V6)
IDX_DI, IDX_DII, IDX_V1, IDX_V6 = 0, 1, 2, 3

# Índices locales dentro de seg['senal_qrs'] (canales extraídos: DI, V1, V6)
LOCAL_DI, LOCAL_V1, LOCAL_V6 = 0, 1, 2

# Umbral de nivel relativo para la delimitación adaptativa
NIVEL_RELATIVO_FINAL = 0.05

# Configuración Espectral (Welch)
BANDA_TOTAL = (0.5, 40.0)
BANDA_QRS   = (2.0, 20.0)
MARGEN_MS   = 60

# Configuración Wavelet (SWT)
WAVELET_DWT   = 'coif5'
NIVEL_DWT     = 4
MARGEN_DWT_MS = 60
FS_REF        = 100.0
NYQUIST       = FS_REF / 2.0

# Bandas de frecuencia aproximadas DWT (fs=100 Hz, Nyquist=50 Hz)
BANDAS_DWT = {
    'aprox': (0.0, NYQUIST / (2**NIVEL_DWT)),
    'detalle4': (NYQUIST / (2**4), NYQUIST / (2**3)),
    'detalle3': (NYQUIST / (2**3), NYQUIST / (2**2)),
    'detalle2': (NYQUIST / (2**2), NYQUIST / (2**1)),
    'detalle1': (NYQUIST / (2**1), NYQUIST)
}

SUBBANDAS_DWT = {'aprox': (0, NIVEL_DWT)}
for _i in range(1, NIVEL_DWT + 1):
    _nivel_detalle = NIVEL_DWT - _i + 1
    SUBBANDAS_DWT[f'detalle{_nivel_detalle}'] = (_i, _nivel_detalle)

LEADS_DWT = [('v1', IDX_V1, '#2c3e50'), ('v6', IDX_V6, '#27ae60'), ('lead_I', IDX_DI, '#8e44ad')]


# =============================================================================
# 1. CARACTERÍSTICAS MORFOLÓGICAS (9 Features)
# =============================================================================
def calcular_area_qrs_v1(senal, fs, id_registro='', plot=False):
    """1. Área absoluta (energía) del QRS en V1."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    areas = [trapezoid(np.abs(seg['senal_qrs'][:, LOCAL_V1]), dx=1/fs) for seg in segmentos]
    
    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal[:, IDX_V1], color='#2c3e50', lw=1.2, label='Señal V1')
        for seg in segmentos:
            v1 = seg['senal_qrs'][:, LOCAL_V1]
            t_seg = np.linspace(seg['inicio']/fs, seg['fin']/fs, len(v1))
            plt.fill_between(t_seg, 0, v1, color='#e74c3c', alpha=0.5)
        plt.title(f'Feature 1: Área Absoluta del QRS ({id_registro})')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    return float(np.mean(areas))


def calcular_ancho_qrs_lead_I(senal, fs, id_registro='', plot=False):
    """2. Duración promedio del QRS en Lead I (ms)."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    anchos_ms = [seg['ancho_di_ms'] for seg in segmentos]
    return float(np.mean(anchos_ms))


def calcular_polaridad_net_v1(senal, fs, id_registro='', plot=False):
    """3. Polaridad neta (integral con signo) en V1."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    polaridades = [trapezoid(seg['senal_qrs'][:, LOCAL_V1], dx=1/fs) for seg in segmentos]
    return float(np.mean(polaridades))


def calcular_n_picos_pos_v1(senal, fs, id_registro='', plot=False, prominencia_rel=0.15):
    """4. Cantidad promedio de picos positivos en V1 (patrón rsR')."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    conteos = []
    for seg in segmentos:
        v1 = seg['senal_qrs'][:, LOCAL_V1]
        v1_pos = np.clip(v1, 0, None)
        rango_pos = v1_pos.max()
        picos_locales = np.array([], dtype=int)
        if rango_pos > 0:
            picos_locales, _ = signal.find_peaks(v1, prominence=prominencia_rel * rango_pos)
        conteos.append(len(picos_locales))

    return float(np.mean(conteos))


def calcular_sep_r_rprime_v1(senal, fs, id_registro='', plot=False, prominencia_rel=0.15):
    """5. Separación (ms) entre R y R' en V1."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    separaciones = []
    for seg in segmentos:
        v1 = seg['senal_qrs'][:, LOCAL_V1]
        v1_pos = np.clip(v1, 0, None)
        rango_pos = v1_pos.max()
        if rango_pos > 0:
            picos_locales, _ = signal.find_peaks(v1, prominence=prominencia_rel * rango_pos)
            if len(picos_locales) >= 2:
                gap_ms = (picos_locales[1] - picos_locales[0]) * 1000 / fs
                separaciones.append(gap_ms)
            else:
                separaciones.append(0.0)
        else:
            separaciones.append(0.0)

    return float(np.mean(separaciones)) if separaciones else 0.0


def calcular_ratio_rs_v1(senal, fs, id_registro='', plot=False, eps=1e-6):
    """6. Ratio R/S en V1."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    ratios = []
    for seg in segmentos:
        v1 = seg['senal_qrs'][:, LOCAL_V1]
        amp_r = max(np.max(v1), 0.0)
        amp_s = abs(min(np.min(v1), 0.0))
        ratios.append(amp_r / (amp_s + eps))

    return float(np.mean(ratios))


def calcular_s_wave_depth_v6(senal, fs, id_registro='', plot=False):
    """7. Profundidad de onda S en V6."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    profundidades = []
    for seg in segmentos:
        v6 = seg['senal_qrs'][:, LOCAL_V6]
        min_val = np.min(v6)
        profundidades.append(abs(min_val) if min_val < 0 else 0.0)

    return float(np.mean(profundidades))


def calcular_ratio_rs_v6(senal, fs, id_registro='', plot=False, eps=1e-6):
    """8. Ratio R/S en V6."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    ratios = []
    for seg in segmentos:
        v6 = seg['senal_qrs'][:, LOCAL_V6]
        amp_r = max(np.max(v6), 0.0)
        amp_s = abs(min(np.min(v6), 0.0))
        ratios.append(amp_r / (amp_s + eps))

    return float(np.mean(ratios))


def calcular_r_amp_lead_I(senal, fs, id_registro='', plot=False):
    """9. Amplitud de onda R en Lead I."""
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)
    if len(segmentos) == 0:
        return 0.0

    amplitudes = [max(np.max(seg['senal_qrs'][:, LOCAL_DI]), 0.0) for seg in segmentos]
    return float(np.mean(amplitudes))


# =============================================================================
# 2. CARACTERÍSTICAS ESPECTRALES - WELCH PSD (12 Features)
# =============================================================================
def _welch_por_latido(senal, fs, idx_lead, segmentos, margen_ms=MARGEN_MS,
                       banda_total=BANDA_TOTAL, banda_qrs=BANDA_QRS):
    margen = int(round(fs * margen_ms / 1000.0))
    pxx_por_latido = []
    freqs_ref = None

    for seg in segmentos:
        inicio = max(0, seg['inicio'] - margen)
        fin = min(len(senal), seg['fin'] + margen)
        x = senal[inicio:fin, idx_lead]
        if len(x) < 8:
            continue
        freqs, pxx = welch(x, fs=fs, nperseg=len(x), noverlap=0, nfft=256, detrend='constant')
        freqs_ref = freqs
        pxx_por_latido.append(pxx)

    if not pxx_por_latido:
        return None

    pxx_medio = np.mean(pxx_por_latido, axis=0)
    m_total = (freqs_ref >= banda_total[0]) & (freqs_ref <= banda_total[1])
    m_qrs   = (freqs_ref >= banda_qrs[0])   & (freqs_ref <= banda_qrs[1])
    return freqs_ref, pxx_medio, np.array(pxx_por_latido), m_total, m_qrs


def _obtener_metricas_psd(senal, fs, idx_lead, segmentos):
    if len(segmentos) == 0:
        return 0.0, 0.0, 0.0, 0.0

    res = _welch_por_latido(senal, fs, idx_lead, segmentos)
    if res is None:
        return 0.0, 0.0, 0.0, 0.0

    freqs, pxx_medio, _, m_total, m_qrs = res
    pot_total = float(trapezoid(pxx_medio[m_total], freqs[m_total])) if m_total.any() else 0.0
    pot_qrs   = float(trapezoid(pxx_medio[m_qrs], freqs[m_qrs])) if m_qrs.any() else 0.0
    ratio     = pot_qrs / pot_total if pot_total > 0 else 0.0

    centroide = 0.0
    if m_qrs.any() and pxx_medio[m_qrs].sum() > 0:
        centroide = float(np.sum(freqs[m_qrs] * pxx_medio[m_qrs]) / np.sum(pxx_medio[m_qrs]))

    return pot_total, pot_qrs, ratio, centroide


# =============================================================================
# 3. CARACTERÍSTICAS WAVELET - SWT COIFLET-5 (30 Features)
# =============================================================================
def _dwt_por_latido(senal, fs, idx_lead, segmentos, wavelet=WAVELET_DWT, nivel=NIVEL_DWT, margen_ms=MARGEN_DWT_MS):
    x = senal[:, idx_lead]
    n_orig = len(x)
    if n_orig < 8 * (2 ** nivel):
        return None

    factor_pad = 2 ** nivel
    resto = n_orig % factor_pad
    x_pad = np.pad(x, (0, factor_pad - resto), mode='reflect') if resto != 0 else x

    coeffs = pywt.swt(x_pad, wavelet, level=nivel, trim_approx=True)
    acumulado = {nombre: {'energia': [], 'entropia': []} for nombre in SUBBANDAS_DWT}
    margen_muestras = fs * margen_ms / 1000.0

    for seg in segmentos:
        inicio, fin = seg['inicio'], seg['fin']
        i0 = max(0, int(np.floor(inicio - margen_muestras)))
        i1 = min(n_orig, int(np.ceil(fin + margen_muestras)) + 1)
        if i1 <= i0:
            continue

        energias_abs = {}
        for nombre, (idx_coef, _) in SUBBANDAS_DWT.items():
            ventana = coeffs[idx_coef][i0:i1]
            energias_abs[nombre] = float(np.sum(ventana ** 2))

        energia_total_latido = sum(energias_abs.values()) or 1e-10

        for nombre, (idx_coef, _) in SUBBANDAS_DWT.items():
            ventana = coeffs[idx_coef][i0:i1]
            energia = energias_abs[nombre] / energia_total_latido

            potencia = ventana ** 2
            suma_potencia = potencia.sum()
            if suma_potencia > 0:
                p = potencia[potencia > 0] / suma_potencia
                entropia = float(-np.sum(p * np.log2(p)))
            else:
                entropia = 0.0

            acumulado[nombre]['energia'].append(energia)
            acumulado[nombre]['entropia'].append(entropia)

    resumen = {}
    for nombre, valores in acumulado.items():
        e_media = float(np.mean(valores['energia'])) if valores['energia'] else 0.0
        h_media = float(np.mean(valores['entropia'])) if valores['entropia'] else 0.0
        resumen[nombre] = {'energia_media': e_media, 'entropia_media': h_media}
    return resumen


# =============================================================================
# 4. EXTRACCIÓN MAESTRA CONSOLIDADORA (51 FEATURES)
# =============================================================================
def extraer_todas_las_caracteristicas(senal: np.ndarray, fs: float = 100.0) -> dict:
    """
    Extrae la suite completa de 51 características para un registro de ECG.
    Devuelve un diccionario plano listo para insertarse en un DataFrame.
    """
    feats = {}

    # 1. MORFOLÓGICAS (9)
    feats['area_qrs_v1']       = calcular_area_qrs_v1(senal, fs)
    feats['ancho_qrs_lead_I']  = calcular_ancho_qrs_lead_I(senal, fs)
    feats['polaridad_net_v1']  = calcular_polaridad_net_v1(senal, fs)
    feats['n_picos_pos_v1']    = calcular_n_picos_pos_v1(senal, fs)
    feats['sep_r_rprime_v1']   = calcular_sep_r_rprime_v1(senal, fs)
    feats['ratio_rs_v1']       = calcular_ratio_rs_v1(senal, fs)
    feats['s_wave_depth_v6']   = calcular_s_wave_depth_v6(senal, fs)
    feats['ratio_rs_v6']       = calcular_ratio_rs_v6(senal, fs)
    feats['r_amp_lead_I']      = calcular_r_amp_lead_I(senal, fs)

    # Segmentación compartida para acelerar Welch y SWT
    segmentos = segmentar_qrs_registro(senal, fs, nivel_relativo=NIVEL_RELATIVO_FINAL)

    # 2. ESPECTRALES WELCH PSD (12)
    leads_psd = [('v1', IDX_V1), ('v6', IDX_V6), ('lead_I', IDX_DI)]
    for sufijo, idx_c in leads_psd:
        p_tot, p_qrs, ratio, centr = _obtener_metricas_psd(senal, fs, idx_c, segmentos)
        feats[f'psd_potencia_total_{sufijo}']    = p_tot
        feats[f'psd_potencia_qrs_{sufijo}']      = p_qrs
        feats[f'psd_ratio_qrs_total_{sufijo}']  = ratio
        feats[f'psd_frecuencia_pico_{sufijo}']   = centr

    # 3. WAVELET SWT COIFLET-5 (30)
    for lead_suf, idx_lead, _ in LEADS_DWT:
        resumen_dwt = _dwt_por_latido(senal, fs, idx_lead, segmentos) if segmentos else None
        for sb in SUBBANDAS_DWT:
            feats[f'dwt_energia_{sb}_{lead_suf}']  = resumen_dwt[sb]['energia_media'] if resumen_dwt else 0.0
            feats[f'dwt_entropia_{sb}_{lead_suf}'] = resumen_dwt[sb]['entropia_media'] if resumen_dwt else 0.0

    return feats