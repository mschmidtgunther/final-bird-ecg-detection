"""
Pan-Tompkins multi-derivacion para deteccion de QRS en bloqueos de rama.

Implementa el algoritmo clasico (Pan & Tompkins, 1985) con dos ajustes
justificados en la bibliografia del proyecto:

1. Banda pasante 8-20 Hz en lugar de la clasica 5-11 Hz, siguiendo el
   resultado empirico de Elgendi, Jonkman & De Boer (2010) sobre 48
   registros de MIT-BIH (incluye LBBB y RBBB), que reporta el mejor
   SNR QRS/ruido en esa banda.
2. Deteccion en las tres derivaciones del proyecto (V1, V6, DI) con
   consenso: un pico solo se acepta como QRS si aparece en al menos
   2 de las 3 derivaciones dentro de una ventana de tolerancia. Esto
   compensa que en LBBB/RBBB el QRS ensanchado puede tener una
   morfologia poco prominente en una derivacion puntual (p. ej. V1
   en LBBB) pero clara en otra (V6, DI).

Referencias:
- J. Pan, W. J. Tompkins, "A real-time QRS detection algorithm", 1985.
- M. Elgendi, M. Jonkman, F. De Boer, "Frequency bands effects on QRS
  detection", BIOSIGNALS 2010.
"""

import numpy as np
from scipy.signal import butter, lfilter, find_peaks


# ---------------------------------------------------------------------
# Etapas del pipeline (cada una recibe/devuelve un array 1D)
# ---------------------------------------------------------------------

def bandpass_filter(sig, fs, low=8.0, high=20.0, order=2):
    """Filtro Butterworth pasabanda. Banda por defecto 8-20 Hz (Elgendi et al.)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, sig)


def derivative_filter(sig, fs):
    """Derivada de 5 puntos de Pan-Tompkins: resalta la pendiente del QRS."""
    T = 1.0 / fs
    b = np.array([1, 2, 0, -2, -1]) * (1.0 / (8.0 * T))
    return lfilter(b, [1.0], sig)


def squaring(sig):
    """Vuelve todo positivo y amplifica las frecuencias altas del QRS."""
    return sig ** 2


def moving_window_integration(sig, fs, window_ms=150):
    """Integra energia en una ventana ~ ancho de QRS (150 ms por defecto)."""
    n = max(1, int(round(fs * window_ms / 1000.0)))
    b = np.ones(n) / n
    return lfilter(b, [1.0], sig)


# ---------------------------------------------------------------------
# Deteccion de picos con umbral adaptativo (SPKI/NPKI) por derivacion
# ---------------------------------------------------------------------

def _adaptive_threshold_peaks(integrated, fs, refractory_ms=200):
    """Umbral adaptativo dual de Pan-Tompkins sobre la senal integrada.

    Mantiene un nivel de pico de senal (SPKI) y de ruido (NPKI) que se
    actualizan latido a latido, con un umbral de respaldo mas laxo
    para no perder QRS ensanchados/atipicos de LBBB o RBBB.
    """
    refractory = max(1, int(round(fs * refractory_ms / 1000.0)))
    candidate_peaks, _ = find_peaks(integrated, distance=refractory)
    if len(candidate_peaks) == 0:
        return np.array([], dtype=int)

    init_n = min(len(integrated), int(2 * fs))
    spki = float(np.max(integrated[:init_n])) * 0.25 if init_n > 0 else 0.0
    npki = float(np.mean(integrated[:init_n])) * 0.5 if init_n > 0 else 0.0

    accepted = []
    rr_hist = []

    for p in candidate_peaks:
        val = integrated[p]
        thr1 = npki + 0.25 * (spki - npki)
        thr2 = 0.5 * thr1

        is_qrs = val > thr1
        if not is_qrs and val > thr2 and accepted:
            avg_rr = np.mean(rr_hist[-8:]) if rr_hist else None
            if avg_rr is not None and (p - accepted[-1]) > 1.66 * avg_rr:
                is_qrs = True  # posible latido perdido -> umbral de respaldo

        if is_qrs:
            spki = 0.125 * val + 0.875 * spki
            if accepted:
                rr_hist.append(p - accepted[-1])
            accepted.append(p)
        else:
            npki = 0.125 * val + 0.875 * npki

    return np.array(accepted, dtype=int)


def _refine_peak_locations(peaks, filtered_signal, fs, window_ms, search_fwd_ms=40):
    """Corrige el retardo de grupo de la integracion/filtrado.

    La integracion de ventana movil (y los filtros previos) desplazan el
    pico detectado hacia adelante respecto del R real. Se busca el maximo
    absoluto de la senal ya filtrada (pasabanda) en una ventana que
    retrocede aproximadamente el ancho de la integracion, evitando asi
    reportar el R-peak sistematicamente tarde.
    """
    back = int(round(fs * window_ms / 1000.0))
    fwd = int(round(fs * search_fwd_ms / 1000.0))
    refined = []
    n = len(filtered_signal)
    for p in peaks:
        lo = max(0, p - back)
        hi = min(n, p + fwd)
        if hi <= lo:
            refined.append(p)
            continue
        segment = filtered_signal[lo:hi]
        refined.append(lo + int(np.argmax(np.abs(segment))))
    return np.array(refined, dtype=int)


def rescue_low_amplitude_qrs(peaks, filtered_signal, fs, refractory_ms=200, thr3_factor=1.5):
    """Recupera latidos de baja amplitud omitidos por el umbral principal.

    Basado en el paso 12 (umbral THR3) de Elgendi, Jonkman & De Boer,
    "Improved QRS detection algorithm using dynamic thresholds" (2009):
    si un intervalo RR supera 1.5x el RR modal del registro, es probable
    que se haya perdido un latido de baja amplitud dentro de ese hueco
    (en vez de una pausa real). Se busca ahi el maximo local remanente
    de la senal filtrada, respetando el periodo refractario.
    """
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if len(peaks) < 3:
        return peaks

    rr = np.diff(peaks)
    vals, counts = np.unique(np.round(rr / 10.0) * 10, return_counts=True)
    rr_mode = float(vals[np.argmax(counts)])
    thr3 = thr3_factor * rr_mode
    refractory = int(round(fs * refractory_ms / 1000.0))

    rescued = list(peaks)
    for i in range(len(rr)):
        if rr[i] <= thr3:
            continue
        lo = peaks[i] + refractory
        hi = peaks[i + 1] - refractory
        if hi <= lo:
            continue
        segment = np.abs(filtered_signal[lo:hi])
        if segment.size and segment.max() > 0:
            rescued.append(lo + int(np.argmax(segment)))

    return np.array(sorted(rescued), dtype=int)


def pan_tompkins_single_lead(sig, fs, band=(8.0, 20.0), window_ms=150, refractory_ms=200,
                              rescue_low_amplitude=True, thr3_factor=1.5):
    """Pipeline completo de Pan-Tompkins sobre una sola derivacion.

    Devuelve los indices de muestra de los R-peaks detectados, corregidos
    por el retardo de grupo de la integracion y, opcionalmente, con
    rescate de latidos de baja amplitud (Elgendi et al., 2009).
    """
    filtered = bandpass_filter(sig, fs, *band)
    deriv = derivative_filter(filtered, fs)
    sq = squaring(deriv)
    integrated = moving_window_integration(sq, fs, window_ms)
    raw_peaks = _adaptive_threshold_peaks(integrated, fs, refractory_ms)
    peaks = _refine_peak_locations(raw_peaks, filtered, fs, window_ms)
    if rescue_low_amplitude:
        peaks = rescue_low_amplitude_qrs(peaks, filtered, fs, refractory_ms, thr3_factor)
    return peaks


# ---------------------------------------------------------------------
# Consenso multi-derivacion (V1, V6, DI)
# ---------------------------------------------------------------------

def detect_qrs_multilead(leads, fs, band=(8.0, 20.0), tolerance_ms=80, min_leads=2):
    """Detecta QRS combinando V1, V6 y DI (o el subconjunto que se pase).

    Parameters
    ----------
    leads : dict[str, np.ndarray]
        p. ej. {"V1": senal_v1, "V6": senal_v6, "DI": senal_di}, todas
        alineadas muestra a muestra y a la misma fs.
    fs : float
        Frecuencia de muestreo (Hz).
    tolerance_ms : float
        Ventana para considerar que dos picos de derivaciones distintas
        corresponden al mismo latido.
    min_leads : int
        Minimo de derivaciones en las que debe aparecer un pico para
        aceptarlo como QRS de consenso (por defecto 2 de 3).

    Returns
    -------
    consensus_peaks : np.ndarray
        Indices de muestra de los R-peaks aceptados por consenso.
    per_lead_peaks : dict[str, np.ndarray]
        Picos crudos detectados en cada derivacion (util para
        validacion visual, tal como pide el objetivo 2 del anteproyecto).
    """
    tol = max(1, int(round(fs * tolerance_ms / 1000.0)))
    per_lead_peaks = {
        name: pan_tompkins_single_lead(sig, fs, band=band)
        for name, sig in leads.items()
    }

    pooled = sorted(
        (int(p), name) for name, peaks in per_lead_peaks.items() for p in peaks
    )

    clusters = []
    for p, name in pooled:
        if clusters and p - clusters[-1]["center"] <= tol:
            clusters[-1]["points"].append(p)
            clusters[-1]["leads"].add(name)
            clusters[-1]["center"] = int(np.mean(clusters[-1]["points"]))
        else:
            clusters.append({"points": [p], "leads": {name}, "center": p})

    consensus_peaks = np.array(
        [c["center"] for c in clusters if len(c["leads"]) >= min_leads], dtype=int
    )
    return consensus_peaks, per_lead_peaks


