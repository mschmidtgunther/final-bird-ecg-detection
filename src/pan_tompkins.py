"""
pan_tompkins.py
================
Detección de complejos QRS mediante el algoritmo de Pan-Tompkins (1985),
adaptado como reemplazo del find_peaks usado en la versión anterior
(features.py, celda 7).

Reemplaza la lógica de detectar_latidos_v1() del pipeline original.

Diferencia metodológica clave respecto de la versión anterior:
    - Antes: find_peaks directo sobre V1 invertida (V1 tiene deflexión
      dominante negativa), sin garantía de que funcionara en otras
      derivaciones.
    - Ahora: Pan-Tompkins corre sobre Lead I (deflexión R dominante
      positiva, el supuesto clásico del algoritmo) y los índices
      resultantes se reutilizan para V1 y V6, ya que las tres
      derivaciones comparten la misma base temporal (mismo fs,
      mismo registro). Esto evita tener que correr y validar la
      detección por separado en cada derivación.

Nota de adaptación: el filtro pasa-banda 5-15 Hz y el filtro derivada
se implementan con scipy en vez de los filtros enteros recursivos del
paper original de 1985 (pensados para fs=200 Hz en hardware de la
época). La lógica de las 5 etapas (pasa-banda → derivada → cuadrado →
integración por ventana móvil → umbral adaptativo con búsqueda hacia
atrás) es la misma y es la que corresponde documentar en el informe.

Uso típico:
    from pan_tompkins import detectar_qrs_pan_tompkins
    picos_r = detectar_qrs_pan_tompkins(senal_lead_I, fs=100, plot=True)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# =============================================================================
# ETAPA 1 — Filtro pasa-banda 5-15 Hz (realza el complejo QRS)
# =============================================================================

def _filtro_pasabanda_qrs(senal: np.ndarray, fs: float, orden: int = 2) -> np.ndarray:
    """
    Acentúa la banda de frecuencias donde se concentra la energía del QRS
    (5-15 Hz), atenuando onda P, onda T y ruido de alta frecuencia.
    La señal de entrada ya viene filtrada en 0.5-40 Hz (data_loader.py),
    así que este es un segundo filtrado, más angosto, específico para
    la detección.
    """
    nyquist = fs / 2.0
    low = 5.0 / nyquist
    high = min(15.0 / nyquist, 0.99)  # por si fs es baja (ej. fs=100 -> nyq=50)
    b, a = butter(orden, [low, high], btype='band')
    return filtfilt(b, a, senal)


# =============================================================================
# ETAPA 2 — Filtro derivada (resalta la pendiente pronunciada del QRS)
# =============================================================================

def _filtro_derivada(senal: np.ndarray, fs: float) -> np.ndarray:
    """
    Derivada de 5 puntos, la misma usada en el paper original:
        y[n] = (1/8T) * (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
    Aproxima la pendiente de la señal, que es máxima durante el QRS.
    """
    T = 1.0 / fs
    kernel = np.array([-1, -2, 0, 2, 1]) / (8 * T)
    return np.convolve(senal, kernel, mode='same')


# =============================================================================
# ETAPA 3 — Elevación al cuadrado (vuelve todo positivo, enfatiza picos altos)
# =============================================================================

def _elevar_al_cuadrado(senal: np.ndarray) -> np.ndarray:
    return senal ** 2


# =============================================================================
# ETAPA 4 — Integración por ventana móvil (suaviza y da un "bulto" por latido)
# =============================================================================

def _integracion_ventana_movil(senal: np.ndarray, fs: float, ancho_ms: float = 150.0) -> np.ndarray:
    """
    Promedio móvil de ancho ~150 ms (recomendado en el paper original,
    aproxima la duración típica de un QRS). Cada muestra del resultado
    es el promedio de las N muestras previas, generando una "joroba"
    por cada latido que facilita la detección de picos.
    """
    n_muestras = max(1, int((ancho_ms / 1000.0) * fs))
    kernel = np.ones(n_muestras) / n_muestras
    return np.convolve(senal, kernel, mode='same')


# =============================================================================
# ETAPA 5 — Umbral adaptativo con búsqueda hacia atrás (search-back)
# =============================================================================

def _detectar_picos_integrada(senal_integrada: np.ndarray, fs: float) -> np.ndarray:
    """
    Detecta picos candidatos en la señal integrada usando el esquema de
    doble umbral adaptativo (SPKI/NPKI) de Pan-Tompkins, con período
    refractario y búsqueda hacia atrás (search-back) para latidos
    perdidos por umbral demasiado alto.

    Devuelve los índices (en la señal integrada) donde se ubica cada
    "joroba" QRS candidata. La corrección al pico R real sobre la señal
    original se hace después, en detectar_qrs_pan_tompkins().
    """
    # 250 ms en vez de los 200 ms clásicos del paper: en QRS anchos con
    # patrón rSR' (IRBBB/CRBBB) el pico R y el R' pueden generar dos máximos
    # separados en la señal integrada por un poco más de 200 ms una vez que
    # se les aplica el suavizado de la ventana móvil. 250 ms sigue siendo
    # seguro para no fusionar dos latidos reales distintos: incluso a 200
    # lpm (taquicardia extrema, poco esperable en reposo) el RR mínimo es
    # de 300 ms.
    refractario = int(0.25 * fs)

    # Inicialización de umbrales con los primeros 2 segundos de señal
    ventana_init = senal_integrada[: min(len(senal_integrada), int(2 * fs))]
    if len(ventana_init) == 0 or np.max(ventana_init) == 0:
        return np.array([], dtype=int)

    SPKI = float(np.max(ventana_init)) * 0.25   # estimador de pico de señal
    NPKI = float(np.mean(ventana_init)) * 0.5   # estimador de pico de ruido
    umbral1 = NPKI + 0.25 * (SPKI - NPKI)
    umbral2 = 0.5 * umbral1

    picos = []
    rr_promedio = None
    ultimo_pico = None
    ya_busque_atras = False  # evita disparar search-back repetidas veces
                              # sobre el mismo tramo (esto generaba
                              # detecciones duplicadas en QRS anchos/con muesca)

    i = 1
    n = len(senal_integrada)
    while i < n - 1:
        # Buscar un máximo local
        if senal_integrada[i] > senal_integrada[i - 1] and senal_integrada[i] >= senal_integrada[i + 1]:
            if senal_integrada[i] > umbral1:
                # Respetar período refractario
                if ultimo_pico is None or (i - ultimo_pico) > refractario:
                    picos.append(i)
                    SPKI = 0.125 * senal_integrada[i] + 0.875 * SPKI

                    if ultimo_pico is not None:
                        rr = i - ultimo_pico
                        rr_promedio = rr if rr_promedio is None else 0.8 * rr_promedio + 0.2 * rr
                    ultimo_pico = i
                    ya_busque_atras = False
                    i += refractario  # saltar el período refractario
                    umbral1 = NPKI + 0.25 * (SPKI - NPKI)
                    umbral2 = 0.5 * umbral1
                    continue
            else:
                NPKI = 0.125 * senal_integrada[i] + 0.875 * NPKI
                umbral1 = NPKI + 0.25 * (SPKI - NPKI)
                umbral2 = 0.5 * umbral1

            # Búsqueda hacia atrás (search-back): si pasó más de 1.66x el RR
            # promedio sin detectar nada, es probable que haya un latido
            # "escondido" entre umbral2 y umbral1 -> lo buscamos, pero SOLO
            # una vez por tramo (si no, cualquier micro-máximo de ruido en la
            # subida hacia el próximo QRS real dispara una detección extra
            # antes de llegar al pico verdadero).
            if rr_promedio is not None and ultimo_pico is not None and not ya_busque_atras:
                if (i - ultimo_pico) > int(1.66 * rr_promedio):
                    ya_busque_atras = True
                    inicio_busq = ultimo_pico + refractario
                    fin_busq = i
                    if fin_busq > inicio_busq:
                        segmento = senal_integrada[inicio_busq:fin_busq]
                        if len(segmento) > 0:
                            idx_local = int(np.argmax(segmento))
                            if segmento[idx_local] > umbral2:
                                idx_global = inicio_busq + idx_local
                                picos.append(idx_global)
                                SPKI = 0.25 * segmento[idx_local] + 0.75 * SPKI
                                rr = idx_global - ultimo_pico
                                rr_promedio = 0.8 * rr_promedio + 0.2 * rr
                                ultimo_pico = idx_global
        i += 1

    picos = np.array(sorted(set(picos)), dtype=int)

    # Red de seguridad: si por algún motivo quedaron dos detecciones a menos
    # del período refractario (p.ej. un QRS con muesca muy profunda produjo
    # dos jorobas separadas en la señal integrada), nos quedamos con la de
    # mayor amplitud y descartamos la otra.
    if len(picos) > 1:
        filtrados = [picos[0]]
        for punto in picos[1:]:
            if punto - filtrados[-1] <= refractario:
                if senal_integrada[punto] > senal_integrada[filtrados[-1]]:
                    filtrados[-1] = punto
            else:
                filtrados.append(punto)
        picos = np.array(filtrados, dtype=int)

    return picos


# =============================================================================
# FUNCIÓN PRINCIPAL — orquesta las 5 etapas
# =============================================================================

def detectar_qrs_pan_tompkins(
    senal: np.ndarray,
    fs: float,
    plot: bool = False,
    ventana_correccion_ms: float = 100.0,
    titulo: str = "",
) -> np.ndarray:
    """
    Detecta los picos R de una señal ECG mediante el algoritmo de
    Pan-Tompkins (pasa-banda -> derivada -> cuadrado -> integración
    -> umbral adaptativo con búsqueda hacia atrás).

    Parámetros
    ----------
    senal : np.ndarray
        Señal ECG ya filtrada (0.5-40 Hz), típicamente Lead I por su
        deflexión R dominante positiva.
    fs : float
        Frecuencia de muestreo (Hz).
    plot : bool
        Si True, grafica las etapas intermedias y los picos finales.
    ventana_correccion_ms : float
        Ventana de búsqueda alrededor de cada pico detectado en la señal
        integrada, para corregir el retardo de fase introducido por la
        derivada y la integración y ubicar el pico R real sobre la señal
        original.
    titulo : str
        Texto identificador para el gráfico (ej. nombre del paciente).

    Returns
    -------
    np.ndarray con los índices (en la señal original) de cada pico R detectado.
    """
    pasabanda = _filtro_pasabanda_qrs(senal, fs)
    derivada = _filtro_derivada(pasabanda, fs)
    cuadrado = _elevar_al_cuadrado(derivada)
    integrada = _integracion_ventana_movil(cuadrado, fs)

    picos_integrada = _detectar_picos_integrada(integrada, fs)

    # Corrección: buscar el verdadero pico R en la señal pasabanda original,
    # dentro de una ventana alrededor de cada detección de la señal integrada,
    # porque la derivada + integración introducen un corrimiento de fase.
    media_ventana = int((ventana_correccion_ms / 1000.0) * fs)
    picos_r = []
    for idx in picos_integrada:
        inicio = max(0, idx - media_ventana)
        fin = min(len(pasabanda), idx + media_ventana)
        seg = pasabanda[inicio:fin]
        if len(seg) == 0:
            continue
        idx_local = int(np.argmax(np.abs(seg)))
        picos_r.append(inicio + idx_local)

    picos_r = np.array(sorted(set(picos_r)), dtype=int)

    # Segundo filtro de fusión, ahora sobre la señal original (no la
    # integrada): un QRS ancho con patrón rSR' (IRBBB/CRBBB/CLBBB) puede
    # generar dos "jorobas" en la señal integrada que sobreviven al primer
    # refractario. Acá fusionamos cualquier par de detecciones a menos de
    # 120 ms (separación intra-QRS típica, incluso en bloqueos completos)
    # y nos quedamos con la de mayor amplitud absoluta real -- que es la
    # mejor aproximación al pico R verdadero. 120 ms es seguro frente a
    # taquicardias de reposo (RR mínimo esperable ~300-400 ms).
    dist_fusion = int(0.18 * fs)
    if len(picos_r) > 1:
        fusionados = [picos_r[0]]
        for punto in picos_r[1:]:
            if punto - fusionados[-1] <= dist_fusion:
                if abs(pasabanda[punto]) > abs(pasabanda[fusionados[-1]]):
                    fusionados[-1] = punto
            else:
                fusionados.append(punto)
        picos_r = np.array(fusionados, dtype=int)

    if plot:
        tiempo = np.arange(len(senal)) / fs
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axs[0].plot(tiempo, senal, color='#2c3e50')
        axs[0].set_title(f'Señal original — {titulo}', fontweight='bold')
        axs[1].plot(tiempo, pasabanda, color='#2980b9')
        axs[1].set_title('Pasa-banda 5-15 Hz')
        axs[2].plot(tiempo, integrada, color='#27ae60')
        axs[2].set_title('Señal integrada (ventana móvil 150 ms)')
        axs[3].plot(tiempo, senal, color='#2c3e50', label='Señal original')
        if len(picos_r) > 0:
            axs[3].plot(tiempo[picos_r], senal[picos_r], "kx",
                        markersize=12, markeredgewidth=3, label='Picos R (Pan-Tompkins)')
        axs[3].set_title('Detección final')
        axs[3].set_xlabel('Tiempo (s)')
        axs[3].legend()
        for ax in axs:
            ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return picos_r
