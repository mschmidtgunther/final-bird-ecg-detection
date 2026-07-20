# Clasificación de Bloqueos de Rama (LBBB/RBBB) mediante Machine Learning

Este proyecto implementa modelos de Machine Learning para la clasificación morfológica intraventricular en señales de electrocardiograma (ECG), enfocándose en la detección automática de bloqueos de rama izquierda y derecha (LBBB, RBBB) frente a ritmo normal (NORM).

El dataset combinado incluye anotaciones más granulares para algunos registros (bloqueos completos e incompletos: CRBBB, IRBBB, CLBBB, ILBBB, disponibles en PTB-XL), pero esa distinción no está disponible en CPSC2018 ni en Chapman-Shaoxing, que solo etiquetan LBBB/RBBB de forma unificada. Para poder combinar las tres bases sin descartar ninguna y maximizar el volumen de datos por clase, se optó por trabajar con 3 clases objetivo (**NORM / LBBB / RBBB**), colapsando en cada base la granularidad completo/incompleto donde existía.

## 📊 Bases de Datos Utilizadas y Justificación Metodológica

Para garantizar la robustez, generalización y estabilidad temporal del modelo, el entrenamiento y validación se sostienen sobre tres repositorios clínicos complementarios. La unificación de estas bases permite mitigar el sesgo inter-hospitalario y evaluar el algoritmo tanto en ventanas estáticas como en entornos dinámicos (Holter).

| Base de Datos | Derivaciones | Frecuencia ($f_s$) | Aporte principal al proyecto y justificación |
| :--- | :--- | :--- | :--- |
| **PTB-XL** | 12 derivaciones (clínicas estándar) | 100 Hz / 500 Hz | Base de control principal. Aporta un volumen masivo de casos para clase normal (NORM). Incluye anotaciones granulares para bloqueos de rama completos (CRBBB, CLBBB) e incompletos (IRBBB, ILBBB), que se consolidan en LBBB/RBBB para mantener consistencia con el resto del dataset combinado. |
| **CPSC 2018** | 12 derivaciones (longitud variable) | 500 Hz | Introduce variabilidad poblacional masiva (recopilada en 11 hospitales distintos). Su inclusión es crítica para expandir las clases minoritarias (LBBB/RBBB), mitigar el sobreajuste (overfitting) de entorno y validar la capacidad de generalización del modelo frente a distintos equipos de adquisición. |
| **Chapman-Shaoxing** | 12 derivaciones | 500 Hz | Uno de los repositorios abiertos más extensos de arritmias (45.151 registros), con etiquetado sistemático de LBBB y RBBB. Es la base que más volumen aporta al proyecto, elevando sustancialmente la representación de bloqueos de rama en el dataset combinado y permitiendo evaluar la generalización del modelo sobre una población clínica independiente. |

## 📥 Adquisición de Datos

Debido a su gran volumen, los conjuntos de datos no están incluidos en este repositorio. Para replicar este entorno, es necesario descargar los registros de forma externa en formato estándar de PhysioNet (`.mat` para señales digitales crudas y `.hea` para metadatos y etiquetas diagnósticas).

**Enlaces de descarga originales:**
* [PTB-XL Database - PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
* [China Physiological Signal Challenge 2018 - Kaggle](https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018)
* [Chapman-Shaoxing Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

> **Nota de preprocesamiento:** Debido a la heterogeneidad en las frecuencias de muestreo originales (100 Hz, 257 Hz, 500 Hz), el pipeline de datos incluye un bloque inicial de *resampling* paramétrico para unificar la resolución temporal de las señales antes de alimentar la red neuronal. Se recomienda el uso de la librería oficial `wfdb` en Python para la lectura de los registros.


final-bird-ecg-detection/
├── data/
│   ├── raw/                        # Datasets crudos descargados (PTB-XL, CPSC2018, Chapman-Shaoxing)
│   └── processed/
│       ├── metadata/               # Tablas de etiquetas (maestra, limpia)
│       ├── signals/                # Señales filtradas y resampleadas a 100 Hz (.npy)
│       ├── intermediate/           # Matriz de features sin balancear/dividir
│       ├── features_trainval.csv   # Set de entrenamiento + validación (80%)
│       └── features_test.csv       # Set de holdout final (20%)
├── models/                         # Modelos entrenados (.joblib): RF baseline, MLP baseline, RF tuneado
├── notebooks/                      # Notebooks de exploración 
│
├── reports/                        # Matrices de confusión y reporte final de evaluación
├── src/                            # Código fuente del pipeline (paquete importable)
│   ├── __init__.py
│   ├── config.py                   # Rutas centralizadas del proyecto
│   ├── dataset_consolidator.py     # Etapa 1: unifica etiquetas de las 3 bases
│   ├── preprocess_signals.py       # Etapa 2: filtrado, resampleo y control de calidad
│   ├── pan_tompkins.py             # Detección de QRS multi-derivación (consenso)
│   ├── qrs_segmentation.py         # Delimitación de complejos QRS por amplitud
│   ├── features.py                 # Extracción de las 51 características (morfológicas, PSD, wavelet)
│   ├── extract_features.py         # Etapa 3: orquesta la extracción sobre todo el dataset
│   ├── dataset_splitter.py         # Etapa 4: balanceo (undersampling) y split estratificado
│   ├── train_rf.py                 # Etapa 5: baseline Random Forest
│   ├── train_mlp.py                # Etapa 6: baseline MLP
│   ├── tune_rf.py                  # Etapa 7: tuning conservador de Random Forest (GridSearchCV)
│   └── evaluate_test.py            # Etapa 8: evaluación final sobre el holdout
├── main.py                         # Orquestador: corre las 8 etapas del pipeline en orden
├── requirements.txt
└── README.md