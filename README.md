# Detección de Bloqueos de Rama (Completos e Incompletos) mediante Machine Learning

Este proyecto implementa modelos de Machine Learning para la clasificación morfológica intraventricular en señales de electrocardiograma (ECG), enfocándose específicamente en la diferenciación entre bloqueos de rama completos e incompletos (CRBBB, IRBBB, CLBBB, ILBBB).

## 📊 Bases de Datos Utilizadas y Justificación Metodológica

Para garantizar la robustez, generalización y estabilidad temporal del modelo, el entrenamiento y validación se sostienen sobre tres repositorios clínicos complementarios. La unificación de estas bases permite mitigar el sesgo inter-hospitalario y evaluar el algoritmo tanto en ventanas estáticas como en entornos dinámicos (Holter).

| Base de Datos | Derivaciones | Frecuencia ($f_s$) | Aporte principal al proyecto y justificación |
| :--- | :--- | :--- | :--- |
| **PTB-XL** | 12 derivaciones (clínicas estándar) | 100 Hz / 500 Hz | Base de control principal. Aporta un volumen masivo de casos para clases normales (NORM) y una subclasificación clínica sumamente detallada para bloqueos de rama completos (CRBBB, CLBBB) e incompletos (IRBBB, ILBBB). |
| **CPSC 2018** | 12 derivaciones (longitud variable) | 500 Hz | Introduce variabilidad poblacional masiva (recopilada en 11 hospitales distintos). Su inclusión es crítica para expandir las clases minoritarias, mitigar el sobreajuste (overfitting) de entorno y validar la capacidad de generalización del modelo frente a distintos equipos de adquisición. |
| **St. Petersburg INCART** | 12 derivaciones (registros continuos de 30 min) | 257 Hz | Aporta continuidad temporal clínica. Al estar anotada latido a latido, permite extraer complejos QRS individuales para evaluar la estabilidad de las predicciones a lo largo del tiempo, mitigar el ruido dinámico por movimiento del paciente y robustecer el sistema frente a variaciones de frecuencia cardíaca. |

## 📥 Adquisición de Datos

Debido a su gran volumen, los conjuntos de datos no están incluidos en este repositorio. Para replicar este entorno, es necesario descargar los registros de forma externa en formato estándar de PhysioNet (`.mat` para señales digitales crudas y `.hea` para metadatos y etiquetas diagnósticas).

**Enlaces de descarga originales:**
* [PTB-XL Database - PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
* [China Physiological Signal Challenge 2018 - Kaggle](https://www.kaggle.com/datasets/physionet/china-physiological-signal-challenge-in-2018)
* 

> **Nota de preprocesamiento:** Debido a la heterogeneidad en las frecuencias de muestreo originales (100 Hz, 257 Hz, 500 Hz), el pipeline de datos incluye un bloque inicial de *resampling* paramétrico para unificar la resolución temporal de las señales antes de alimentar la red neuronal. Se recomienda el uso de la librería oficial `wfdb` en Python para la lectura de los registros.