"""
main.py — Orquestador del pipeline completo de CardioExpert V6.

Ejecuta, en orden, todas las etapas del proyecto:

  1. Consolidación de etiquetas (Chapman + CPSC2018 + PTB-XL -> tabla maestra)
  2. Preprocesamiento de señales (.hea/.dat -> .npy filtrados a 100 Hz)
  3. Extracción de las 51 características (morfológicas + PSD + wavelet)
  4. Balanceo (undersampling parcial) y split estratificado Train/Val - Test
  5. Entrenamiento baseline Random Forest        [opcional, informativo]
  6. Entrenamiento baseline MLP                  [opcional, informativo]
  7. Tuning conservador de Random Forest (GridSearchCV) -> modelo final
  8. Evaluación final sobre el holdout de test

Uso:
    python main.py                     # corre el pipeline completo
    python main.py --from 3            # arranca en la etapa 3 (asume que 1-2 ya corrieron)
    python main.py --to 4              # corre solo hasta la etapa 4 (sin entrenar/evaluar)
    python main.py --skip-baselines    # se salta las etapas 5 y 6 (van directo a tuning)
    python main.py --only 7            # corre únicamente la etapa 7

Requiere ejecutarse desde la raíz del repo (donde está la carpeta src/),
por ejemplo: `python main.py`, con `src/` como paquete importable
(debe existir `src/__init__.py`).
"""

import argparse
import sys
import time
from pathlib import Path

from src.config import (
    METADATA_DIR,
    INTERMEDIATE_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TRAIN_FEAT_PATH,
)

ETAPAS = {
    1: "Consolidación de etiquetas",
    2: "Preprocesamiento de señales",
    3: "Extracción de características",
    4: "Balanceo + split Train/Val / Test",
    5: "Entrenamiento baseline Random Forest",
    6: "Entrenamiento baseline MLP",
    7: "Tuning Random Forest (GridSearchCV)",
    8: "Evaluación final (holdout test)",
}


def _banner(n: int) -> None:
    print("\n" + "#" * 60)
    print(f"# ETAPA {n}/8 — {ETAPAS[n]}")
    print("#" * 60)


def etapa_1_consolidar():
    from src.dataset_consolidator import generar_tabla_maestra

    _banner(1)
    rutas_input = {
        "Chapman": METADATA_DIR / "chapman_etiquetas.csv",
        "CPSC2018": METADATA_DIR / "cpsc2018_etiquetas.csv",
        "PTB-XL": METADATA_DIR / "ptbxl_etiquetas.csv",
    }
    ruta_output = METADATA_DIR / "etiquetas_maestras.csv"
    generar_tabla_maestra(
        rutas_datasets=rutas_input,
        output_csv_path=ruta_output,
        clases_objetivo=["NORM", "LBBB", "RBBB"],
    )


def etapa_2_preprocesar():
    from src.config import RAW_DIR, SIGNALS_DIR
    from src.preprocess_signals import ejecutar_preprocesamiento_masivo

    _banner(2)
    csv_master = METADATA_DIR / "etiquetas_maestras.csv"
    csv_limpio_out = METADATA_DIR / "etiquetas_maestras_limpias.csv"

    if not csv_master.exists():
        sys.exit(
            f"❌ Falta {csv_master}. Corré la etapa 1 (consolidación) primero."
        )

    ejecutar_preprocesamiento_masivo(
        ruta_raw=RAW_DIR,
        ruta_destino=SIGNALS_DIR,
        ruta_csv_master=csv_master,
        ruta_csv_limpio=csv_limpio_out,
    )


def etapa_3_extraer_features():
    import pandas as pd
    from src.extract_features import procesar_dataset_features

    _banner(3)
    input_path = METADATA_DIR / "etiquetas_maestras_limpias.csv"
    output_path = INTERMEDIATE_DIR / "df_features.csv"

    if not input_path.exists():
        sys.exit(
            f"❌ Falta {input_path}. Corré la etapa 2 (preprocesamiento) primero."
        )

    df_in = pd.read_csv(input_path)
    df_out = procesar_dataset_features(df_in, fs=100.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"💾 Guardado archivo final en: {output_path}")


def etapa_4_split():
    import pandas as pd
    from src.dataset_splitter import balancear_dataset, crear_splits_train_test

    _banner(4)
    ruta_entrada = INTERMEDIATE_DIR / "df_features.csv"
    ruta_balanceado_out = INTERMEDIATE_DIR / "df_features_balanceado.csv"

    if not ruta_entrada.exists():
        sys.exit(
            f"❌ Falta {ruta_entrada}. Corré la etapa 3 (extracción de features) primero."
        )

    conteo_objetivo = {
        "NORM": 2800,
        "RBBB": 2200,
        "LBBB": None,  # se conservan todos los registros disponibles
    }

    df_master = pd.read_csv(ruta_entrada, dtype={"id_registro": str})
    df_bal = balancear_dataset(df_master, conteo_objetivo=conteo_objetivo, random_state=42)
    df_bal.to_csv(ruta_balanceado_out, index=False)

    crear_splits_train_test(
        df=df_bal, output_dir=PROCESSED_DIR, test_size=0.2, random_state=42
    )


def etapa_5_train_rf_baseline():
    from src.train_rf import entrenar_evaluar_rf

    _banner(5)
    if not TRAIN_FEAT_PATH.exists():
        sys.exit(f"❌ Falta {TRAIN_FEAT_PATH}. Corré la etapa 4 (split) primero.")
    entrenar_evaluar_rf(TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR)


def etapa_6_train_mlp_baseline():
    from src.train_mlp import entrenar_evaluar_mlp

    _banner(6)
    if not TRAIN_FEAT_PATH.exists():
        sys.exit(f"❌ Falta {TRAIN_FEAT_PATH}. Corré la etapa 4 (split) primero.")
    entrenar_evaluar_mlp(TRAIN_FEAT_PATH, MODELS_DIR, REPORTS_DIR)


def etapa_7_tune_rf():
    from src.tune_rf import optimizar_hiperparametros_conservador

    _banner(7)
    if not TRAIN_FEAT_PATH.exists():
        sys.exit(f"❌ Falta {TRAIN_FEAT_PATH}. Corré la etapa 4 (split) primero.")
    optimizar_hiperparametros_conservador(
        data_path=TRAIN_FEAT_PATH, output_dir=MODELS_DIR
    )


def etapa_8_evaluar():
    from src.config import MODEL_PATH, TEST_FEAT_PATH
    from src.evaluate_test import evaluate_holdout

    _banner(8)
    if not MODEL_PATH.exists():
        sys.exit(f"❌ Falta {MODEL_PATH}. Corré la etapa 7 (tuning) primero.")
    if not TEST_FEAT_PATH.exists():
        sys.exit(f"❌ Falta {TEST_FEAT_PATH}. Corré la etapa 4 (split) primero.")
    evaluate_holdout()


ETAPAS_FN = {
    1: etapa_1_consolidar,
    2: etapa_2_preprocesar,
    3: etapa_3_extraer_features,
    4: etapa_4_split,
    5: etapa_5_train_rf_baseline,
    6: etapa_6_train_mlp_baseline,
    7: etapa_7_tune_rf,
    8: etapa_8_evaluar,
}


def main():
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline CardioExpert V6 (NORM/LBBB/RBBB)."
    )
    parser.add_argument(
        "--from", dest="desde", type=int, default=1, choices=range(1, 9),
        help="Etapa por la que arrancar (default: 1).",
    )
    parser.add_argument(
        "--to", dest="hasta", type=int, default=8, choices=range(1, 9),
        help="Última etapa a correr (default: 8).",
    )
    parser.add_argument(
        "--only", dest="solo", type=int, default=None, choices=range(1, 9),
        help="Correr únicamente esta etapa (ignora --from/--to).",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Se salta los baselines informativos (etapas 5 y 6: RF y MLP sin tunear).",
    )
    args = parser.parse_args()

    if args.solo is not None:
        secuencia = [args.solo]
    else:
        secuencia = [n for n in range(args.desde, args.hasta + 1)]
        if args.skip_baselines:
            secuencia = [n for n in secuencia if n not in (5, 6)]

    print("Pipeline a ejecutar:")
    for n in secuencia:
        print(f"  {n}. {ETAPAS[n]}")

    inicio_total = time.time()
    for n in secuencia:
        t0 = time.time()
        ETAPAS_FN[n]()
        print(f"⏱️  Etapa {n} completada en {time.time() - t0:.1f}s")

    print(f"\n✅ Pipeline finalizado en {time.time() - inicio_total:.1f}s")


if __name__ == "__main__":
    main()