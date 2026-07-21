"""
gui.py — Interfaz gráfica de CardioExpert (NORM / LBBB / RBBB).

Tres pestañas:
  1. Registro individual  — carga un .hea crudo, un .npy preprocesado, o un
                             ID del catálogo. Muestra la señal con la ventana
                             adaptativa del QRS, la clase predicha, sus
                             probabilidades y (opcional) el detalle de las
                             51 features.
  2. Análisis por lote     — procesa una carpeta de archivos crudos o una
                             lista de IDs del catálogo. Tabla de resultados
                             exportable + distribución de clases predichas.
  3. Evaluar algoritmo     — carga un CSV de features ya extraídas con
                             etiqueta real (p. ej. features_test.csv) y
                             muestra classification_report, matriz de
                             confusión y AUC-ROC macro.

Requiere ejecutarse desde la raíz del repo: `python gui.py`
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit, QGroupBox,
    QMessageBox, QRadioButton, QButtonGroup, QCheckBox, QHeaderView,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import MODEL_PATH, RAW_DIR
from src.data_loader import cargar_registro, cargar_catalogo, RegistroInvalido
from src.features import extraer_todas_las_caracteristicas, IDX_DI, IDX_V1, IDX_V6
from src.qrs_segmentation import segmentar_qrs_registro

FS = 100.0
CLASES = ["NORM", "LBBB", "RBBB"]
COLOR_CLASE = {"NORM": "#27ae60", "LBBB": "#c0392b", "RBBB": "#2980b9"}


# =============================================================================
# LÓGICA COMPARTIDA (no depende de Qt — reusada por las 3 pestañas)
# =============================================================================
def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {MODEL_PATH}.\n"
            f"Corré 'python main.py --only 7' (tuning) para generarlo."
        )
    return joblib.load(MODEL_PATH)


def predecir_registro(senal: np.ndarray, modelo, fs: float = FS) -> dict:
    """Extrae las 51 features y corre el modelo sobre un único registro."""
    feats = extraer_todas_las_caracteristicas(senal, fs=fs)
    X = pd.DataFrame([feats])

    if hasattr(modelo, "feature_names_in_"):
        for c in modelo.feature_names_in_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[modelo.feature_names_in_]

    clase_pred = modelo.predict(X)[0]
    proba = modelo.predict_proba(X)[0]
    return {
        "clase_predicha": clase_pred,
        "probabilidades": dict(zip(modelo.classes_, proba)),
        "features": feats,
    }


def buscar_hea_recursivo(carpeta: Path) -> list[Path]:
    return sorted(carpeta.rglob("*.hea"))


# =============================================================================
# WIDGET AUXILIAR: canvas de matplotlib embebido en Qt
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=4):
        self.fig = Figure(figsize=(width, height), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def limpiar(self):
        self.ax.clear()
        self.draw()


# =============================================================================
# PESTAÑA 1 — Registro individual
# =============================================================================
class TabIndividual(QWidget):
    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo
        self.catalogo = None  # se carga lazy, solo si se usa el modo "ID"

        layout = QVBoxLayout(self)

        # --- Selector de entrada ---
        grupo_entrada = QGroupBox("Entrada")
        form = QFormLayout(grupo_entrada)

        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems([
            "Archivo crudo (.hea)", "Archivo preprocesado (.npy)", "ID del catálogo",
        ])
        self.combo_tipo.currentIndexChanged.connect(self._actualizar_modo_entrada)

        fila_entrada = QHBoxLayout()
        self.txt_entrada = QLineEdit()
        self.btn_examinar = QPushButton("Examinar…")
        self.btn_examinar.clicked.connect(self._examinar_archivo)
        fila_entrada.addWidget(self.txt_entrada)
        fila_entrada.addWidget(self.btn_examinar)

        form.addRow("Tipo de entrada:", self.combo_tipo)
        form.addRow("Archivo / ID:", fila_entrada)

        self.btn_analizar = QPushButton("Analizar registro")
        self.btn_analizar.clicked.connect(self._analizar)
        form.addRow(self.btn_analizar)

        layout.addWidget(grupo_entrada)

        # --- Resultado ---
        self.lbl_resultado = QLabel("Sin analizar todavía.")
        self.lbl_resultado.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        layout.addWidget(self.lbl_resultado)

        fila_graficos = QHBoxLayout()
        self.canvas_senal = MplCanvas(width=7, height=3.5)
        self.canvas_proba = MplCanvas(width=3.5, height=3.5)
        fila_graficos.addWidget(self.canvas_senal, stretch=3)
        fila_graficos.addWidget(self.canvas_proba, stretch=1)
        layout.addLayout(fila_graficos)

        # --- Detalle de features (colapsable con checkbox) ---
        self.chk_features = QCheckBox("Mostrar detalle de las 51 features")
        self.chk_features.stateChanged.connect(
            lambda: self.tabla_features.setVisible(self.chk_features.isChecked())
        )
        layout.addWidget(self.chk_features)

        self.tabla_features = QTableWidget(0, 2)
        self.tabla_features.setHorizontalHeaderLabels(["Feature", "Valor"])
        self.tabla_features.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tabla_features.setVisible(False)
        layout.addWidget(self.tabla_features)

        self._actualizar_modo_entrada()

    def _actualizar_modo_entrada(self):
        es_id = self.combo_tipo.currentIndex() == 2
        self.btn_examinar.setVisible(not es_id)
        self.txt_entrada.setPlaceholderText(
            "Ej: JS39226" if es_id else "Seleccioná un archivo…"
        )

    def _examinar_archivo(self):
        if self.combo_tipo.currentIndex() == 0:
            ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar registro crudo", str(RAW_DIR), "WFDB (*.hea)")
        else:
            ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar .npy preprocesado", "", "NumPy (*.npy)")
        if ruta:
            self.txt_entrada.setText(ruta)

    def _analizar(self):
        entrada = self.txt_entrada.text().strip()
        if not entrada:
            QMessageBox.warning(self, "Falta la entrada", "Ingresá un archivo o un ID antes de analizar.")
            return

        tipo = {0: "hea", 1: "npy", 2: "id"}[self.combo_tipo.currentIndex()]

        try:
            if tipo == "id":
                if self.catalogo is None:
                    self.catalogo = cargar_catalogo()
                senal, meta = cargar_registro(entrada, tipo="id", catalogo=self.catalogo)
            else:
                senal, meta = cargar_registro(entrada, tipo=tipo)

            resultado = predecir_registro(senal, self.modelo)
        except RegistroInvalido as e:
            QMessageBox.critical(self, "Registro inválido", str(e))
            return
        except Exception:
            QMessageBox.critical(self, "Error inesperado", traceback.format_exc())
            return

        self._mostrar_resultado(senal, resultado, meta)

    def _mostrar_resultado(self, senal: np.ndarray, resultado: dict, meta: dict):
        clase_pred = resultado["clase_predicha"]
        proba = resultado["probabilidades"]
        color = COLOR_CLASE.get(clase_pred, "black")

        texto = f"Predicción: {clase_pred}  (confianza: {proba[clase_pred]*100:.1f}%)"
        if meta.get("clase_real"):
            acierto = "✅" if meta["clase_real"] == clase_pred else "❌"
            texto += f"   |   Clase real: {meta['clase_real']} {acierto}"
        if meta.get("dataset"):
            texto += f"   |   Dataset: {meta['dataset']}"
        self.lbl_resultado.setText(texto)
        self.lbl_resultado.setStyleSheet(
            f"font-size: 16px; font-weight: bold; padding: 6px; color: {color};"
        )

        # --- Señal + ventanas adaptativas (V1) ---
        ax = self.canvas_senal.ax
        ax.clear()
        tiempo = np.arange(len(senal)) / FS
        ax.plot(tiempo, senal[:, IDX_V1], color="#2c3e50", lw=1.0, label="V1")
        segmentos = segmentar_qrs_registro(senal, FS)
        for i, seg in enumerate(segmentos):
            ax.axvspan(seg["inicio"] / FS, seg["fin"] / FS, color="orange", alpha=0.25,
                       label="Ventana QRS" if i == 0 else None)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud (mV)")
        ax.set_title(f"Señal V1 — {len(segmentos)} latidos detectados")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        self.canvas_senal.draw()

        # --- Barras de probabilidad ---
        ax2 = self.canvas_proba.ax
        ax2.clear()
        nombres = list(proba.keys())
        valores = [proba[c] * 100 for c in nombres]
        colores = [COLOR_CLASE.get(c, "gray") for c in nombres]
        ax2.barh(nombres, valores, color=colores)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Probabilidad (%)")
        ax2.set_title("Confianza del modelo")
        for i, v in enumerate(valores):
            ax2.text(v + 2, i, f"{v:.1f}%", va="center", fontsize=9)
        self.canvas_proba.draw()

        # --- Tabla de features ---
        feats = resultado["features"]
        self.tabla_features.setRowCount(len(feats))
        for i, (k, v) in enumerate(feats.items()):
            self.tabla_features.setItem(i, 0, QTableWidgetItem(k))
            self.tabla_features.setItem(i, 1, QTableWidgetItem(f"{v:.5f}"))


# =============================================================================
# PESTAÑA 2 — Análisis por lote
# =============================================================================
class TabLote(QWidget):
    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo
        self.df_resultados: pd.DataFrame | None = None

        layout = QVBoxLayout(self)

        grupo = QGroupBox("Fuente de datos")
        form = QFormLayout(grupo)

        self.radio_carpeta = QRadioButton("Carpeta con archivos crudos (.hea)")
        self.radio_csv_ids = QRadioButton("CSV con columna 'id_registro' (usa el catálogo)")
        self.radio_carpeta.setChecked(True)
        grupo_radio = QButtonGroup(self)
        grupo_radio.addButton(self.radio_carpeta)
        grupo_radio.addButton(self.radio_csv_ids)

        fila_ruta = QHBoxLayout()
        self.txt_ruta = QLineEdit()
        self.btn_examinar = QPushButton("Examinar…")
        self.btn_examinar.clicked.connect(self._examinar)
        fila_ruta.addWidget(self.txt_ruta)
        fila_ruta.addWidget(self.btn_examinar)

        form.addRow(self.radio_carpeta)
        form.addRow(self.radio_csv_ids)
        form.addRow("Ruta:", fila_ruta)

        self.btn_procesar = QPushButton("Procesar lote")
        self.btn_procesar.clicked.connect(self._procesar_lote)
        form.addRow(self.btn_procesar)

        layout.addWidget(grupo)

        self.progreso = QProgressBar()
        layout.addWidget(self.progreso)

        fila_resultados = QHBoxLayout()
        self.tabla = QTableWidget(0, 3)
        self.tabla.setHorizontalHeaderLabels(["ID", "Clase predicha", "Confianza"])
        self.tabla.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.canvas_dist = MplCanvas(width=4, height=3.5)
        fila_resultados.addWidget(self.tabla, stretch=2)
        fila_resultados.addWidget(self.canvas_dist, stretch=1)
        layout.addLayout(fila_resultados)

        self.btn_exportar = QPushButton("Exportar resultados a CSV")
        self.btn_exportar.clicked.connect(self._exportar_csv)
        self.btn_exportar.setEnabled(False)
        layout.addWidget(self.btn_exportar)

    def _examinar(self):
        if self.radio_carpeta.isChecked():
            ruta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con .hea")
        else:
            ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV de IDs", "", "CSV (*.csv)")
        if ruta:
            self.txt_ruta.setText(ruta)

    def _procesar_lote(self):
        ruta = self.txt_ruta.text().strip()
        if not ruta:
            QMessageBox.warning(self, "Falta la ruta", "Elegí una carpeta o un CSV antes de procesar.")
            return

        try:
            if self.radio_carpeta.isChecked():
                archivos = buscar_hea_recursivo(Path(ruta))
                entradas = [(f.stem, str(f), "hea") for f in archivos]
                catalogo = None
            else:
                df_ids = pd.read_csv(ruta, dtype=str)
                if "id_registro" not in df_ids.columns:
                    raise RegistroInvalido("El CSV debe tener una columna 'id_registro'.")
                catalogo = cargar_catalogo()
                entradas = [(id_, id_, "id") for id_ in df_ids["id_registro"]]
        except RegistroInvalido as e:
            QMessageBox.critical(self, "Entrada inválida", str(e))
            return
        except Exception:
            QMessageBox.critical(self, "Error inesperado", traceback.format_exc())
            return

        if not entradas:
            QMessageBox.information(self, "Sin registros", "No se encontraron registros para procesar.")
            return

        self.progreso.setMaximum(len(entradas))
        self.progreso.setValue(0)

        filas = []
        for i, (id_reg, ruta_o_id, tipo) in enumerate(entradas):
            try:
                senal, _ = cargar_registro(ruta_o_id, tipo=tipo, catalogo=catalogo)
                resultado = predecir_registro(senal, self.modelo)
                filas.append({
                    "id_registro": id_reg,
                    "clase_predicha": resultado["clase_predicha"],
                    "confianza": resultado["probabilidades"][resultado["clase_predicha"]],
                })
            except Exception:
                filas.append({"id_registro": id_reg, "clase_predicha": "ERROR", "confianza": np.nan})

            self.progreso.setValue(i + 1)
            QApplication.processEvents()  # mantiene la UI responsiva durante el loop

        self.df_resultados = pd.DataFrame(filas)
        self._actualizar_tabla()
        self._actualizar_distribucion()
        self.btn_exportar.setEnabled(True)

    def _actualizar_tabla(self):
        df = self.df_resultados
        self.tabla.setRowCount(len(df))
        for i, row in df.iterrows():
            self.tabla.setItem(i, 0, QTableWidgetItem(str(row["id_registro"])))
            self.tabla.setItem(i, 1, QTableWidgetItem(str(row["clase_predicha"])))
            conf = row["confianza"]
            self.tabla.setItem(i, 2, QTableWidgetItem("" if pd.isna(conf) else f"{conf*100:.1f}%"))

    def _actualizar_distribucion(self):
        ax = self.canvas_dist.ax
        ax.clear()
        conteo = self.df_resultados["clase_predicha"].value_counts()
        colores = [COLOR_CLASE.get(c, "gray") for c in conteo.index]
        ax.bar(conteo.index, conteo.values, color=colores)
        ax.set_title("Distribución de diagnósticos")
        ax.set_ylabel("Cantidad de registros")
        self.canvas_dist.draw()

    def _exportar_csv(self):
        if self.df_resultados is None:
            return
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar resultados", "resultados_lote.csv", "CSV (*.csv)")
        if ruta:
            self.df_resultados.to_csv(ruta, index=False)
            QMessageBox.information(self, "Guardado", f"Resultados exportados a:\n{ruta}")


# =============================================================================
# PESTAÑA 3 — Evaluar algoritmo
# =============================================================================
class TabEvaluar(QWidget):
    """Corre el modelo sobre un CSV de features ya extraídas (formato
    features_test.csv: columnas de features + 'clase' + 'id_registro') y
    reporta las métricas de clasificación."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

        layout = QVBoxLayout(self)

        grupo = QGroupBox("Conjunto de evaluación")
        form = QFormLayout(grupo)

        fila_ruta = QHBoxLayout()
        self.txt_ruta = QLineEdit()
        self.txt_ruta.setPlaceholderText("CSV con features + columna 'clase' (ej. features_test.csv)")
        self.btn_examinar = QPushButton("Examinar…")
        self.btn_examinar.clicked.connect(self._examinar)
        fila_ruta.addWidget(self.txt_ruta)
        fila_ruta.addWidget(self.btn_examinar)
        form.addRow("Archivo:", fila_ruta)

        self.btn_evaluar = QPushButton("Evaluar")
        self.btn_evaluar.clicked.connect(self._evaluar)
        form.addRow(self.btn_evaluar)

        layout.addWidget(grupo)

        self.lbl_auc = QLabel("")
        self.lbl_auc.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.lbl_auc)

        fila_resultados = QHBoxLayout()
        self.txt_reporte = QTextEdit()
        self.txt_reporte.setReadOnly(True)
        self.txt_reporte.setFontFamily("Courier New")
        self.canvas_cm = MplCanvas(width=4.5, height=4.5)
        fila_resultados.addWidget(self.txt_reporte, stretch=1)
        fila_resultados.addWidget(self.canvas_cm, stretch=1)
        layout.addLayout(fila_resultados)

    def _examinar(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV de evaluación", "", "CSV (*.csv)")
        if ruta:
            self.txt_ruta.setText(ruta)

    def _evaluar(self):
        ruta = self.txt_ruta.text().strip()
        if not ruta:
            QMessageBox.warning(self, "Falta el archivo", "Elegí un CSV antes de evaluar.")
            return

        try:
            df = pd.read_csv(ruta, dtype={"id_registro": str})
            target_col = "clase" if "clase" in df.columns else "target"
            if target_col not in df.columns:
                raise RegistroInvalido("El CSV debe tener una columna 'clase' con la etiqueta real.")

            X = df.drop(columns=["id_registro", "record_id", "clase", "target", "dataset"], errors="ignore")
            y = df[target_col]

            if hasattr(self.modelo, "feature_names_in_"):
                X = X[self.modelo.feature_names_in_]

            y_pred = self.modelo.predict(X)
            y_proba = self.modelo.predict_proba(X)
        except RegistroInvalido as e:
            QMessageBox.critical(self, "Archivo inválido", str(e))
            return
        except Exception:
            QMessageBox.critical(self, "Error inesperado", traceback.format_exc())
            return

        reporte = classification_report(y, y_pred, digits=4)
        self.txt_reporte.setPlainText(reporte)

        clases_modelo = self.modelo.classes_
        try:
            y_bin = label_binarize(y, classes=clases_modelo)
            auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
            self.lbl_auc.setText(f"AUC-ROC (macro, one-vs-rest): {auc:.4f}   |   Registros evaluados: {len(y)}")
        except Exception:
            self.lbl_auc.setText(f"Registros evaluados: {len(y)}  (no se pudo calcular AUC)")

        ax = self.canvas_cm.ax
        ax.clear()
        cm = confusion_matrix(y, y_pred, labels=CLASES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de confusión")
        self.canvas_cm.draw()


# =============================================================================
# VENTANA PRINCIPAL
# =============================================================================
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioExpert — Clasificador de Bloqueos de Rama (NORM / LBBB / RBBB)")
        self.resize(1100, 720)

        try:
            modelo = cargar_modelo()
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Modelo no encontrado", str(e))
            sys.exit(1)

        tabs = QTabWidget()
        tabs.addTab(TabIndividual(modelo), "Registro individual")
        tabs.addTab(TabLote(modelo), "Análisis por lote")
        tabs.addTab(TabEvaluar(modelo), "Evaluar algoritmo")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
