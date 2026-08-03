from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="SNII Insight · Laboratorio de visualización",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.02);
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }

        .snii-subtitle {
            color: #746a80;
            margin-top: -0.6rem;
            margin-bottom: 1.1rem;
        }

        .snii-note {
            border-left: 4px solid #6d28d9;
            padding: 0.75rem 1rem;
            background: rgba(109, 40, 217, 0.08);
            border-radius: 0 12px 12px 0;
            margin: 0.5rem 0 1rem;
        }

        .lab-chip {
            display: inline-block;
            padding: 0.30rem 0.70rem;
            margin: 0.15rem 0.20rem;
            border-radius: 999px;
            font-weight: 650;
            border: 1px solid transparent;
        }

        .lab-variable {
            background: rgba(37, 99, 235, 0.12);
            border-color: rgba(37, 99, 235, 0.35);
            color: #1d4ed8;
        }

        .lab-tiempo {
            background: rgba(22, 163, 74, 0.12);
            border-color: rgba(22, 163, 74, 0.35);
            color: #15803d;
        }

        .lab-filtro {
            background: rgba(234, 88, 12, 0.12);
            border-color: rgba(234, 88, 12, 0.35);
            color: #c2410c;
        }

        .lab-ubicacion {
            background: rgba(124, 58, 237, 0.12);
            border-color: rgba(124, 58, 237, 0.35);
            color: #6d28d9;
        }

        .lab-objetivo {
            background: rgba(219, 39, 119, 0.12);
            border-color: rgba(219, 39, 119, 0.35);
            color: #be185d;
        }

        .lab-sentence {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.7rem 0 1rem;
            font-size: 1.02rem;
            line-height: 2.1;
        }

        .lab-score {
            font-size: 1.7rem;
            font-weight: 750;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTES
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    # Master v2 para la app actual.
    APP_DIR / "data" / "SNII_MASTER_v2_APP.xlsx",
    APP_DIR / "SNII_MASTER_v2_APP.xlsx",
    APP_DIR / "data" / "SNII_MASTER_v2_APP.parquet",
    APP_DIR / "SNII_MASTER_v2_APP.parquet",

    # Compatibilidad con versiones anteriores.
    APP_DIR / "data" / "SNII_MASTER_v1_PERSONA_ANIO.parquet",
    APP_DIR / "SNII_MASTER_v1_PERSONA_ANIO.parquet",
    APP_DIR / "data" / "SNII_MASTER_v1_PERSONA_ANIO.xlsx",
    APP_DIR / "SNII_MASTER_v1_PERSONA_ANIO.xlsx",
]

COLUMNAS_MINIMAS = [
    "ID_PERSONA_EXACTA",
    "AÑO",
]

COLUMNAS_ANALITICAS = [
    "ID_PERSONA_EXACTA",
    "AÑO",
    "CVU_REFERENCIA",
    "NOMBRE_INVESTIGADOR",
    "SEXO_CONSOLIDADO",
    "PRIMER_AÑO",
    "ULTIMO_AÑO",
    "NUMERO_AÑOS_PRESENTE",
    "ESTA_VIGENTE_EN_2025",
    "INSTITUCION_ANUAL",
    "DEPENDENCIA_ANUAL",
    "SUBDEPENDENCIA_ANUAL",
    "ENTIDAD_FEDERATIVA_ANUAL",
    "PAIS_ANUAL",
    "NIVEL_SNII_STD",
    "NIVEL_SNII_ETIQUETA",
    "AREA_DEL_CONOCIMIENTO_ANUAL",
    "CAMPO_DEL_CONOCIMIENTO_ANUAL",
    "DISCIPLINA_ANUAL",
    "SUBDISCIPLINA_ANUAL",
    "ESPECIALIDAD_ANUAL",
    "CLASIFICACION_STEM_ANUAL",
    "GRUPO_STEM_BINARIO",
    "ES_STEM_ESTRICTO",
    "ES_STEM_AMPLIADO",
    "PORCENTAJE_COMPLETITUD_CLAVE",
    "REQUIERE_REVISION_MASTER",
]


# ============================================================
# UTILIDADES DE DATOS
# ============================================================

def limpiar_texto(serie: pd.Series) -> pd.Series:
    """Normaliza valores textuales vacíos sin destruir el dato."""
    resultado = serie.astype("string").str.strip()

    marcadores = {
        "",
        "NAN",
        "NONE",
        "NULL",
        "NA",
        "N/A",
        "NO HAY INFORMACIÓN AL RESPECTO",
        "NO HAY INFORMACION AL RESPECTO",
    }

    return resultado.mask(resultado.str.upper().isin(marcadores))


def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara tipos esenciales y crea aliases compatibles con el master v2."""
    df = df.copy()

    # La interfaz todavía contiene varios módulos construidos con los nombres
    # del master v1. Estos aliases permiten usar el Excel v2 sin reescribir
    # inmediatamente todas las visualizaciones existentes.
    aliases_v2_a_v1 = {
        "SEXO_ANALITICO": "SEXO_CONSOLIDADO",
        "INSTITUCION_ANALITICA": "INSTITUCION_ANUAL",
        "DEPENDENCIA_ANALITICA": "DEPENDENCIA_ANUAL",
        "ENTIDAD_ANALITICA": "ENTIDAD_FEDERATIVA_ANUAL",
        "PAIS_ANALITICO": "PAIS_ANUAL",
        "NIVEL_SNII_ANALITICO": "NIVEL_SNII_ETIQUETA",
        "AREA_ANALITICA": "AREA_DEL_CONOCIMIENTO_ANUAL",
        "DISCIPLINA_ANALITICA": "DISCIPLINA_ANUAL",
        "PORCENTAJE_COMPLETITUD_ANALITICA": "PORCENTAJE_COMPLETITUD_CLAVE",
        "REQUIERE_REVISION_ANALITICA": "REQUIERE_REVISION_MASTER",
    }

    for origen, destino in aliases_v2_a_v1.items():
        if origen in df.columns and destino not in df.columns:
            df[destino] = df[origen]

    faltantes = [col for col in COLUMNAS_MINIMAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            "La base no contiene las columnas indispensables: "
            + ", ".join(faltantes)
        )

    df["ID_PERSONA_EXACTA"] = limpiar_texto(df["ID_PERSONA_EXACTA"])
    df["AÑO"] = pd.to_numeric(df["AÑO"], errors="coerce").astype("Int64")

    columnas_texto = [
        col
        for col in [
            "CVU_REFERENCIA",
            "NOMBRE_INVESTIGADOR",
            "SEXO_CONSOLIDADO",
            "INSTITUCION_ANUAL",
            "DEPENDENCIA_ANUAL",
            "SUBDEPENDENCIA_ANUAL",
            "ENTIDAD_FEDERATIVA_ANUAL",
            "PAIS_ANUAL",
            "NIVEL_SNII_ETIQUETA",
            "AREA_DEL_CONOCIMIENTO_ANUAL",
            "CAMPO_DEL_CONOCIMIENTO_ANUAL",
            "DISCIPLINA_ANUAL",
            "SUBDISCIPLINA_ANUAL",
            "ESPECIALIDAD_ANUAL",
            "CLASIFICACION_STEM_ANUAL",
            "GRUPO_STEM_BINARIO",
        ]
        if col in df.columns
    ]

    for columna in columnas_texto:
        df[columna] = limpiar_texto(df[columna])

    if "NIVEL_SNII_STD" in df.columns:
        df["NIVEL_SNII_STD"] = pd.to_numeric(
            df["NIVEL_SNII_STD"],
            errors="coerce",
        ).astype("Int64")

    df = df.loc[
        df["ID_PERSONA_EXACTA"].notna()
        & df["AÑO"].between(2000, 2025, inclusive="both")
    ].copy()

    df = df.drop_duplicates(
        subset=["ID_PERSONA_EXACTA", "AÑO"],
        keep="last",
    )

    return df


def leer_excel_snii(origen) -> pd.DataFrame:
    """Lee MASTER, PERSONA_AÑO o, como respaldo, la primera hoja disponible."""
    libro = pd.ExcelFile(origen, engine="openpyxl")

    for hoja_preferida in ("MASTER", "PERSONA_AÑO"):
        if hoja_preferida in libro.sheet_names:
            return pd.read_excel(
                libro,
                sheet_name=hoja_preferida,
                engine="openpyxl",
            )

    if not libro.sheet_names:
        raise ValueError("El archivo Excel no contiene hojas legibles.")

    return pd.read_excel(
        libro,
        sheet_name=libro.sheet_names[0],
        engine="openpyxl",
    )


@st.cache_data(show_spinner="Cargando la base histórica del SNII…")
def cargar_desde_ruta(ruta: str) -> pd.DataFrame:
    """Carga Parquet o Excel desde el repositorio."""
    path = Path(ruta)

    if path.suffix.lower() == ".parquet":
        # Se cargan todas las columnas procesadas del master.
        df = pd.read_parquet(
            path,
            engine="pyarrow",
        )

    elif path.suffix.lower() in {".xlsx", ".xls"}:
        # El Excel se conserva como respaldo; Parquet sigue siendo recomendado.
        df = leer_excel_snii(path)

    else:
        raise ValueError("Formato de archivo no compatible.")

    return preparar_base(df)


@st.cache_data(show_spinner="Procesando el archivo cargado…")
def cargar_desde_upload(
    contenido: bytes,
    nombre: str,
) -> pd.DataFrame:
    """Carga un archivo proporcionado desde la interfaz."""
    from io import BytesIO

    buffer = BytesIO(contenido)
    extension = Path(nombre).suffix.lower()

    if extension == ".parquet":
        df = pd.read_parquet(buffer, engine="pyarrow")

    elif extension in {".xlsx", ".xls"}:
        df = leer_excel_snii(buffer)

    else:
        raise ValueError("Carga un archivo .parquet o .xlsx.")

    # Se conservan todas las columnas disponibles del archivo cargado.
    return preparar_base(df.copy())


def localizar_archivo() -> Path | None:
    for ruta in DATA_CANDIDATES:
        if ruta.exists():
            return ruta
    return None


def obtener_base() -> tuple[pd.DataFrame, str]:
    """Localiza la base del repositorio o solicita una carga manual."""
    ruta = localizar_archivo()

    if ruta is not None:
        return cargar_desde_ruta(str(ruta)), ruta.name

    st.sidebar.warning(
        "No se encontró el master dentro del repositorio. "
        "Carga temporalmente un archivo para probar la aplicación."
    )

    archivo = st.sidebar.file_uploader(
        "Cargar master",
        type=["parquet", "xlsx"],
    )

    if archivo is None:
        st.info(
            "Coloca `SNII_MASTER_v2_APP.xlsx` dentro de "
            "la carpeta `data/` del repositorio o carga el archivo aquí."
        )
        st.stop()

    return (
        cargar_desde_upload(archivo.getvalue(), archivo.name),
        archivo.name,
    )


# ============================================================
# FILTROS Y RESÚMENES
# ============================================================

def filtrar_ambito(
    df: pd.DataFrame,
    ambito: str,
    seleccion: str | None,
) -> pd.DataFrame:
    if ambito == "Nacional" or seleccion is None:
        return df

    columna = (
        "ENTIDAD_FEDERATIVA_ANUAL"
        if ambito == "Por estado"
        else "INSTITUCION_ANUAL"
    )

    if columna not in df.columns:
        return df.iloc[0:0].copy()

    return df.loc[df[columna].eq(seleccion)].copy()


@st.cache_data(show_spinner=False)
def serie_personas_anual(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("AÑO", dropna=False)["ID_PERSONA_EXACTA"]
        .nunique()
        .reindex(range(2000, 2026), fill_value=0)
        .rename("PERSONAS")
        .reset_index()
    )



@st.cache_data(show_spinner=False)
def calcular_movimientos_anuales(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clasifica cada registro persona-año como:
    - Nuevo ingreso
    - Permanencia
    - Reingreso

    También calcula salidas observadas.
    """

    base = (
        df[
            [
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        ]
        .dropna()
        .drop_duplicates()
        .sort_values(
            [
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        )
        .copy()
    )

    base["AÑO"] = pd.to_numeric(
        base["AÑO"],
        errors="coerce",
    ).astype("Int64")

    base = base.loc[
        base["AÑO"].between(
            2000,
            2025,
            inclusive="both",
        )
    ].copy()

    base["PRIMER_AÑO_OBSERVADO"] = (
        base.groupby(
            "ID_PERSONA_EXACTA"
        )["AÑO"]
        .transform("min")
    )

    base["AÑO_ANTERIOR_OBSERVADO"] = (
        base.groupby(
            "ID_PERSONA_EXACTA"
        )["AÑO"]
        .shift()
    )

    es_nuevo = base["AÑO"].eq(
        base["PRIMER_AÑO_OBSERVADO"]
    )

    es_permanencia = (
        base["AÑO_ANTERIOR_OBSERVADO"].notna()
        & base["AÑO"].eq(
            base["AÑO_ANTERIOR_OBSERVADO"] + 1
        )
    )

    base["TIPO_MOVIMIENTO"] = np.select(
        [
            es_nuevo,
            es_permanencia,
        ],
        [
            "Nuevo ingreso",
            "Permanencia",
        ],
        default="Reingreso",
    )

    movimientos = (
        base.groupby(
            [
                "AÑO",
                "TIPO_MOVIMIENTO",
            ]
        )["ID_PERSONA_EXACTA"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(
            range(2000, 2026),
            fill_value=0,
        )
        .reset_index()
    )

    for columna in [
        "Nuevo ingreso",
        "Permanencia",
        "Reingreso",
    ]:
        if columna not in movimientos.columns:
            movimientos[columna] = 0

    movimientos["TOTAL_ACTIVO"] = (
        movimientos[
            [
                "Nuevo ingreso",
                "Permanencia",
                "Reingreso",
            ]
        ]
        .sum(axis=1)
        .astype(int)
    )

    personas_por_anio = {
        int(anio): set(
            grupo["ID_PERSONA_EXACTA"].astype(str)
        )
        for anio, grupo in base.groupby("AÑO")
    }

    salidas = []

    for anio in range(2000, 2026):
        if anio == 2000:
            numero_salidas = 0
        else:
            personas_previas = personas_por_anio.get(
                anio - 1,
                set(),
            )
            personas_actuales = personas_por_anio.get(
                anio,
                set(),
            )
            numero_salidas = len(
                personas_previas - personas_actuales
            )

        salidas.append(
            {
                "AÑO": anio,
                "SALIDAS_OBSERVADAS": numero_salidas,
            }
        )

    movimientos = movimientos.merge(
        pd.DataFrame(salidas),
        on="AÑO",
        how="left",
    )

    movimientos["CRECIMIENTO_NETO"] = (
        movimientos["TOTAL_ACTIVO"]
        .diff()
        .fillna(0)
        .astype(int)
    )

    detalle = base[
        [
            "ID_PERSONA_EXACTA",
            "AÑO",
            "TIPO_MOVIMIENTO",
        ]
    ].copy()

    return movimientos, detalle


@st.cache_data(show_spinner=False)
def calcular_evolucion_sexo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula personas únicas por año y sexo consolidado."""

    if "SEXO_CONSOLIDADO" not in df.columns:
        return pd.DataFrame(
            columns=[
                "AÑO",
                "SEXO",
                "PERSONAS",
            ]
        )

    base = df[
        [
            "ID_PERSONA_EXACTA",
            "AÑO",
            "SEXO_CONSOLIDADO",
        ]
    ].copy()

    base["SEXO"] = (
        base["SEXO_CONSOLIDADO"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(
            {
                "F": "MUJER",
                "FEMENINO": "MUJER",
                "M": "HOMBRE",
                "MASCULINO": "HOMBRE",
            }
        )
    )

    base["SEXO"] = base["SEXO"].where(
        base["SEXO"].isin(
            [
                "MUJER",
                "HOMBRE",
            ]
        ),
        "NO DETERMINADO",
    )

    evolucion = (
        base.dropna(
            subset=[
                "ID_PERSONA_EXACTA",
                "AÑO",
            ]
        )
        .groupby(
            [
                "AÑO",
                "SEXO",
            ]
        )["ID_PERSONA_EXACTA"]
        .nunique()
        .rename("PERSONAS")
        .reset_index()
    )

    años = pd.DataFrame(
        {
            "AÑO": range(2000, 2026)
        }
    )

    sexos = pd.DataFrame(
        {
            "SEXO": [
                "MUJER",
                "HOMBRE",
                "NO DETERMINADO",
            ]
        }
    )

    estructura = años.merge(
        sexos,
        how="cross",
    )

    evolucion = estructura.merge(
        evolucion,
        on=[
            "AÑO",
            "SEXO",
        ],
        how="left",
    )

    evolucion["PERSONAS"] = (
        evolucion["PERSONAS"]
        .fillna(0)
        .astype(int)
    )

    return evolucion


def resumen_categoria(
    df: pd.DataFrame,
    columna: str,
    etiqueta: str,
) -> pd.DataFrame:
    if columna not in df.columns:
        return pd.DataFrame(columns=[etiqueta, "PERSONAS"])

    return (
        df.dropna(subset=[columna])
        .groupby(columna)["ID_PERSONA_EXACTA"]
        .nunique()
        .sort_values(ascending=False)
        .rename("PERSONAS")
        .reset_index()
        .rename(columns={columna: etiqueta})
    )


def ultimo_valor_no_nulo(
    df_persona: pd.DataFrame,
    columna: str,
) -> str:
    if columna not in df_persona.columns:
        return "Sin información"

    valores = (
        df_persona.sort_values("AÑO")[columna]
        .dropna()
        .astype("string")
        .str.strip()
    )

    return str(valores.iloc[-1]) if not valores.empty else "Sin información"


# ============================================================
# MODELOS DE PROYECCIÓN
# ============================================================

def modelo_logistico(
    x: np.ndarray,
    capacidad: float,
    tasa: float,
    punto_medio: float,
) -> np.ndarray:
    return capacidad / (1 + np.exp(-tasa * (x - punto_medio)))


def ajustar_lineal(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    modelo = LinearRegression()
    modelo.fit(x_train.reshape(-1, 1), y_train)
    return modelo.predict(x_pred.reshape(-1, 1))


def ajustar_polinomial(
    grado: int,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def _ajustar(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_pred: np.ndarray,
    ) -> np.ndarray:
        transformador = PolynomialFeatures(
            degree=grado,
            include_bias=False,
        )
        x_train_poly = transformador.fit_transform(x_train.reshape(-1, 1))
        x_pred_poly = transformador.transform(x_pred.reshape(-1, 1))

        modelo = LinearRegression()
        modelo.fit(x_train_poly, y_train)
        return modelo.predict(x_pred_poly)

    return _ajustar


def ajustar_logistico(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    capacidad_inicial = max(float(y_train.max()) * 1.25, 1.0)
    tasa_inicial = 0.15
    punto_medio_inicial = float(np.median(x_train))

    parametros, _ = curve_fit(
        modelo_logistico,
        x_train,
        y_train,
        p0=[
            capacidad_inicial,
            tasa_inicial,
            punto_medio_inicial,
        ],
        bounds=(
            [max(float(y_train.max()), 1.0), 0.0001, x_train.min() - 30],
            [max(float(y_train.max()) * 20, 10.0), 5.0, x_train.max() + 30],
        ),
        maxfev=50000,
    )

    return modelo_logistico(x_pred, *parametros)


def comparar_modelos(
    serie: pd.DataFrame,
    horizonte: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compara modelos con validación temporal y proyecta el mejor."""
    datos = serie.loc[serie["PERSONAS"].gt(0)].copy()

    if len(datos) < 8:
        raise ValueError(
            "Se requieren al menos ocho observaciones anuales con datos."
        )

    x = datos["AÑO"].astype(float).to_numpy()
    y = datos["PERSONAS"].astype(float).to_numpy()

    validacion = min(4, max(2, len(datos) // 5))
    x_train, x_test = x[:-validacion], x[-validacion:]
    y_train, y_test = y[:-validacion], y[-validacion:]

    modelos: dict[str, Callable] = {
        "Lineal": ajustar_lineal,
        "Polinomial grado 2": ajustar_polinomial(2),
        "Polinomial grado 3": ajustar_polinomial(3),
        "Logístico": ajustar_logistico,
    }

    resultados = []
    modelos_validos: dict[str, Callable] = {}

    for nombre, funcion in modelos.items():
        try:
            pred_test = np.maximum(
                funcion(x_train, y_train, x_test),
                0,
            )

            mae = mean_absolute_error(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            resultados.append(
                {
                    "MODELO": nombre,
                    "MAE": mae,
                    "RMSE": rmse,
                }
            )
            modelos_validos[nombre] = funcion

        except Exception as error:
            warnings.warn(f"{nombre} no pudo ajustarse: {error}")

    if not resultados:
        raise ValueError("Ningún modelo pudo ajustarse a la serie.")

    evaluacion = (
        pd.DataFrame(resultados)
        .sort_values(["RMSE", "MAE"])
        .reset_index(drop=True)
    )

    mejor_modelo = str(evaluacion.iloc[0]["MODELO"])
    funcion_mejor = modelos_validos[mejor_modelo]

    años_futuros = np.arange(
        int(x.max()) + 1,
        int(x.max()) + horizonte + 1,
        dtype=float,
    )

    x_completo = np.concatenate([x, años_futuros])
    pred_completa = np.maximum(
        funcion_mejor(x, y, x_completo),
        0,
    )

    proyeccion = pd.DataFrame(
        {
            "AÑO": x_completo.astype(int),
            "VALOR_MODELO": pred_completa,
            "TIPO": np.where(
                x_completo <= x.max(),
                "Ajuste histórico",
                "Proyección",
            ),
        }
    )

    observados = datos[["AÑO", "PERSONAS"]].copy()
    proyeccion = proyeccion.merge(
        observados,
        on="AÑO",
        how="left",
    )

    return evaluacion, proyeccion, mejor_modelo


# ============================================================
# MÓDULO 1: PANORAMA ACTUAL
# ============================================================

def render_panorama(df: pd.DataFrame) -> None:
    st.header("1. Panorama de la investigación en México")
    st.markdown(
        '<p class="snii-subtitle">'
        "Consulta nacional, por entidad federativa o por institución."
        "</p>",
        unsafe_allow_html=True,
    )

    controles = st.columns([1.1, 1.7, 1])

    with controles[0]:
        ambito = st.selectbox(
            "Nivel de consulta",
            ["Nacional", "Por estado", "Por institución"],
        )

    seleccion = None

    with controles[1]:
        if ambito == "Por estado":
            if "ENTIDAD_FEDERATIVA_ANUAL" not in df.columns:
                st.warning("La base no contiene entidad federativa.")
                return

            opciones = sorted(
                df["ENTIDAD_FEDERATIVA_ANUAL"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            seleccion = st.selectbox("Entidad federativa", opciones)

        elif ambito == "Por institución":
            if "INSTITUCION_ANUAL" not in df.columns:
                st.warning("La base no contiene institución.")
                return

            opciones = sorted(
                df["INSTITUCION_ANUAL"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            seleccion = st.selectbox(
                "Universidad o centro de investigación",
                opciones,
            )

        else:
            st.text_input(
                "Cobertura",
                value="Estados Unidos Mexicanos",
                disabled=True,
            )

    base_ambito = filtrar_ambito(df, ambito, seleccion)

    with controles[2]:
        años = sorted(
            base_ambito["AÑO"].dropna().astype(int).unique().tolist()
        )

        if not años:
            st.warning("No hay registros para la selección.")
            return

        año = st.selectbox(
            "Año de referencia",
            años,
            index=len(años) - 1,
        )

    actual = base_ambito.loc[base_ambito["AÑO"].eq(año)].copy()
    serie = serie_personas_anual(base_ambito)

    total_actual = actual["ID_PERSONA_EXACTA"].nunique()
    total_previo = int(
        serie.loc[serie["AÑO"].eq(año - 1), "PERSONAS"].sum()
    )
    delta = total_actual - total_previo if total_previo else None

    mujeres = (
        actual.loc[
            actual.get(
                "SEXO_CONSOLIDADO",
                pd.Series(index=actual.index, dtype="string"),
            )
            .astype("string")
            .str.upper()
            .eq("MUJER"),
            "ID_PERSONA_EXACTA",
        ].nunique()
        if "SEXO_CONSOLIDADO" in actual.columns
        else 0
    )

    porcentaje_mujeres = (
        mujeres / total_actual * 100
        if total_actual
        else 0
    )

    instituciones = (
        actual["INSTITUCION_ANUAL"].nunique(dropna=True)
        if "INSTITUCION_ANUAL" in actual.columns
        else 0
    )

    entidades = (
        actual["ENTIDAD_FEDERATIVA_ANUAL"].nunique(dropna=True)
        if "ENTIDAD_FEDERATIVA_ANUAL" in actual.columns
        else 0
    )

    metricas = st.columns(4)

    metricas[0].metric(
        f"Personas en {año}",
        f"{total_actual:,}",
        delta=f"{delta:+,}" if delta is not None else None,
    )
    metricas[1].metric(
        "Participación de mujeres",
        f"{porcentaje_mujeres:.1f}%",
        help=f"{mujeres:,} mujeres con sexo consolidado.",
    )
    metricas[2].metric("Instituciones", f"{instituciones:,}")
    metricas[3].metric("Entidades representadas", f"{entidades:,}")

    # ========================================================
    # EVOLUCIÓN HISTÓRICA Y MOVIMIENTOS
    # ========================================================

    st.subheader(
        "Evolución histórica y movimientos anuales"
    )

    movimientos, _ = calcular_movimientos_anuales(
        base_ambito
    )

    movimientos_largos = movimientos.melt(
        id_vars=[
            "AÑO",
            "SALIDAS_OBSERVADAS",
            "TOTAL_ACTIVO",
            "CRECIMIENTO_NETO",
        ],
        value_vars=[
            "Permanencia",
            "Nuevo ingreso",
            "Reingreso",
        ],
        var_name="MOVIMIENTO",
        value_name="PERSONAS",
    )

    fig_movimientos = px.bar(
        movimientos_largos,
        x="AÑO",
        y="PERSONAS",
        color="MOVIMIENTO",
        barmode="stack",
        category_orders={
            "MOVIMIENTO": [
                "Permanencia",
                "Nuevo ingreso",
                "Reingreso",
            ]
        },
        labels={
            "AÑO": "Año",
            "PERSONAS": "Personas",
            "MOVIMIENTO": "Condición anual",
        },
    )

    fig_movimientos.add_trace(
        go.Scatter(
            x=movimientos["AÑO"],
            y=movimientos["SALIDAS_OBSERVADAS"],
            mode="lines+markers",
            name="Salidas observadas",
            line=dict(
                width=3,
                dash="dot",
            ),
            marker=dict(size=7),
            hovertemplate=(
                "<b>Año %{x}</b><br>"
                "Salidas observadas: %{y:,}"
                "<extra></extra>"
            ),
        )
    )

    fig_movimientos.update_layout(
        height=520,
        hovermode="x unified",
        legend_title_text="Movimiento",
        xaxis_title="Año",
        yaxis_title="Personas",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
    )

    st.plotly_chart(
        fig_movimientos,
        width="stretch",
        key="grafica_movimientos_anuales",
    )

    st.caption(
        "La barra anual suma permanencias, nuevos ingresos y "
        "reingresos. La línea muestra personas presentes en el "
        "año anterior que ya no aparecen en el año actual. "
        "La ausencia puede representar una salida real o una "
        "limitación de cobertura de la fuente."
    )

    st.subheader(
        "Evolución histórica por sexo"
    )

    evolucion_sexo = calcular_evolucion_sexo(
        base_ambito
    )

    if evolucion_sexo.empty:
        st.info(
            "No hay información de sexo disponible para "
            "construir la evolución histórica."
        )

    else:
        fig_sexo_historico = px.bar(
            evolucion_sexo,
            x="AÑO",
            y="PERSONAS",
            color="SEXO",
            barmode="stack",
            category_orders={
                "SEXO": [
                    "MUJER",
                    "HOMBRE",
                    "NO DETERMINADO",
                ]
            },
            labels={
                "AÑO": "Año",
                "PERSONAS": "Personas",
                "SEXO": "Sexo",
            },
        )

        fig_sexo_historico.update_layout(
            height=500,
            hovermode="x unified",
            legend_title_text="Sexo",
            xaxis_title="Año",
            yaxis_title="Personas",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )

        st.plotly_chart(
            fig_sexo_historico,
            width="stretch",
            key="grafica_evolucion_sexo",
        )

        st.caption(
            "Cada barra representa a las personas únicas "
            "registradas en el año, distribuidas según el "
            "sexo consolidado disponible en la base."
        )

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader(f"Distribución por sexo · {año}")
        sexo = resumen_categoria(actual, "SEXO_CONSOLIDADO", "SEXO")

        if sexo.empty:
            st.info("No hay información de sexo para esta selección.")
        else:
            fig_sexo = px.donut(
                sexo,
                names="SEXO",
                values="PERSONAS",
                hole=0.58,
            )
            fig_sexo.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_sexo, width="stretch")

    with derecha:
        st.subheader(f"Distribución por nivel SNII · {año}")
        nivel = resumen_categoria(
            actual,
            "NIVEL_SNII_ETIQUETA",
            "NIVEL",
        )

        if nivel.empty:
            st.info("No hay información homologada de nivel.")
        else:
            fig_nivel = px.bar(
                nivel,
                x="NIVEL",
                y="PERSONAS",
                labels={"PERSONAS": "Personas"},
            )
            fig_nivel.update_layout(
                height=390,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_nivel, width="stretch")

    izquierda, derecha = st.columns(2)

    with izquierda:
        st.subheader(f"Clasificación académica · {año}")
        stem = resumen_categoria(
            actual,
            "CLASIFICACION_STEM_ANUAL",
            "CLASIFICACIÓN",
        )

        if stem.empty:
            st.info("No hay clasificación académica disponible.")
        else:
            fig_stem = px.bar(
                stem.sort_values("PERSONAS"),
                x="PERSONAS",
                y="CLASIFICACIÓN",
                orientation="h",
            )
            fig_stem.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_stem, width="stretch")

    with derecha:
        st.subheader(f"Principales instituciones · {año}")
        instituciones_df = resumen_categoria(
            actual,
            "INSTITUCION_ANUAL",
            "INSTITUCIÓN",
        ).head(15)

        if instituciones_df.empty:
            st.info("No hay información institucional.")
        else:
            fig_inst = px.bar(
                instituciones_df.sort_values("PERSONAS"),
                x="PERSONAS",
                y="INSTITUCIÓN",
                orientation="h",
            )
            fig_inst.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_inst, width="stretch")

    with st.expander("Ver datos utilizados"):
        st.dataframe(
            actual,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# MÓDULO 2: PROYECCIONES
# ============================================================

def render_proyecciones(df: pd.DataFrame) -> None:
    st.header("2. Proyecciones de la población SNII")
    st.markdown(
        '<div class="snii-note">'
        "Las proyecciones son ejercicios estadísticos exploratorios. "
        "El sistema compara modelos mediante validación temporal y "
        "selecciona el de menor RMSE."
        "</div>",
        unsafe_allow_html=True,
    )

    controles = st.columns([1, 1.6, 1])

    with controles[0]:
        ambito = st.selectbox(
            "Cobertura de la proyección",
            ["Nacional", "Por estado", "Por institución"],
            key="proy_ambito",
        )

    seleccion = None

    with controles[1]:
        if ambito == "Por estado":
            opciones = sorted(
                df.get(
                    "ENTIDAD_FEDERATIVA_ANUAL",
                    pd.Series(dtype="string"),
                )
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            seleccion = st.selectbox(
                "Entidad federativa",
                opciones,
                key="proy_estado",
            )

        elif ambito == "Por institución":
            opciones = sorted(
                df.get(
                    "INSTITUCION_ANUAL",
                    pd.Series(dtype="string"),
                )
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            seleccion = st.selectbox(
                "Institución",
                opciones,
                key="proy_inst",
            )

        else:
            st.text_input(
                "Cobertura",
                value="Nacional",
                disabled=True,
                key="proy_nacional",
            )

    with controles[2]:
        horizonte = st.slider(
            "Horizonte",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )

    base_ambito = filtrar_ambito(df, ambito, seleccion)
    serie = serie_personas_anual(base_ambito)

    if serie["PERSONAS"].gt(0).sum() < 8:
        st.warning(
            "La serie no tiene suficientes años con observaciones "
            "para construir una proyección robusta."
        )
        return

    try:
        evaluacion, proyeccion, mejor_modelo = comparar_modelos(
            serie,
            horizonte,
        )
    except Exception as error:
        st.error(f"No fue posible construir la proyección: {error}")
        return

    mejor = evaluacion.iloc[0]

    metricas = st.columns(3)
    metricas[0].metric("Modelo seleccionado", mejor_modelo)
    metricas[1].metric("RMSE de validación", f"{mejor['RMSE']:,.1f}")
    metricas[2].metric("MAE de validación", f"{mejor['MAE']:,.1f}")

    fig = go.Figure()

    observados = proyeccion.loc[proyeccion["PERSONAS"].notna()]

    fig.add_trace(
        go.Scatter(
            x=observados["AÑO"],
            y=observados["PERSONAS"],
            mode="lines+markers",
            name="Observado",
        )
    )

    ajuste = proyeccion.loc[proyeccion["TIPO"].eq("Ajuste histórico")]
    futuro = proyeccion.loc[proyeccion["TIPO"].eq("Proyección")]

    fig.add_trace(
        go.Scatter(
            x=ajuste["AÑO"],
            y=ajuste["VALOR_MODELO"],
            mode="lines",
            name=f"Ajuste: {mejor_modelo}",
            line=dict(dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=futuro["AÑO"],
            y=futuro["VALOR_MODELO"],
            mode="lines+markers",
            name="Proyección",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Año",
        yaxis_title="Personas",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, width="stretch")

    izquierda, derecha = st.columns([1, 1.35])

    with izquierda:
        st.subheader("Comparación de modelos")
        tabla_evaluacion = evaluacion.copy()
        tabla_evaluacion[["MAE", "RMSE"]] = tabla_evaluacion[
            ["MAE", "RMSE"]
        ].round(2)

        st.dataframe(
            tabla_evaluacion,
            width="stretch",
            hide_index=True,
        )

    with derecha:
        st.subheader("Valores proyectados")
        tabla_futuro = futuro[["AÑO", "VALOR_MODELO"]].copy()
        tabla_futuro["VALOR_MODELO"] = (
            tabla_futuro["VALOR_MODELO"].round().astype(int)
        )
        tabla_futuro = tabla_futuro.rename(
            columns={"VALOR_MODELO": "PERSONAS_PROYECTADAS"}
        )

        st.dataframe(
            tabla_futuro,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# MÓDULO 3: INVESTIGADOR
# ============================================================

def render_investigador(df: pd.DataFrame) -> None:
    st.header("3. Historial del investigador")
    st.markdown(
        '<p class="snii-subtitle">'
        "Consulta la trayectoria anual registrada entre 2000 y 2025."
        "</p>",
        unsafe_allow_html=True,
    )

    if "NOMBRE_INVESTIGADOR" not in df.columns:
        st.warning("La base no contiene NOMBRE_INVESTIGADOR.")
        return

    catalogo = (
        df[
            [
                "ID_PERSONA_EXACTA",
                "NOMBRE_INVESTIGADOR",
                *(
                    ["CVU_REFERENCIA"]
                    if "CVU_REFERENCIA" in df.columns
                    else []
                ),
            ]
        ]
        .drop_duplicates("ID_PERSONA_EXACTA")
        .dropna(subset=["NOMBRE_INVESTIGADOR"])
        .sort_values("NOMBRE_INVESTIGADOR")
        .copy()
    )

    catalogo["ETIQUETA"] = (
        catalogo["NOMBRE_INVESTIGADOR"].astype(str)
        + " · "
        + catalogo["ID_PERSONA_EXACTA"].astype(str)
    )

    busqueda = st.text_input(
        "Buscar por nombre, CVU o ID",
        placeholder="Escribe al menos tres caracteres…",
    ).strip()

    if len(busqueda) < 3:
        st.info("Escribe al menos tres caracteres para buscar.")
        return

    mascara = (
        catalogo["ETIQUETA"]
        .astype(str)
        .str.contains(busqueda, case=False, na=False, regex=False)
    )

    if "CVU_REFERENCIA" in catalogo.columns:
        mascara = mascara | (
            catalogo["CVU_REFERENCIA"]
            .astype(str)
            .str.contains(busqueda, case=False, na=False, regex=False)
        )

    coincidencias = catalogo.loc[mascara].head(100)

    if coincidencias.empty:
        st.warning("No se encontraron coincidencias.")
        return

    etiqueta = st.selectbox(
        "Selecciona una persona",
        coincidencias["ETIQUETA"].tolist(),
    )

    id_persona = coincidencias.loc[
        coincidencias["ETIQUETA"].eq(etiqueta),
        "ID_PERSONA_EXACTA",
    ].iloc[0]

    historial = (
        df.loc[df["ID_PERSONA_EXACTA"].eq(id_persona)]
        .sort_values("AÑO")
        .copy()
    )

    nombre = ultimo_valor_no_nulo(
        historial,
        "NOMBRE_INVESTIGADOR",
    )
    sexo = ultimo_valor_no_nulo(historial, "SEXO_CONSOLIDADO")
    institucion = ultimo_valor_no_nulo(historial, "INSTITUCION_ANUAL")
    entidad = ultimo_valor_no_nulo(
        historial,
        "ENTIDAD_FEDERATIVA_ANUAL",
    )

    primer_año = int(historial["AÑO"].min())
    ultimo_año = int(historial["AÑO"].max())
    años_presentes = historial["AÑO"].nunique()

    st.subheader(nombre)

    metricas = st.columns(4)
    metricas[0].metric("Primer año observado", primer_año)
    metricas[1].metric("Último año observado", ultimo_año)
    metricas[2].metric("Años con registro", años_presentes)
    metricas[3].metric("Sexo consolidado", sexo)

    st.markdown(
        f"""
        **Institución más reciente:** {institucion}  
        **Entidad más reciente:** {entidad}
        """
    )

    if "NIVEL_SNII_STD" in historial.columns:
        historial_nivel = historial.dropna(subset=["NIVEL_SNII_STD"]).copy()

        if not historial_nivel.empty:
            nivel_maximo = int(historial_nivel["NIVEL_SNII_STD"].max())
            nivel_ultimo = ultimo_valor_no_nulo(
                historial_nivel,
                "NIVEL_SNII_ETIQUETA",
            )

            cols = st.columns(3)
            cols[0].metric("Nivel más reciente", nivel_ultimo)
            cols[1].metric("Código máximo histórico", nivel_maximo)

            años_posibles = ultimo_año - primer_año + 1
            continuidad = (
                años_presentes / años_posibles * 100
                if años_posibles > 0
                else 0
            )
            cols[2].metric("Continuidad observada", f"{continuidad:.1f}%")

    st.subheader("Trayectoria de nivel")

    if "NIVEL_SNII_STD" in historial.columns:
        datos_nivel = historial.dropna(subset=["NIVEL_SNII_STD"])

        if datos_nivel.empty:
            st.info("No existe nivel homologado para esta persona.")
        else:
            fig_nivel = px.line(
                datos_nivel,
                x="AÑO",
                y="NIVEL_SNII_STD",
                markers=True,
                hover_data=[
                    col
                    for col in [
                        "NIVEL_SNII_ETIQUETA",
                        "INSTITUCION_ANUAL",
                        "ENTIDAD_FEDERATIVA_ANUAL",
                    ]
                    if col in datos_nivel.columns
                ],
            )
            fig_nivel.update_yaxes(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=[
                    "Candidato",
                    "Nivel I",
                    "Nivel II",
                    "Nivel III",
                    "Emérito",
                ],
            )
            fig_nivel.update_layout(
                height=430,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_nivel, width="stretch")

    st.subheader("Estimación exploratoria de continuidad")

    años_posibles = ultimo_año - primer_año + 1
    continuidad_historica = (
        años_presentes / años_posibles
        if años_posibles > 0
        else 0
    )

    últimos_cinco = set(range(max(primer_año, ultimo_año - 4), ultimo_año + 1))
    observados_ultimos = set(historial["AÑO"].astype(int).tolist())
    continuidad_reciente = (
        len(últimos_cinco & observados_ultimos)
        / len(últimos_cinco)
        if últimos_cinco
        else 0
    )

    probabilidad_exploratoria = (
        0.4 * continuidad_historica
        + 0.6 * continuidad_reciente
    ) * 100

    st.metric(
        "Índice exploratorio de presencia futura",
        f"{probabilidad_exploratoria:.1f}%",
        help=(
            "No es todavía un modelo predictivo entrenado. Combina "
            "continuidad histórica y presencia reciente como referencia."
        ),
    )

    st.caption(
        "La predicción individual formal se incorporará después de "
        "construir variables de cohorte, promociones, interrupciones "
        "y permanencia por nivel."
    )

    columnas_historial = [
        col
        for col in [
            "AÑO",
            "NIVEL_SNII_ETIQUETA",
            "INSTITUCION_ANUAL",
            "DEPENDENCIA_ANUAL",
            "ENTIDAD_FEDERATIVA_ANUAL",
            "AREA_DEL_CONOCIMIENTO_ANUAL",
            "DISCIPLINA_ANUAL",
            "CLASIFICACION_STEM_ANUAL",
        ]
        if col in historial.columns
    ]

    with st.expander("Ver historial anual"):
        st.dataframe(
            historial[columnas_historial],
            width="stretch",
            hide_index=True,
        )



# ============================================================
# MÓDULO 4: LABORATORIO DE VISUALIZACIÓN
# ============================================================

# Catálogo analítico curado.
# Nombres, apellidos, CVU, Nobilis e identificadores quedan
# exclusivamente en el módulo Historial del investigador.

VARIABLES_ANALITICAS = {
    "Sexo": {
        "familia": "Características de la persona",
        "tipo": "categorica",
        "alias": [
            "SEXO_CONSOLIDADO",
            "SEXO",
        ],
    },
    "Nivel SNII": {
        "familia": "Reconocimiento SNII",
        "tipo": "categorica",
        "alias": [
            "NIVEL_SNII_ETIQUETA",
            "NIVEL_SNII",
            "NIVEL_ANUAL",
            "NIVEL",
        ],
    },
    "Código de nivel SNII": {
        "familia": "Reconocimiento SNII",
        "tipo": "ordinal",
        "alias": [
            "NIVEL_SNII_STD",
            "CODIGO_NIVEL_SNII",
        ],
    },
    "STEM / No STEM": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "GRUPO_STEM_BINARIO",
            "STEM_BINARIO",
        ],
    },
    "Clasificación STEM ampliada": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "CLASIFICACION_STEM_ANUAL",
            "CLASIFICACION_STEM",
        ],
    },
    "Área del conocimiento": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "AREA_DEL_CONOCIMIENTO_ANUAL",
            "AREA_DEL_CONOCIMIENTO",
            "AREA_ANUAL",
        ],
    },
    "Campo del conocimiento": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "CAMPO_DEL_CONOCIMIENTO_ANUAL",
            "CAMPO_DEL_CONOCIMIENTO",
            "CAMPO_ANUAL",
        ],
    },
    "Disciplina": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "DISCIPLINA_ANUAL",
            "DISCIPLINA",
        ],
    },
    "Subdisciplina": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "SUBDISCIPLINA_ANUAL",
            "SUBDISCIPLINA",
        ],
    },
    "Especialidad": {
        "familia": "Clasificación académica",
        "tipo": "categorica",
        "alias": [
            "ESPECIALIDAD_ANUAL",
            "ESPECIALIDAD",
        ],
    },
    "Institución": {
        "familia": "Ubicación e institución",
        "tipo": "categorica",
        "alias": [
            "INSTITUCION_ANUAL",
            "INSTITUCION",
        ],
    },
    "Dependencia": {
        "familia": "Ubicación e institución",
        "tipo": "categorica",
        "alias": [
            "DEPENDENCIA_ANUAL",
            "DEPENDENCIA",
        ],
    },
    "Subdependencia": {
        "familia": "Ubicación e institución",
        "tipo": "categorica",
        "alias": [
            "SUBDEPENDENCIA_ANUAL",
            "SUBDEPENDENCIA",
        ],
    },
    "Entidad federativa": {
        "familia": "Ubicación e institución",
        "tipo": "categorica",
        "alias": [
            "ENTIDAD_FEDERATIVA_ANUAL",
            "ENTIDAD_FEDERATIVA",
            "ESTADO",
        ],
    },
    "País": {
        "familia": "Ubicación e institución",
        "tipo": "categorica",
        "alias": [
            "PAIS_ANUAL",
            "PAIS",
        ],
    },
    "Años con registro": {
        "familia": "Trayectoria",
        "tipo": "numerica",
        "alias": [
            "NUMERO_AÑOS_PRESENTE",
            "NUMERO_ANIOS_PRESENTE",
        ],
    },
    "Antigüedad acumulada": {
        "familia": "Trayectoria",
        "tipo": "numerica",
        "alias": [
            "ANTIGUEDAD_ACUMULADA_AÑOS",
            "ANTIGUEDAD_ACUMULADA_ANIOS",
        ],
    },
    "Primer año observado": {
        "familia": "Trayectoria",
        "tipo": "numerica",
        "alias": [
            "PRIMER_AÑO",
            "PRIMER_ANIO",
        ],
    },
    "Último año observado": {
        "familia": "Trayectoria",
        "tipo": "numerica",
        "alias": [
            "ULTIMO_AÑO",
            "ULTIMO_ANIO",
        ],
    },
    "Vigente en 2025": {
        "familia": "Trayectoria",
        "tipo": "binaria",
        "alias": [
            "ESTA_VIGENTE_EN_2025",
            "VIGENTE_2025",
        ],
    },
    "Es primer año": {
        "familia": "Trayectoria",
        "tipo": "binaria",
        "alias": [
            "ES_PRIMER_AÑO_PERSONA",
            "ES_PRIMER_ANIO_PERSONA",
        ],
    },
    "Es último año": {
        "familia": "Trayectoria",
        "tipo": "binaria",
        "alias": [
            "ES_ULTIMO_AÑO_PERSONA",
            "ES_ULTIMO_ANIO_PERSONA",
        ],
    },
    "Porcentaje de completitud": {
        "familia": "Calidad de datos",
        "tipo": "numerica",
        "alias": [
            "PORCENTAJE_COMPLETITUD_CLAVE",
            "PORCENTAJE_COMPLETITUD",
        ],
    },
    "Campos clave disponibles": {
        "familia": "Calidad de datos",
        "tipo": "numerica",
        "alias": [
            "NUMERO_CAMPOS_CLAVE_DISPONIBLES",
        ],
    },
    "Campos clave faltantes": {
        "familia": "Calidad de datos",
        "tipo": "numerica",
        "alias": [
            "NUMERO_CAMPOS_CLAVE_FALTANTES",
        ],
    },
    "Requiere revisión del master": {
        "familia": "Calidad de datos",
        "tipo": "binaria",
        "alias": [
            "REQUIERE_REVISION_MASTER",
        ],
    },
}


OBJETIVOS_LABORATORIO = {
    "Distribución": {
        "descripcion":
            "Conocer cómo se reparte una característica en la población.",
        "tipos_principales": [
            "categorica",
            "binaria",
            "ordinal",
            "numerica",
        ],
        "requiere_secundaria": False,
        "permite_secundaria": False,
        "tiempo_obligatorio": False,
    },
    "Comparación": {
        "descripcion":
            "Comparar una característica o medida entre dos o más grupos.",
        "tipos_principales": [
            "categorica",
            "binaria",
            "ordinal",
            "numerica",
        ],
        "requiere_secundaria": True,
        "permite_secundaria": True,
        "tiempo_obligatorio": False,
    },
    "Relación": {
        "descripcion":
            "Explorar la asociación entre dos variables.",
        "tipos_principales": [
            "numerica",
            "categorica",
            "binaria",
            "ordinal",
        ],
        "requiere_secundaria": True,
        "permite_secundaria": True,
        "tiempo_obligatorio": False,
    },
    "Evolución temporal": {
        "descripcion":
            "Observar cómo cambia una variable durante varios años.",
        "tipos_principales": [
            "categorica",
            "binaria",
            "ordinal",
            "numerica",
        ],
        "requiere_secundaria": False,
        "permite_secundaria": True,
        "tiempo_obligatorio": True,
    },
}


def resolver_columna_analitica(
    df: pd.DataFrame,
    variable: str,
) -> str | None:
    """Encuentra la primera columna existente y con datos."""

    for columna in VARIABLES_ANALITICAS[
        variable
    ]["alias"]:
        if (
            columna in df.columns
            and df[columna].notna().any()
        ):
            return columna

    return None


def catalogo_analitico_disponible(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el catálogo analítico disponible en el master."""

    filas = []

    for variable, especificacion in (
        VARIABLES_ANALITICAS.items()
    ):
        columna = resolver_columna_analitica(
            df,
            variable,
        )

        filas.append(
            {
                "VARIABLE": variable,
                "FAMILIA": especificacion["familia"],
                "TIPO": especificacion["tipo"],
                "COLUMNA": columna,
                "DISPONIBLE": columna is not None,
            }
        )

    return pd.DataFrame(
        filas
    )


def normalizar_binaria_laboratorio(
    serie: pd.Series,
) -> pd.Series:
    """Homologa variables binarias para visualización."""

    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
        .replace(
            {
                "TRUE": "Sí",
                "FALSE": "No",
                "1": "Sí",
                "0": "No",
                "SI": "Sí",
                "SÍ": "Sí",
                "NO": "No",
            }
        )
    )


def preparar_base_laboratorio(
    df: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """Prepara tipos para las variables analíticas disponibles."""

    base = df.copy()

    for _, fila in catalogo.loc[
        catalogo["DISPONIBLE"]
    ].iterrows():
        columna = fila["COLUMNA"]
        tipo = fila["TIPO"]

        if tipo in {
            "numerica",
            "ordinal",
        }:
            base[columna] = pd.to_numeric(
                base[columna],
                errors="coerce",
            )

        elif tipo == "binaria":
            base[columna] = normalizar_binaria_laboratorio(
                base[columna]
            )

        else:
            base[columna] = (
                base[columna]
                .astype("string")
                .str.strip()
            )

    return base


def tipo_reducido(
    tipo: str,
) -> str:
    """Reduce los tipos a numérico o categórico."""

    if tipo in {
        "categorica",
        "binaria",
        "ordinal",
    }:
        return "categorica"

    return "numerica"


def opciones_variable_principal(
    catalogo: pd.DataFrame,
    objetivo: str,
) -> list[str]:
    """Filtra las variables principales según el objetivo."""

    tipos = OBJETIVOS_LABORATORIO[
        objetivo
    ]["tipos_principales"]

    return (
        catalogo.loc[
            catalogo["DISPONIBLE"]
            & catalogo["TIPO"].isin(
                tipos
            )
        ]
        .sort_values(
            [
                "FAMILIA",
                "VARIABLE",
            ]
        )["VARIABLE"]
        .tolist()
    )


def opciones_variable_secundaria(
    catalogo: pd.DataFrame,
    objetivo: str,
    variable_principal: str,
) -> list[str]:
    """Devuelve variables secundarias compatibles con la principal."""

    tipo_principal = VARIABLES_ANALITICAS[
        variable_principal
    ]["tipo"]

    tipo_principal_reducido = tipo_reducido(
        tipo_principal
    )

    candidatas = catalogo.loc[
        catalogo["DISPONIBLE"]
        & catalogo["VARIABLE"].ne(
            variable_principal
        )
    ].copy()

    if objetivo == "Comparación":
        if tipo_principal_reducido == "numerica":
            candidatas = candidatas.loc[
                candidatas["TIPO"].isin(
                    [
                        "categorica",
                        "binaria",
                        "ordinal",
                    ]
                )
            ]
        else:
            candidatas = candidatas.loc[
                candidatas["TIPO"].isin(
                    [
                        "categorica",
                        "binaria",
                        "ordinal",
                    ]
                )
            ]

    elif objetivo == "Relación":
        if tipo_principal_reducido == "numerica":
            candidatas = candidatas.loc[
                candidatas["TIPO"].isin(
                    [
                        "numerica",
                        "categorica",
                        "binaria",
                        "ordinal",
                    ]
                )
            ]
        else:
            candidatas = candidatas.loc[
                candidatas["TIPO"].eq(
                    "numerica"
                )
            ]

    elif objetivo == "Evolución temporal":
        candidatas = candidatas.loc[
            candidatas["TIPO"].isin(
                [
                    "categorica",
                    "binaria",
                    "ordinal",
                ]
            )
        ]

    else:
        return []

    return (
        candidatas.sort_values(
            [
                "FAMILIA",
                "VARIABLE",
            ]
        )["VARIABLE"]
        .tolist()
    )


def recomendar_graficas_arbol(
    objetivo: str,
    variable_principal: str,
    variable_secundaria: str | None,
    usa_tiempo: bool,
) -> list[dict[str, object]]:
    """Recomienda gráficas y asigna una idoneidad."""

    tipo_1 = tipo_reducido(
        VARIABLES_ANALITICAS[
            variable_principal
        ]["tipo"]
    )

    tipo_2 = (
        tipo_reducido(
            VARIABLES_ANALITICAS[
                variable_secundaria
            ]["tipo"]
        )
        if variable_secundaria
        else None
    )

    recomendaciones = []

    def agregar(
        grafica: str,
        porcentaje: int,
        razon: str,
    ) -> None:
        recomendaciones.append(
            {
                "grafica": grafica,
                "porcentaje": porcentaje,
                "razon": razon,
            }
        )

    if objetivo == "Distribución":
        if tipo_1 == "categorica":
            agregar(
                "Dona",
                96,
                "Resume la composición de una variable categórica.",
            )
            agregar(
                "Barras",
                94,
                "Compara con mayor precisión el tamaño de las categorías.",
            )
        else:
            agregar(
                "Histograma",
                98,
                "Muestra la distribución de una variable numérica.",
            )
            agregar(
                "Boxplot",
                91,
                "Resume mediana, dispersión y valores atípicos.",
            )

    elif objetivo == "Comparación":
        if tipo_1 == "numerica":
            agregar(
                "Boxplot",
                97,
                "Compara una distribución numérica entre grupos.",
            )
            agregar(
                "Violin plot",
                89,
                "Muestra la forma y densidad de cada grupo.",
            )
        else:
            agregar(
                "Barras apiladas",
                96,
                "Compara dos clasificaciones categóricas.",
            )
            agregar(
                "Mapa de calor",
                92,
                "Destaca concentraciones entre las categorías.",
            )

    elif objetivo == "Relación":
        if (
            tipo_1 == "numerica"
            and tipo_2 == "numerica"
        ):
            agregar(
                "Dispersión",
                98,
                "Representa la relación entre dos variables numéricas.",
            )
        else:
            agregar(
                "Boxplot",
                96,
                "Relaciona una variable numérica con grupos categóricos.",
            )
            agregar(
                "Violin plot",
                88,
                "Permite comparar densidades entre grupos.",
            )

    elif objetivo == "Evolución temporal":
        if tipo_1 == "categorica":
            agregar(
                "Barras apiladas",
                97,
                "Muestra la composición de las categorías por año.",
            )
            agregar(
                "Líneas múltiples",
                91,
                "Permite seguir la tendencia de cada categoría.",
            )
            agregar(
                "Área apilada",
                84,
                "Destaca el crecimiento y la composición total.",
            )
        else:
            agregar(
                "Línea temporal",
                98,
                "Muestra la evolución anual de la variable numérica.",
            )

    if usa_tiempo and objetivo != "Evolución temporal":
        ajustadas = []

        for item in recomendaciones:
            if item["grafica"] in {
                "Dona",
                "Histograma",
                "Boxplot",
                "Violin plot",
                "Dispersión",
            }:
                continue

            ajustadas.append(
                item
            )

        if ajustadas:
            recomendaciones = ajustadas
        else:
            if tipo_1 == "categorica":
                recomendaciones = [
                    {
                        "grafica": "Barras apiladas",
                        "porcentaje": 96,
                        "razon":
                            "Permite comparar categorías a lo largo del tiempo.",
                    },
                    {
                        "grafica": "Líneas múltiples",
                        "porcentaje": 90,
                        "razon":
                            "Muestra la tendencia anual de cada categoría.",
                    },
                ]
            else:
                recomendaciones = [
                    {
                        "grafica": "Línea temporal",
                        "porcentaje": 97,
                        "razon":
                            "Resume la evolución anual de la variable numérica.",
                    }
                ]

    return recomendaciones


def limitar_categorias(
    df: pd.DataFrame,
    columna: str,
    maximo: int = 12,
) -> pd.DataFrame:
    """Agrupa las categorías menos frecuentes como OTROS."""

    conteos = (
        df.groupby(
            columna
        )["ID_PERSONA_EXACTA"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    if len(conteos) <= maximo:
        return df

    principales = set(
        conteos.head(
            maximo
        ).index
    )

    salida = df.copy()

    salida[columna] = salida[columna].where(
        salida[columna].isin(
            principales
        ),
        "OTROS",
    )

    return salida


def construir_dataset_arbol(
    df: pd.DataFrame,
    catalogo: pd.DataFrame,
    variable_principal: str,
    variable_secundaria: str | None,
    usa_tiempo: bool,
    periodo: tuple[int, int] | None,
    anio: int | None,
) -> tuple[
    pd.DataFrame,
    str,
    str | None,
    str,
    str | None,
]:
    """Filtra el periodo y resuelve las columnas elegidas."""

    mapa = dict(
        zip(
            catalogo["VARIABLE"],
            catalogo["COLUMNA"],
        )
    )

    columna_1 = mapa[
        variable_principal
    ]

    columna_2 = (
        mapa[
            variable_secundaria
        ]
        if variable_secundaria
        else None
    )

    tipo_1 = VARIABLES_ANALITICAS[
        variable_principal
    ]["tipo"]

    tipo_2 = (
        VARIABLES_ANALITICAS[
            variable_secundaria
        ]["tipo"]
        if variable_secundaria
        else None
    )

    base = df.copy()

    if usa_tiempo:
        base = base.loc[
            base["AÑO"].between(
                periodo[0],
                periodo[1],
                inclusive="both",
            )
        ].copy()
    else:
        base = base.loc[
            base["AÑO"].eq(
                anio
            )
        ].copy()

    columnas = [
        "ID_PERSONA_EXACTA",
        "AÑO",
        columna_1,
    ]

    if columna_2:
        columnas.append(
            columna_2
        )

    base = base[
        list(
            dict.fromkeys(
                columnas
            )
        )
    ].dropna(
        subset=[
            columna
            for columna in [
                columna_1,
                columna_2,
            ]
            if columna is not None
        ],
        how="any",
    )

    if tipo_1 in {
        "categorica",
        "binaria",
        "ordinal",
    }:
        base = limitar_categorias(
            base,
            columna_1,
        )

    if (
        columna_2
        and tipo_2 in {
            "categorica",
            "binaria",
            "ordinal",
        }
    ):
        base = limitar_categorias(
            base,
            columna_2,
        )

    return (
        base,
        columna_1,
        columna_2,
        tipo_1,
        tipo_2,
    )


def generar_figura_arbol(
    base: pd.DataFrame,
    variable_principal: str,
    variable_secundaria: str | None,
    columna_1: str,
    columna_2: str | None,
    tipo_1: str,
    tipo_2: str | None,
    grafica: str,
    usa_tiempo: bool,
) -> tuple[
    go.Figure,
    pd.DataFrame,
]:
    """Genera la gráfica seleccionada por el usuario."""

    if grafica == "Dona":
        datos = (
            base.groupby(
                columna_1
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .rename("PERSONAS")
            .reset_index()
        )

        figura = px.pie(
            datos,
            names=columna_1,
            values="PERSONAS",
            hole=0.58,
            labels={
                columna_1:
                    variable_principal,
                "PERSONAS":
                    "Personas",
            },
        )

        figura.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        return figura, datos

    if grafica == "Barras":
        datos = (
            base.groupby(
                columna_1
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_values(
                ascending=False
            )
            .rename("PERSONAS")
            .reset_index()
        )

        figura = px.bar(
            datos,
            x=columna_1,
            y="PERSONAS",
            labels={
                columna_1:
                    variable_principal,
                "PERSONAS":
                    "Personas",
            },
        )

        return figura, datos

    if grafica in {
        "Barras apiladas",
        "Líneas múltiples",
        "Área apilada",
    }:
        color_columna = (
            columna_2
            if columna_2
            else columna_1
        )

        color_nombre = (
            variable_secundaria
            if variable_secundaria
            else variable_principal
        )

        if usa_tiempo:
            datos = (
                base.groupby(
                    [
                        "AÑO",
                        color_columna,
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            if grafica == "Barras apiladas":
                figura = px.bar(
                    datos,
                    x="AÑO",
                    y="PERSONAS",
                    color=color_columna,
                    barmode="stack",
                    labels={
                        color_columna:
                            color_nombre,
                        "PERSONAS":
                            "Personas",
                    },
                )

            elif grafica == "Líneas múltiples":
                figura = px.line(
                    datos,
                    x="AÑO",
                    y="PERSONAS",
                    color=color_columna,
                    markers=True,
                    labels={
                        color_columna:
                            color_nombre,
                        "PERSONAS":
                            "Personas",
                    },
                )

            else:
                figura = px.area(
                    datos,
                    x="AÑO",
                    y="PERSONAS",
                    color=color_columna,
                    labels={
                        color_columna:
                            color_nombre,
                        "PERSONAS":
                            "Personas",
                    },
                )

        else:
            datos = (
                base.groupby(
                    [
                        columna_1,
                        columna_2,
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            figura = px.bar(
                datos,
                x=columna_1,
                y="PERSONAS",
                color=columna_2,
                barmode="stack",
                labels={
                    columna_1:
                        variable_principal,
                    columna_2:
                        variable_secundaria,
                    "PERSONAS":
                        "Personas",
                },
            )

        return figura, datos

    if grafica == "Mapa de calor":
        if usa_tiempo:
            categoria = (
                columna_2
                if columna_2
                else columna_1
            )

            datos = (
                base.groupby(
                    [
                        categoria,
                        "AÑO",
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            matriz = datos.pivot(
                index=categoria,
                columns="AÑO",
                values="PERSONAS",
            ).fillna(0)

        else:
            datos = (
                base.groupby(
                    [
                        columna_1,
                        columna_2,
                    ]
                )["ID_PERSONA_EXACTA"]
                .nunique()
                .rename("PERSONAS")
                .reset_index()
            )

            matriz = datos.pivot(
                index=columna_1,
                columns=columna_2,
                values="PERSONAS",
            ).fillna(0)

        figura = px.imshow(
            matriz,
            aspect="auto",
            labels={
                "color":
                    "Personas",
            },
        )

        return figura, datos

    if grafica == "Histograma":
        figura = px.histogram(
            base,
            x=columna_1,
            nbins=30,
            labels={
                columna_1:
                    variable_principal,
            },
        )

        return (
            figura,
            base[
                [
                    columna_1,
                ]
            ].copy(),
        )

    if grafica in {
        "Boxplot",
        "Violin plot",
    }:
        if tipo_reducido(
            tipo_1
        ) == "numerica":
            columna_numerica = columna_1
            nombre_numerica = variable_principal
            columna_categoria = columna_2
            nombre_categoria = variable_secundaria

        else:
            columna_numerica = columna_2
            nombre_numerica = variable_secundaria
            columna_categoria = columna_1
            nombre_categoria = variable_principal

        if grafica == "Violin plot":
            figura = px.violin(
                base,
                x=columna_categoria,
                y=columna_numerica,
                box=True,
                points=False,
                labels={
                    columna_categoria:
                        nombre_categoria,
                    columna_numerica:
                        nombre_numerica,
                },
            )
        else:
            figura = px.box(
                base,
                x=columna_categoria,
                y=columna_numerica,
                points="outliers",
                labels={
                    columna_categoria:
                        nombre_categoria,
                    columna_numerica:
                        nombre_numerica,
                },
            )

        return figura, base.copy()

    if grafica == "Línea temporal":
        datos = (
            base.groupby(
                "AÑO"
            )[columna_1]
            .median()
            .rename("MEDIANA")
            .reset_index()
        )

        figura = px.line(
            datos,
            x="AÑO",
            y="MEDIANA",
            markers=True,
            labels={
                "MEDIANA":
                    f"Mediana de {variable_principal}",
            },
        )

        return figura, datos

    if grafica == "Dispersión":
        figura = px.scatter(
            base,
            x=columna_1,
            y=columna_2,
            labels={
                columna_1:
                    variable_principal,
                columna_2:
                    variable_secundaria,
            },
        )

        return figura, base.copy()

    raise ValueError(
        f"La gráfica '{grafica}' todavía no está implementada."
    )


def describir_grafica_arbol(
    objetivo: str,
    variable_principal: str,
    variable_secundaria: str | None,
    usa_tiempo: bool,
    periodo: tuple[int, int] | None,
    anio: int | None,
    escala: str,
    ubicacion: str,
) -> str:
    """Construye la descripción que aparece debajo del encabezado."""

    variables = (
        f"{variable_principal} y {variable_secundaria}"
        if variable_secundaria
        else variable_principal
    )

    if usa_tiempo:
        referencia_tiempo = (
            f"durante el periodo {periodo[0]}–{periodo[1]}"
        )
    else:
        referencia_tiempo = (
            f"en el año {anio}"
        )

    referencia_espacial = (
        "a nivel nacional"
        if escala == "Nacional"
        else f"para {ubicacion}"
    )

    return (
        f"Esta gráfica responde a un análisis de "
        f"{objetivo.lower()} de {variables}, "
        f"{referencia_tiempo}, {referencia_espacial}."
    )


def interpretar_tres_componentes(
    base: pd.DataFrame,
    columna_1: str,
    columna_2: str | None,
    tipo_1: str,
    tipo_2: str | None,
    usa_tiempo: bool,
) -> str:
    """
    Genera un párrafo con:
    1. tendencia general;
    2. categoría o grupo superior;
    3. comparación frente al inferior.
    """

    partes = []

    # --------------------------------------------------------
    # 1. TENDENCIA GENERAL
    # --------------------------------------------------------

    if usa_tiempo:
        serie = (
            base.groupby(
                "AÑO"
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_index()
        )

        if len(serie) >= 2:
            inicial = float(
                serie.iloc[0]
            )
            final = float(
                serie.iloc[-1]
            )

            cambio = (
                (final - inicial)
                / inicial
                * 100
                if inicial > 0
                else None
            )

            if cambio is None:
                tendencia = (
                    "no pudo determinarse con precisión"
                )
            elif cambio > 5:
                tendencia = "creciente"
            elif cambio < -5:
                tendencia = "decreciente"
            else:
                tendencia = "relativamente estable"

            partes.append(
                f"La tendencia general fue {tendencia}"
                + (
                    f", con un cambio de {cambio:+.1f}% "
                    "entre el primer y el último año"
                    if cambio is not None
                    else ""
                )
                + "."
            )

    else:
        partes.append(
            "El análisis corresponde a una fotografía descriptiva "
            "del año seleccionado."
        )

    # --------------------------------------------------------
    # 2 Y 3. SUPERIOR E INFERIOR
    # --------------------------------------------------------

    columna_categoria = None

    if tipo_1 in {
        "categorica",
        "binaria",
        "ordinal",
    }:
        columna_categoria = columna_1

    elif (
        columna_2 is not None
        and tipo_2 in {
            "categorica",
            "binaria",
            "ordinal",
        }
    ):
        columna_categoria = columna_2

    if columna_categoria is not None:
        resumen = (
            base.loc[
                ~base[
                    columna_categoria
                ]
                .astype("string")
                .str.upper()
                .isin(
                    [
                        "SIN INFORMACIÓN",
                        "SIN INFORMACION",
                        "NO DETERMINADO",
                        "OTROS",
                        "<NA>",
                    ]
                )
            ]
            .groupby(
                columna_categoria
            )["ID_PERSONA_EXACTA"]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        if len(resumen) >= 2:
            superior = resumen.index[0]
            inferior = resumen.index[-1]

            valor_superior = int(
                resumen.iloc[0]
            )
            valor_inferior = int(
                resumen.iloc[-1]
            )

            diferencia = (
                valor_superior
                - valor_inferior
            )

            porcentaje = (
                diferencia
                / valor_inferior
                * 100
                if valor_inferior > 0
                else None
            )

            partes.append(
                f"La categoría numéricamente superior fue "
                f"'{superior}', con {valor_superior:,} personas."
            )

            comparacion = (
                f"Superó a '{inferior}', que registró "
                f"{valor_inferior:,}, por {diferencia:,} personas"
            )

            if porcentaje is not None:
                comparacion += (
                    f", equivalente a {porcentaje:.1f}% "
                    "más respecto de la categoría inferior"
                )

            partes.append(
                comparacion + "."
            )

        elif len(resumen) == 1:
            categoria = resumen.index[0]
            valor = int(
                resumen.iloc[0]
            )

            partes.append(
                f"La única categoría con información válida fue "
                f"'{categoria}', con {valor:,} personas."
            )

    else:
        columna_numerica = (
            columna_1
            if tipo_1 == "numerica"
            else columna_2
        )

        if columna_numerica is not None:
            valores = base[
                columna_numerica
            ].dropna()

            if not valores.empty:
                partes.append(
                    f"El valor mediano fue {valores.median():.1f}; "
                    f"el 50% central se ubicó entre "
                    f"{valores.quantile(0.25):.1f} y "
                    f"{valores.quantile(0.75):.1f}."
                )

    return " ".join(
        partes
    )


def mostrar_nodo_completado(
    numero: int,
    titulo: str,
    valor: str,
) -> None:
    """Muestra un resumen compacto de un nodo ya resuelto."""

    st.markdown(
        (
            '<div class="snii-note">'
            f'<strong>✓ {numero}. {titulo}:</strong> {valor}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_laboratorio_visualizacion(
    df: pd.DataFrame,
) -> None:
    """Árbol de decisión progresivo para construir visualizaciones."""

    st.header(
        "4. Laboratorio de visualización"
    )

    st.markdown(
        '<p class="snii-subtitle">'
        "Construye tu análisis paso a paso. Cada decisión habilita "
        "el siguiente nodo del árbol."
        "</p>",
        unsafe_allow_html=True,
    )

    catalogo = catalogo_analitico_disponible(
        df
    )

    base = preparar_base_laboratorio(
        df,
        catalogo,
    )

    disponibles = catalogo.loc[
        catalogo["DISPONIBLE"]
    ].copy()

    if disponibles.empty:
        st.warning(
            "No se encontraron variables analíticas disponibles."
        )
        return

    st.info(
        "Los nombres, apellidos, CVU, Nobilis e identificadores "
        "se consultan únicamente en Historial del investigador."
    )

    # ========================================================
    # NODO 1. OBJETIVO
    # ========================================================

    st.subheader(
        "1. ¿Qué deseas analizar?"
    )

    objetivo = st.radio(
        "Objetivo del análisis",
        list(
            OBJETIVOS_LABORATORIO.keys()
        ),
        horizontal=True,
        key="arbol_objetivo",
    )

    st.caption(
        OBJETIVOS_LABORATORIO[
            objetivo
        ]["descripcion"]
    )

    mostrar_nodo_completado(
        1,
        "Objetivo",
        objetivo,
    )

    # ========================================================
    # NODO 2. VARIABLE PRINCIPAL
    # ========================================================

    st.subheader(
        "2. Selecciona la variable principal"
    )

    opciones_principales = opciones_variable_principal(
        catalogo,
        objetivo,
    )

    if not opciones_principales:
        st.warning(
            "No hay variables compatibles con el objetivo seleccionado."
        )
        return

    familias = sorted(
        {
            VARIABLES_ANALITICAS[
                variable
            ]["familia"]
            for variable in opciones_principales
        }
    )

    familia = st.selectbox(
        "Tema principal",
        familias,
        key="arbol_familia",
    )

    variables_familia = [
        variable
        for variable in opciones_principales
        if VARIABLES_ANALITICAS[
            variable
        ]["familia"] == familia
    ]

    variable_principal = st.selectbox(
        "Variable principal",
        variables_familia,
        key="arbol_variable_principal",
    )

    mostrar_nodo_completado(
        2,
        "Variable principal",
        variable_principal,
    )

    # ========================================================
    # NODO 3. VARIABLE SECUNDARIA
    # ========================================================

    requiere_secundaria = (
        OBJETIVOS_LABORATORIO[
            objetivo
        ]["requiere_secundaria"]
    )

    permite_secundaria = (
        OBJETIVOS_LABORATORIO[
            objetivo
        ]["permite_secundaria"]
    )

    variable_secundaria = None

    if requiere_secundaria or permite_secundaria:
        st.subheader(
            "3. Selecciona la variable de comparación"
        )

        opciones_secundarias = opciones_variable_secundaria(
            catalogo,
            objetivo,
            variable_principal,
        )

        if requiere_secundaria:
            if not opciones_secundarias:
                st.warning(
                    "No hay una segunda variable compatible."
                )
                return

            variable_secundaria = st.selectbox(
                "Variable secundaria",
                opciones_secundarias,
                key="arbol_variable_secundaria",
            )

        else:
            variable_secundaria = st.selectbox(
                "Variable secundaria opcional",
                [
                    "Ninguna",
                    *opciones_secundarias,
                ],
                key="arbol_variable_secundaria_opcional",
            )

            if variable_secundaria == "Ninguna":
                variable_secundaria = None

        mostrar_nodo_completado(
            3,
            "Variable de comparación",
            (
                variable_secundaria
                if variable_secundaria
                else "No aplica"
            ),
        )

        numero_tiempo = 4

    else:
        numero_tiempo = 3

    # ========================================================
    # NODO TIEMPO
    # ========================================================

    st.subheader(
        f"{numero_tiempo}. Define la dimensión temporal"
    )

    tiempo_obligatorio = (
        OBJETIVOS_LABORATORIO[
            objetivo
        ]["tiempo_obligatorio"]
    )

    if tiempo_obligatorio:
        usa_tiempo = True

        st.info(
            "El objetivo Evolución temporal requiere utilizar un periodo."
        )
    else:
        modo_tiempo = st.radio(
            "¿Cómo deseas considerar el tiempo?",
            [
                "Un año específico",
                "Un periodo",
                "Toda la serie histórica",
            ],
            horizontal=True,
            key="arbol_modo_tiempo",
        )

        usa_tiempo = (
            modo_tiempo
            != "Un año específico"
        )

    años = sorted(
        base["AÑO"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if usa_tiempo:
        if (
            not tiempo_obligatorio
            and modo_tiempo
            == "Toda la serie histórica"
        ):
            periodo = (
                min(
                    años
                ),
                max(
                    años
                ),
            )

            st.write(
                f"Periodo seleccionado: {periodo[0]}–{periodo[1]}"
            )

        else:
            periodo = st.slider(
                "Periodo",
                min_value=min(
                    años
                ),
                max_value=max(
                    años
                ),
                value=(
                    min(
                        años
                    ),
                    max(
                        años
                    ),
                ),
                key="arbol_periodo",
            )

        anio = None
        valor_tiempo = (
            f"{periodo[0]}–{periodo[1]}"
        )

    else:
        anio = st.selectbox(
            "Año de referencia",
            años,
            index=len(
                años
            ) - 1,
            key="arbol_anio",
        )

        periodo = None
        valor_tiempo = str(
            anio
        )

    mostrar_nodo_completado(
        numero_tiempo,
        "Tiempo",
        valor_tiempo,
    )

    # ========================================================
    # NODO UBICACIÓN
    # ========================================================

    numero_ubicacion = (
        numero_tiempo
        + 1
    )

    st.subheader(
        f"{numero_ubicacion}. Selecciona el alcance"
    )

    escala = st.radio(
        "Alcance geográfico o institucional",
        [
            "Nacional",
            "Por estado",
            "Por institución o centro",
        ],
        horizontal=True,
        key="arbol_escala",
    )

    ubicacion = "México"
    base_escala = base

    if escala == "Por estado":
        columna_estado = resolver_columna_analitica(
            base,
            "Entidad federativa",
        )

        if columna_estado is None:
            st.warning(
                "La entidad federativa todavía no tiene datos "
                "recuperados en el master."
            )
            return

        opciones_estado = sorted(
            base[
                columna_estado
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        ubicacion = st.selectbox(
            "Estado",
            opciones_estado,
            key="arbol_estado",
        )

        base_escala = base.loc[
            base[
                columna_estado
            ].eq(
                ubicacion
            )
        ].copy()

    elif escala == "Por institución o centro":
        columna_institucion = resolver_columna_analitica(
            base,
            "Institución",
        )

        if columna_institucion is None:
            st.warning(
                "La institución todavía no tiene datos "
                "recuperados en el master."
            )
            return

        opciones_institucion = sorted(
            base[
                columna_institucion
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        ubicacion = st.selectbox(
            "Institución o centro",
            opciones_institucion,
            key="arbol_institucion",
        )

        base_escala = base.loc[
            base[
                columna_institucion
            ].eq(
                ubicacion
            )
        ].copy()

    mostrar_nodo_completado(
        numero_ubicacion,
        "Alcance",
        ubicacion,
    )

    # ========================================================
    # PREGUNTA CONSTRUIDA
    # ========================================================

    variables_texto = (
        f"{variable_principal} respecto a {variable_secundaria}"
        if variable_secundaria
        else variable_principal
    )

    tiempo_texto = (
        f"durante {periodo[0]}–{periodo[1]}"
        if usa_tiempo
        else f"en {anio}"
    )

    st.markdown(
        (
            '<div class="lab-sentence">'
            '<strong>Pregunta construida:</strong><br>'
            f'¿Cómo se comporta {variables_texto}, '
            f'{tiempo_texto}, para {ubicacion}?'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    # ========================================================
    # RECOMENDACIÓN DE GRÁFICA
    # ========================================================

    numero_grafica = (
        numero_ubicacion
        + 1
    )

    st.subheader(
        f"{numero_grafica}. Selecciona el tipo de gráfica"
    )

    recomendaciones = recomendar_graficas_arbol(
        objetivo,
        variable_principal,
        variable_secundaria,
        usa_tiempo,
    )

    if not recomendaciones:
        st.warning(
            "No se encontró una visualización adecuada."
        )
        return

    etiquetas_grafica = [
        (
            f"{item['grafica']} "
            f"({item['porcentaje']}%)"
        )
        for item in recomendaciones
        if item["porcentaje"] >= 50
    ]

    grafica_seleccionada_etiqueta = st.radio(
        "Gráficas recomendadas",
        etiquetas_grafica,
        horizontal=True,
        key="arbol_tipo_grafica",
    )

    grafica_seleccionada = (
        grafica_seleccionada_etiqueta
        .rsplit(
            " (",
            1,
        )[0]
    )

    recomendacion_actual = next(
        item
        for item in recomendaciones
        if item["grafica"]
        == grafica_seleccionada
    )

    st.caption(
        recomendacion_actual[
            "razon"
        ]
    )

    # ========================================================
    # GENERACIÓN
    # ========================================================

    if not st.button(
        "Generar gráfica",
        type="primary",
        width="stretch",
        key="arbol_generar",
    ):
        return

    try:
        (
            base_analisis,
            columna_1,
            columna_2,
            tipo_1,
            tipo_2,
        ) = construir_dataset_arbol(
            base_escala,
            catalogo,
            variable_principal,
            variable_secundaria,
            usa_tiempo,
            periodo,
            anio,
        )

        if base_analisis.empty:
            st.warning(
                "No existen registros completos para esta selección."
            )
            return

        figura, datos = generar_figura_arbol(
            base_analisis,
            variable_principal,
            variable_secundaria,
            columna_1,
            columna_2,
            tipo_1,
            tipo_2,
            grafica_seleccionada,
            usa_tiempo,
        )

        titulo = (
            f"{objetivo}: {variables_texto}"
        )

        st.header(
            titulo
        )

        descripcion = describir_grafica_arbol(
            objetivo,
            variable_principal,
            variable_secundaria,
            usa_tiempo,
            periodo,
            anio,
            escala,
            ubicacion,
        )

        st.markdown(
            f"*{descripcion}*"
        )

        figura.update_layout(
            height=550,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20,
            ),
            hovermode=(
                "x unified"
                if usa_tiempo
                else "closest"
            ),
        )

        st.plotly_chart(
            figura,
            width="stretch",
            key="arbol_resultado",
        )

        interpretacion = interpretar_tres_componentes(
            base_analisis,
            columna_1,
            columna_2,
            tipo_1,
            tipo_2,
            usa_tiempo,
        )

        st.subheader(
            "Interpretación automática"
        )

        st.write(
            interpretacion
        )

        if (
            usa_tiempo
            and periodo is not None
            and periodo[1] >= 2025
        ):
            st.warning(
                "El año 2025 debe interpretarse con cautela "
                "hasta confirmar la cobertura completa de la fuente."
            )

        with st.expander(
            "Ver datos utilizados"
        ):
            st.dataframe(
                datos,
                width="stretch",
                hide_index=True,
            )

    except Exception as error:
        st.error(
            "No fue posible generar la gráfica: "
            f"{error}"
        )


# ============================================================
# APLICACIÓN
# ============================================================

def main() -> None:
    """Ejecuta únicamente el Laboratorio de visualización."""

    st.title("SNII Insight")
    st.markdown(
        '<p class="snii-subtitle">'
        "Laboratorio guiado para construir una pregunta mediante "
        "un árbol de decisión, seleccionar la gráfica recomendada "
        "e interpretar el resultado."
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        df, fuente = obtener_base()

    except Exception as error:
        st.error(f"No fue posible cargar la base: {error}")
        st.stop()

    st.sidebar.title("SNII Insight")
    st.sidebar.caption("Laboratorio de visualización")
    st.sidebar.caption(f"Fuente activa: {fuente}")
    st.sidebar.metric("Filas persona-año", f"{len(df):,}")
    st.sidebar.metric(
        "Personas únicas",
        f"{df['ID_PERSONA_EXACTA'].nunique():,}",
    )

    años = pd.to_numeric(
        df["AÑO"],
        errors="coerce",
    ).dropna()

    if not años.empty:
        st.sidebar.metric(
            "Cobertura temporal",
            f"{int(años.min())}–{int(años.max())}",
        )

    st.sidebar.info(
        "Esta versión contiene únicamente el árbol de decisión "
        "y la generación de visualizaciones."
    )

    render_laboratorio_visualizacion(df)


if __name__ == "__main__":
    main()
