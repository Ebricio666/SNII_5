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
    page_title="SNII Insight",
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTES
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
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
    """Prepara tipos esenciales y conserva sólo columnas disponibles."""
    faltantes = [col for col in COLUMNAS_MINIMAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            "La base no contiene las columnas indispensables: "
            + ", ".join(faltantes)
        )

    df = df.copy()

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


@st.cache_data(show_spinner="Cargando la base histórica del SNII…")
def cargar_desde_ruta(ruta: str) -> pd.DataFrame:
    """Carga Parquet o Excel desde el repositorio."""
    path = Path(ruta)

    if path.suffix.lower() == ".parquet":
        columnas_disponibles = pd.read_parquet(path, engine="pyarrow").columns
        columnas = [
            col for col in COLUMNAS_ANALITICAS if col in columnas_disponibles
        ]
        df = pd.read_parquet(path, columns=columnas, engine="pyarrow")

    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(
            path,
            sheet_name="PERSONA_AÑO",
            engine="openpyxl",
        )
        columnas = [col for col in COLUMNAS_ANALITICAS if col in df.columns]
        df = df[columnas].copy()

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
        df = pd.read_excel(
            buffer,
            sheet_name="PERSONA_AÑO",
            engine="openpyxl",
        )

    else:
        raise ValueError("Carga un archivo .parquet o .xlsx.")

    columnas = [col for col in COLUMNAS_ANALITICAS if col in df.columns]
    return preparar_base(df[columnas].copy())


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
            "Coloca `SNII_MASTER_v1_PERSONA_ANIO.parquet` dentro de "
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

    st.subheader("Evolución histórica")

    fig_serie = px.bar(
        serie,
        x="AÑO",
        y="PERSONAS",
        labels={"PERSONAS": "Personas únicas", "AÑO": "Año"},
        text_auto=False,
    )
    fig_serie.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_serie, use_container_width=True)

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
            st.plotly_chart(fig_sexo, use_container_width=True)

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
            st.plotly_chart(fig_nivel, use_container_width=True)

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
            st.plotly_chart(fig_stem, use_container_width=True)

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
            st.plotly_chart(fig_inst, use_container_width=True)

    with st.expander("Ver datos utilizados"):
        st.dataframe(
            actual,
            use_container_width=True,
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

    st.plotly_chart(fig, use_container_width=True)

    izquierda, derecha = st.columns([1, 1.35])

    with izquierda:
        st.subheader("Comparación de modelos")
        tabla_evaluacion = evaluacion.copy()
        tabla_evaluacion[["MAE", "RMSE"]] = tabla_evaluacion[
            ["MAE", "RMSE"]
        ].round(2)

        st.dataframe(
            tabla_evaluacion,
            use_container_width=True,
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
            use_container_width=True,
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
            st.plotly_chart(fig_nivel, use_container_width=True)

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
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# APLICACIÓN
# ============================================================

def main() -> None:
    st.title("SNII Insight")
    st.markdown(
        '<p class="snii-subtitle">'
        "Plataforma para explorar la evolución histórica del "
        "Sistema Nacional de Investigadoras e Investigadores."
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        df, fuente = obtener_base()

    except Exception as error:
        st.error(f"No fue posible cargar la base: {error}")
        st.stop()

    st.sidebar.title("SNII Insight")
    st.sidebar.caption(f"Fuente activa: {fuente}")
    st.sidebar.metric("Filas persona-año", f"{len(df):,}")
    st.sidebar.metric(
        "Personas únicas",
        f"{df['ID_PERSONA_EXACTA'].nunique():,}",
    )

    modulo = st.sidebar.radio(
        "Módulo",
        [
            "Panorama actual",
            "Proyecciones",
            "Historial del investigador",
        ],
    )

    if modulo == "Panorama actual":
        render_panorama(df)

    elif modulo == "Proyecciones":
        render_proyecciones(df)

    else:
        render_investigador(df)


if __name__ == "__main__":
    main()
