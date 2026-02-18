# -*- coding: utf-8 -*-
"""
resumen_eda_completo.py
Función EDA completa basada en Clases 1, 2 y 3
Gestión de Datos - Pontificia Universidad Javeriana
Docente: MSc. Edison Leonardo Neira Espitia

Uso en Google Colab
-------------------
    import subprocess, sys

    def clonar_repo(repo_origen, repo_destino):
        subprocess.run(["git", "clone", repo_origen, repo_destino])

    clonar_repo("https://github.com/TU_USUARIO/eda-javeriana.git", "eda-javeriana")
    sys.path.insert(0, "eda-javeriana")

    from resumen_eda_completo import resumen_completov2, faltantes, resumen_numerico
"""

import numpy as np
import pandas as pd
import statistics as statspy
from scipy.stats import skew, kurtosis
from typing import Optional, Union


def resumen_completov2(
    df: pd.DataFrame,
    nombre: str = "DataFrame",
    max_cols_sin_preguntar: int = 20
) -> pd.DataFrame:
    """
    Genera un resumen EDA completo para todas las columnas.

    Incluye:
    - Identificación automática de tipo de variable (nominal, ordinal, numérica)
    - Análisis de nulos con diagnóstico visual
    - Medidas de tendencia central (media, mediana, moda)
    - Medidas de dispersión (varianza, desv. estándar, IQR)
    - Análisis de asimetría (sesgo) y curtosis
    - Detección de outliers con umbrales

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar
    nombre : str
        Nombre descriptivo del DataFrame
    max_cols_sin_preguntar : int
        Si tiene más columnas, pregunta antes de continuar

    Retorna
    -------
    pd.DataFrame
        Tabla resumen con métricas por columna
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se esperaba un pd.DataFrame")

    n_filas, n_cols = df.shape
    print("=" * 70)
    print(f"  RESUMEN EDA — {nombre}")
    print(f"  Filas: {n_filas:,}  |  Columnas: {n_cols}")
    print("=" * 70)

    # Pregunta si hay muchas columnas
    if n_cols > max_cols_sin_preguntar:
        resp = input(
            f"\n⚠️  El DataFrame tiene {n_cols} columnas (>{max_cols_sin_preguntar}). "
            "¿Desea continuar? (s/n): "
        ).strip().lower()
        if resp != "s":
            print("❌ Análisis cancelado.")
            return pd.DataFrame()

    # Analizar cada columna
    registros = []
    for col in df.columns:
        registros.append(_analizar_columna_completa(df[col]))

    resultado = pd.DataFrame(registros).set_index("columna")

    # ══════════════════════════════════════════════════════════
    # RESUMEN GLOBAL DE NULOS
    # ══════════════════════════════════════════════════════════
    faltantes_total = df.isna().sum().sum()
    celdas_total = n_filas * n_cols
    pct_global = round(100 * faltantes_total / celdas_total, 2)

    print(f"\n{'─'*70}")
    print(f"  📊 RESUMEN GLOBAL DE DATOS FALTANTES")
    print(f"{'─'*70}")
    print(f"   Celdas totales : {celdas_total:,}")
    print(f"   Celdas nulas   : {faltantes_total:,}")
    print(f"   % nulo global  : {pct_global}%")

    # ══════════════════════════════════════════════════════════
    # DIAGNÓSTICO POR COLUMNA
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  🔍 DIAGNÓSTICO POR COLUMNA")
    print(f"{'─'*70}\n")

    for col in df.columns:
        pct = round(100 * df[col].isna().mean(), 2)
        tipo_var = resultado.loc[col, "tipo_variable"]

        if pct == 0:
            estado = "✅ OK"
        elif pct < 5:
            estado = f"🟡 BAJO ({pct}%)"
        elif pct < 30:
            estado = f"🟠 MEDIO ({pct}%)"
        else:
            estado = f"🔴 ALTO ({pct}%)"

        print(f"   {col:<35s} [{tipo_var:^15s}] {estado}")

    print("\n" + "=" * 70)
    print("  ✅ Análisis completo. Tabla resumen retornada como DataFrame.")
    print("=" * 70 + "\n")

    return resultado


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN INTERNA: Análisis completo de una columna
# ══════════════════════════════════════════════════════════════════
def _analizar_columna_completa(serie: pd.Series) -> dict:
    """Análisis completo de una sola columna."""

    col_name = serie.name
    total = len(serie)
    nulos = int(serie.isna().sum())
    pct_nulos = round(100 * nulos / total, 2) if total > 0 else 0.0
    no_nulos = total - nulos
    valores_unicos = serie.nunique(dropna=True)

    base = {
        "columna": col_name,
        "dtype_almacenamiento": str(serie.dtype),
        "total": total,
        "no_nulos": no_nulos,
        "nulos": nulos,
        "%_nulos": pct_nulos,
        "valores_unicos": valores_unicos,
    }

    # ══════════════════════════════════════════════════════════
    # VARIABLES NUMÉRICAS
    # ══════════════════════════════════════════════════════════
    if pd.api.types.is_numeric_dtype(serie):
        s = serie.dropna()

        if len(s) == 0:
            base["tipo_variable"] = "Numérica (vacía)"
            return base

        tipo_num = _clasificar_numerica(s)

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = s[(s < low) | (s > high)]

        sk = float(skew(s, bias=False, nan_policy="omit"))
        if sk > 0.5:
            tipo_sesgo = "Positivo (cola derecha)"
        elif sk < -0.5:
            tipo_sesgo = "Negativo (cola izquierda)"
        else:
            tipo_sesgo = "Aprox. simétrica"

        kt = float(kurtosis(s, fisher=True, bias=False, nan_policy="omit"))
        if kt > 0:
            tipo_curtosis = "Leptocúrtica (colas pesadas)"
        elif kt < 0:
            tipo_curtosis = "Platicúrtica (colas ligeras)"
        else:
            tipo_curtosis = "Mesocúrtica (≈ normal)"

        try:
            modas = statspy.multimode(s.tolist())
            if len(modas) == 1:
                moda_str = str(modas[0])
            elif len(modas) <= 3:
                moda_str = f"Bimodal: {modas}"
            elif len(modas) <= 5:
                moda_str = f"Trimodal: {modas}"
            else:
                moda_str = f"Multimodal ({len(modas)} modas)"
        except:
            moda_str = "N/A"

        base.update({
            "tipo_variable": f"Numérica-{tipo_num}",
            "min": round(s.min(), 4),
            "q1": round(q1, 4),
            "mediana": round(s.median(), 4),
            "media": round(s.mean(), 4),
            "q3": round(q3, 4),
            "max": round(s.max(), 4),
            "moda": moda_str,
            "rango": round(s.max() - s.min(), 4),
            "IQR": round(iqr, 4),
            "varianza": round(s.var(ddof=1), 4),
            "desv_std": round(s.std(ddof=1), 4),
            "umbral_bajo": round(low, 4),
            "umbral_alto": round(high, 4),
            "%_outliers": round(100 * len(outliers) / len(s), 2),
            "sesgo_skew": round(sk, 4),
            "interpretacion_sesgo": tipo_sesgo,
            "curtosis_fisher": round(kt, 4),
            "interpretacion_curtosis": tipo_curtosis,
        })

    # ══════════════════════════════════════════════════════════
    # VARIABLES CATEGÓRICAS / TEMPORALES
    # ══════════════════════════════════════════════════════════
    else:
        s = serie.dropna()
        es_temporal = pd.api.types.is_datetime64_any_dtype(serie)

        tipo_cat = _clasificar_categorica(s) if not es_temporal else "Temporal"

        try:
            moda_val = statspy.mode(s.tolist()) if len(s) > 0 else "N/A"
        except:
            moda_val = "N/A"

        top = s.value_counts().head(1)
        top_val = top.index[0] if len(top) > 0 else "N/A"
        top_count = int(top.values[0]) if len(top) > 0 else 0
        top_pct = round(100 * top_count / len(s), 2) if len(s) > 0 else 0

        base.update({
            "tipo_variable": tipo_cat,
            "moda": str(moda_val),
            "valor_mas_frecuente": str(top_val),
            "frecuencia_top": top_count,
            "%_frecuencia_top": top_pct,
            "min": str(s.min()) if es_temporal else "N/A",
            "max": str(s.max()) if es_temporal else "N/A",
            "rango": str(s.max() - s.min()) if es_temporal else "N/A",
        })

    return base


# ══════════════════════════════════════════════════════════════════
#  CLASIFICADORES DE TIPO DE VARIABLE
# ══════════════════════════════════════════════════════════════════
def _clasificar_numerica(serie: pd.Series) -> str:
    """
    Clasifica una variable numérica en Continua o Discreta.

    Heurística:
    - Si todos los valores son enteros → Discreta
    - Si hay decimales → Continua
    """
    if (serie % 1 == 0).all():
        return "Discreta"
    else:
        return "Continua"


def _clasificar_categorica(serie: pd.Series) -> str:
    """
    Clasifica una variable categórica en Nominal u Ordinal.

    Heurística básica:
    - Si los valores únicos son pocos y parecen tener orden → Ordinal
    - De lo contrario → Nominal

    (En casos reales, necesitarías conocimiento del dominio)
    """
    keywords_ordinal = [
        "malo", "regular", "bueno", "excelente",
        "bajo", "medio", "alto",
        "primero", "segundo", "tercero",
        "small", "medium", "large",
        "low", "high"
    ]

    valores_str = [str(v).lower() for v in serie.unique() if pd.notna(v)]

    for val in valores_str:
        for kw in keywords_ordinal:
            if kw in val:
                return "Categórica-Ordinal"

    return "Categórica-Nominal"


# ══════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES (compatibles con la Clase 3)
# ══════════════════════════════════════════════════════════════════
def faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de faltantes ordenada (de la Clase 3)."""
    f = df.isna().sum()
    p = (df.isna().mean() * 100).round(2)
    return pd.DataFrame({
        "faltantes": f,
        "%": p
    }).sort_values("%", ascending=False)


def resumen_numerico(serie: pd.Series) -> dict:
    """Resumen numérico de una serie (de la Clase 3)."""
    s = serie.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    return {
        "count": len(s),
        "mean": round(s.mean(), 4),
        "median": round(s.median(), 4),
        "std": round(s.std(ddof=1), 4),
        "min": round(s.min(), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "max": round(s.max(), 4),
        "IQR": round(iqr, 4),
        "low_thr": round(low, 4),
        "high_thr": round(high, 4),
        "outliers_%": round(100 * len(s[(s < low) | (s > high)]) / len(s), 2),
        "skew": round(float(skew(s, bias=False)), 4),
        "kurtosis_fisher": round(float(kurtosis(s, fisher=True, bias=False)), 4),
    }


# ══════════════════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Ejecutando ejemplo con datos de prueba...\n")

    np.random.seed(42)
    df_test = pd.DataFrame({
        "precio": np.random.exponential(100, 500),
        "cantidad": np.random.randint(1, 20, 500),
        "satisfaccion": np.random.choice(["Malo", "Regular", "Bueno", "Excelente"], 500),
        "categoria": np.random.choice(["A", "B", "C", "D"], 500),
        "fecha": pd.date_range("2024-01-01", periods=500, freq="h"),
    })

    # Insertar nulos
    df_test.loc[10:20, "precio"] = np.nan
    df_test.loc[50:55, "categoria"] = np.nan

    resultado = resumen_completov2(df_test, nombre="Datos de prueba")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("\n" + "="*70)
    print("TABLA RESUMEN TRANSPUESTA (para mejor visualización)")
    print("="*70)
    print(resultado.T.to_string())
