"""
===============================================================================
PLANTILLA ESTÁNDAR — VISTA MINABLE (Clase 5 - Gestión de Datos)
Pontificia Universidad Javeriana
Docente: MSc. Edison Leonardo Neira Espitia
===============================================================================
Uso:
  1. Carga tu DataFrame en la variable `df`
  2. Configura las variables en la sección CONFIGURACIÓN
  3. Ejecuta celda por celda o todo el script
===============================================================================
"""

# =============================================================================
# 0. LIBRERÍAS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats

# Sklearn - Normalización
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, Normalizer
)
# Sklearn - Discretización
from sklearn.preprocessing import KBinsDiscretizer
# Sklearn - Numerización
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
# Imbalanced-learn - Balanceo
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
# Anonimización
import hashlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 40)

# =============================================================================
# 1. CONFIGURACIÓN  ← ← ← MODIFICA AQUÍ SEGÚN TU DATASET
# =============================================================================

# --- Carga de datos ----------------------------------------------------------
# df = pd.read_csv("tu_archivo.csv")
# df = pd.read_excel("tu_archivo.xlsx")
# Ejemplo con dataset de prueba:
from sklearn.datasets import fetch_california_housing
_data = fetch_california_housing(as_frame=True)
df = _data.frame.copy()
print(f"Shape: {df.shape}")
print(df.head())

# --- Variables por tipo ------------------------------------------------------
# Numéricas continuas para normalización
COLS_NUMERICAS = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]

# Variables para discretización (con posibles outliers)
COLS_DISCRETIZAR = ["MedInc", "AveOccup"]

# Variable categórica ordinal (si aplica) — None si no hay
COL_ORDINAL = None
ORDEN_ORDINAL = None  # ej: [["bajo", "medio", "alto"]]

# Variable categórica nominal para One-Hot (si aplica) — None si no hay
COL_NOMINAL = None

# Variables para crear variable derivada (ratio)
COL_DERIVADA_NUM = "AveRooms"      # numerador
COL_DERIVADA_DEN = "AveBedrms"     # denominador
NOMBRE_DERIVADA  = "rooms_per_bedroom"

# Variable objetivo para balanceo (binaria) — None si no aplica
COL_TARGET = None  # ej: "is_delivered"
COLS_FEATURES_BALANCEO = ["MedInc", "HouseAge"]

# Columnas identificadoras para anonimizar — None si no aplica
COLS_ANONIMIZAR = None  # ej: ["customer_id", "email"]

# Percentiles de winsorización
WINSOR_LOW, WINSOR_HIGH = 1, 99


# =============================================================================
# 2. FUNCIONES UTILITARIAS
# =============================================================================

def plot_antes_despues(df_orig, col_orig, df_new, col_new, titulo=""):
    """Grafica KDE antes y después de una transformación."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sns.kdeplot(df_orig[col_orig].dropna(), fill=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Original: {col_orig}")
    sns.kdeplot(df_new[col_new].dropna(), fill=True, ax=axes[1], color="coral")
    axes[1].set_title(f"Transformada: {col_new}")
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def freedman_diaconis_bins(series):
    """Calcula número óptimo de bins con Freedman-Diaconis."""
    x = series.dropna()
    n = len(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    h = 2 * iqr * n ** (-1/3)
    if h == 0:
        return int(np.sqrt(n))
    return max(2, int((x.max() - x.min()) / h))


def sturges_bins(series):
    """Calcula número de bins con Sturges."""
    return int(1 + np.log2(len(series.dropna())))


def scott_bins(series):
    """Calcula número de bins con Scott."""
    x = series.dropna()
    n = len(x)
    h = 3.5 * x.std() * n ** (-1/3)
    if h == 0:
        return int(np.sqrt(n))
    return max(2, int((x.max() - x.min()) / h))


def plot_balance(y, titulo=""):
    """Gráfico de barras de distribución de clases."""
    from collections import Counter
    counts = Counter(y)
    plt.figure(figsize=(5, 3))
    plt.bar(counts.keys(), counts.values(), color=["steelblue", "coral"])
    plt.title(titulo)
    plt.ylabel("Frecuencia")
    plt.show()
    print(f"  Distribución: {dict(counts)}")


# =============================================================================
# 3. NORMALIZACIÓN
# =============================================================================
print("\n" + "="*70)
print("3. NORMALIZACIÓN")
print("="*70)

df_norm = df.copy()

# --- 3.1 Min-Max Scaling ----------------------------------------------------
print("\n--- 3.1 Min-Max ---")
scaler_mm = MinMaxScaler()
cols_mm = [f"{c}_minmax" for c in COLS_NUMERICAS]
df_norm[cols_mm] = scaler_mm.fit_transform(df_norm[COLS_NUMERICAS])
for c in COLS_NUMERICAS:
    plot_antes_despues(df, c, df_norm, f"{c}_minmax", "Min-Max Scaling")
print(df_norm[cols_mm].describe().round(3))

# --- 3.2 Z-Score (StandardScaler) -------------------------------------------
print("\n--- 3.2 Z-Score ---")
scaler_z = StandardScaler()
cols_z = [f"{c}_zscore" for c in COLS_NUMERICAS]
df_norm[cols_z] = scaler_z.fit_transform(df_norm[COLS_NUMERICAS])
for c in COLS_NUMERICAS:
    plot_antes_despues(df, c, df_norm, f"{c}_zscore", "Z-Score")
print(df_norm[cols_z].describe().round(3))

# --- 3.3 Normalización Robusta -----------------------------------------------
print("\n--- 3.3 Robusta ---")
scaler_r = RobustScaler()
cols_r = [f"{c}_robust" for c in COLS_NUMERICAS]
df_norm[cols_r] = scaler_r.fit_transform(df_norm[COLS_NUMERICAS])
for c in COLS_NUMERICAS:
    plot_antes_despues(df, c, df_norm, f"{c}_robust", "Robust Scaling")

# --- 3.4 Logarítmica --------------------------------------------------------
print("\n--- 3.4 Logarítmica (log1p) ---")
cols_log = [f"{c}_log1p" for c in COLS_NUMERICAS]
for c in COLS_NUMERICAS:
    df_norm[f"{c}_log1p"] = np.log1p(df_norm[c].clip(lower=0))
    plot_antes_despues(df, c, df_norm, f"{c}_log1p", "Log1p")

# --- 3.5 L1 / L2 ------------------------------------------------------------
print("\n--- 3.5 L1 y L2 ---")
X_num = df_norm[COLS_NUMERICAS].fillna(0).values
l1 = Normalizer(norm='l1').fit_transform(X_num)
l2 = Normalizer(norm='l2').fit_transform(X_num)
df_norm[[f"{c}_l1" for c in COLS_NUMERICAS]] = l1
df_norm[[f"{c}_l2" for c in COLS_NUMERICAS]] = l2
print("Suma L1 por fila (promedio):", df_norm[[f"{c}_l1" for c in COLS_NUMERICAS]].abs().sum(axis=1).mean().round(4))
print("Norma L2 por fila (promedio):", np.sqrt((l2**2).sum(axis=1)).mean().round(4))

# --- 3.6 Yeo-Johnson --------------------------------------------------------
print("\n--- 3.6 Yeo-Johnson ---")
scaler_yj = PowerTransformer(method='yeo-johnson', standardize=True)
cols_yj = [f"{c}_yeojohnson" for c in COLS_NUMERICAS]
df_norm[cols_yj] = scaler_yj.fit_transform(df_norm[COLS_NUMERICAS])
for c in COLS_NUMERICAS:
    plot_antes_despues(df, c, df_norm, f"{c}_yeojohnson", "Yeo-Johnson")

# --- 3.7 Box-Cox (solo positivos) -------------------------------------------
print("\n--- 3.7 Box-Cox ---")
cols_pos = [c for c in COLS_NUMERICAS if (df[c] > 0).all()]
if cols_pos:
    scaler_bc = PowerTransformer(method='box-cox', standardize=True)
    cols_bc = [f"{c}_boxcox" for c in cols_pos]
    df_bc = df[df[cols_pos].gt(0).all(axis=1)].copy()
    df_bc[cols_bc] = scaler_bc.fit_transform(df_bc[cols_pos])
    for c in cols_pos:
        plot_antes_despues(df_bc, c, df_bc, f"{c}_boxcox", "Box-Cox")
else:
    print("  No hay columnas estrictamente positivas para Box-Cox.")

# --- 3.8 Quantile → Gaussian ------------------------------------------------
print("\n--- 3.8 QuantileTransformer (Gaussian) ---")
qt = QuantileTransformer(output_distribution="normal", random_state=42)
cols_qt = [f"{c}_gaussquant" for c in COLS_NUMERICAS]
df_norm[cols_qt] = qt.fit_transform(df_norm[COLS_NUMERICAS])
for c in COLS_NUMERICAS:
    plot_antes_despues(df, c, df_norm, f"{c}_gaussquant", "Quantile → Gaussian")


# =============================================================================
# 4. DISCRETIZACIÓN (BUCKETING / WINSORIZACIÓN)
# =============================================================================
print("\n" + "="*70)
print("4. DISCRETIZACIÓN")
print("="*70)

df_disc = df.copy()

for col in COLS_DISCRETIZAR:
    print(f"\n--- Discretizando: {col} ---")

    # 4.1 Winsorización
    low, high = np.percentile(df_disc[col].dropna(), [WINSOR_LOW, WINSOR_HIGH])
    print(f"  Winsorización: clip [{low:.2f}, {high:.2f}]")
    df_disc[f"{col}_winsor"] = df_disc[col].clip(lower=low, upper=high)

    # 4.2 Cálculo de bins
    col_w = f"{col}_winsor"
    n_fd = freedman_diaconis_bins(df_disc[col_w])
    n_st = sturges_bins(df_disc[col_w])
    n_sc = scott_bins(df_disc[col_w])
    print(f"  Bins → Freedman-Diaconis: {n_fd}, Sturges: {n_st}, Scott: {n_sc}")
    n_bins = n_fd  # elegimos FD por defecto

    # 4.3 Equal-width
    kb_w = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    df_disc[f"{col}_bin_width"] = kb_w.fit_transform(df_disc[[col_w]]).astype(int)

    # 4.4 Equal-frequency (quantile)
    kb_q = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    df_disc[f"{col}_bin_quant"] = kb_q.fit_transform(df_disc[[col_w]]).astype(int)

    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].hist(df_disc[col].dropna(), bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title(f"Original: {col}")
    axes[1].hist(df_disc[f"{col}_bin_width"], bins=n_bins, color="coral", edgecolor="white")
    axes[1].set_title(f"Equal-Width ({n_bins} bins)")
    axes[2].hist(df_disc[f"{col}_bin_quant"], bins=n_bins, color="seagreen", edgecolor="white")
    axes[2].set_title(f"Quantile ({n_bins} bins)")
    plt.suptitle(f"Discretización: {col}", fontweight="bold")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. NUMERIZACIÓN
# =============================================================================
print("\n" + "="*70)
print("5. NUMERIZACIÓN")
print("="*70)

df_num = df.copy()

# --- 5.1 Ordinal Encoding ---------------------------------------------------
if COL_ORDINAL and ORDEN_ORDINAL:
    print(f"\n--- 5.1 Ordinal: {COL_ORDINAL} ---")
    enc_ord = OrdinalEncoder(categories=ORDEN_ORDINAL)
    df_num[f"{COL_ORDINAL}_ord"] = enc_ord.fit_transform(df_num[[COL_ORDINAL]]).astype(int)
    print(df_num[[COL_ORDINAL, f"{COL_ORDINAL}_ord"]].drop_duplicates().sort_values(f"{COL_ORDINAL}_ord"))
else:
    print("\n--- 5.1 Ordinal: No configurada ---")

# --- 5.2 One-Hot Encoding ---------------------------------------------------
if COL_NOMINAL:
    print(f"\n--- 5.2 One-Hot: {COL_NOMINAL} ---")
    n_cats = df_num[COL_NOMINAL].nunique()
    print(f"  Categorías únicas: {n_cats}")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)
    ohe_array = ohe.fit_transform(df_num[[COL_NOMINAL]])
    ohe_cols = [f"{COL_NOMINAL}_{cat}" for cat in ohe.categories_[0]]
    df_ohe = pd.DataFrame(ohe_array, columns=ohe_cols, index=df_num.index)
    df_num = pd.concat([df_num, df_ohe], axis=1)
    print(f"  Columnas creadas: {len(ohe_cols)}")
    print(df_num[ohe_cols].head())
else:
    print("\n--- 5.2 One-Hot: No configurada ---")


# =============================================================================
# 6. VARIABLES DERIVADAS
# =============================================================================
print("\n" + "="*70)
print("6. VARIABLES DERIVADAS")
print("="*70)

df_der = df.copy()

if COL_DERIVADA_NUM and COL_DERIVADA_DEN:
    # Ratio entre dos variables
    df_der[NOMBRE_DERIVADA] = (
        df_der[COL_DERIVADA_NUM] / df_der[COL_DERIVADA_DEN].replace(0, np.nan)
    )
    print(f"\n  {NOMBRE_DERIVADA} = {COL_DERIVADA_NUM} / {COL_DERIVADA_DEN}")
    print(df_der[NOMBRE_DERIVADA].describe().round(3))

    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(df_der[NOMBRE_DERIVADA].dropna().clip(upper=df_der[NOMBRE_DERIVADA].quantile(0.99)),
                 bins=50, color="steelblue", ax=ax)
    ax.set_title(f"Variable Derivada: {NOMBRE_DERIVADA}")
    plt.tight_layout()
    plt.show()

# Variable relativa (respecto a la media)
for c in COLS_NUMERICAS[:2]:
    col_rel = f"{c}_relativo"
    df_der[col_rel] = df_der[c] - df_der[c].mean()
    print(f"  {col_rel} = {c} - media({c})")


# =============================================================================
# 7. BALANCEO DE DATOS (OVERSAMPLING / UNDERSAMPLING)
# =============================================================================
print("\n" + "="*70)
print("7. BALANCEO DE DATOS")
print("="*70)

if COL_TARGET:
    X = df[COLS_FEATURES_BALANCEO].fillna(0)
    y = df[COL_TARGET]

    print("\n--- Original ---")
    plot_balance(y, "Distribución Original")

    # Undersampling
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    X_under, y_under = rus.fit_resample(X, y)
    print("\n--- Undersampling ---")
    plot_balance(y_under, "Después de Undersampling")

    # Oversampling
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_over, y_over = ros.fit_resample(X, y)
    print("\n--- Oversampling ---")
    plot_balance(y_over, "Después de Oversampling")

    # SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    print("\n--- SMOTE ---")
    plot_balance(y_smote, "Después de SMOTE")
else:
    print("\n  No se configuró variable objetivo para balanceo.")


# =============================================================================
# 8. ANONIMIZACIÓN
# =============================================================================
print("\n" + "="*70)
print("8. ANONIMIZACIÓN")
print("="*70)

if COLS_ANONIMIZAR:
    df_anon = df[COLS_ANONIMIZAR].drop_duplicates().head(10).reset_index(drop=True).copy()
    print("\nDatos originales:")
    print(df_anon)

    for col in COLS_ANONIMIZAR:
        # Hashing (SHA-256, irreversible)
        df_anon[f"{col}_hash"] = df_anon[col].astype(str).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )
        # Masking (enmascaramiento parcial)
        df_anon[f"{col}_masked"] = df_anon[col].astype(str).apply(
            lambda x: x[:2] + "*" * max(0, len(x)-5) + x[-3:] if len(x) > 5 else "***"
        )
        # Tokenización (ID secuencial)
        token_map = {v: f"TOK_{i:04d}" for i, v in enumerate(df_anon[col].unique())}
        df_anon[f"{col}_token"] = df_anon[col].map(token_map)

    print("\nDatos anonimizados:")
    print(df_anon)
else:
    print("\n  No se configuraron columnas para anonimizar.")


# =============================================================================
# 9. RESUMEN FINAL
# =============================================================================
print("\n" + "="*70)
print("9. RESUMEN")
print("="*70)
print(f"""
  Dataset original:     {df.shape}
  Columnas numéricas:   {COLS_NUMERICAS}
  Discretizadas:        {COLS_DISCRETIZAR}
  Ordinal:              {COL_ORDINAL or 'N/A'}
  One-Hot:              {COL_NOMINAL or 'N/A'}
  Variable derivada:    {NOMBRE_DERIVADA or 'N/A'}
  Target balanceo:      {COL_TARGET or 'N/A'}
  Anonimización:        {COLS_ANONIMIZAR or 'N/A'}

  ✅ Pipeline completo. Modifica la sección CONFIGURACIÓN para tu dataset.
""")
