#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Núcleo común del material de Investigación Operativa (CONIPE 2025)
==================================================================

Este módulo reúne, en un solo lugar, las piezas que comparten el análisis
comparativo (`analisis_comparativo_io.py`), la aplicación Streamlit
(`app_streamlit_io.py`) y los tests:

1. **Un "mundo" sintético** de flota pesquera. Los datos NO son reales: se
   generan con fórmulas conocidas (ver `rendimiento_verdadero`) más ruido
   aleatorio. Conocer la fórmula permite comprobar qué recupera cada método.
2. **El modelo de programación lineal (PL)** de asignación de barcos y su
   resolución con HiGHS, incluidos los precios sombra.
3. **Métricas de predicción** sin recortes: un R² negativo se informa tal cual.

Dos preguntas distintas
-----------------------
* **Decidir** (PL / Simplex): con unos rendimientos y unos recursos dados,
  ¿cuántos barcos de cada tipo conviene enviar?
* **Predecir** (ML): a partir de las condiciones, ¿cuánto se va a capturar?

Se combinan en el enfoque "predecir y luego optimizar": el ML estima los
rendimientos (coeficientes inciertos) y la PL decide con esas estimaciones.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# 1. Modelo de decisión (programación lineal)
# ---------------------------------------------------------------------------

TIPOS_BARCO = ("pequeño", "mediano", "grande")

#: Rendimiento nominal (kg capturados por barco y salida) en condiciones de
#: calma: temperatura 20 °C, sin viento, mes 6 y fase lunar neutra.
RENDIMIENTO_NOMINAL = np.array([50.0, 120.0, 200.0])

RECURSOS = ("combustible", "tripulación", "presupuesto")
UNIDADES_RECURSOS = ("L", "personas", "$")
UNIDAD_SINGULAR = ("litro", "persona", "dólar")

#: Consumo de cada recurso por barco (filas: recursos; columnas: tipos).
A_RECURSOS = np.array([
    [20.0, 50.0, 100.0],        # combustible (L por barco)
    [2.0, 5.0, 8.0],            # tripulación (personas por barco)
    [1000.0, 2500.0, 5000.0],   # presupuesto ($ por barco)
])

#: Barcos disponibles de cada tipo (cotas de las variables).
COTAS_BARCOS = [(0, 20), (0, 15), (0, 10)]

#: Escenario base del curso (valores por defecto de la app Streamlit).
RECURSOS_BASE = np.array([3000.0, 150.0, 125000.0])


def resolver_lp(coeficientes=RENDIMIENTO_NOMINAL, recursos=RECURSOS_BASE,
                A=A_RECURSOS, cotas=COTAS_BARCOS):
    """Resuelve  max c·x  s.a.  A·x <= recursos,  cotas inferiores/superiores.

    Devuelve un diccionario con la solución óptima, el valor objetivo, el uso y
    la holgura de cada recurso y los **precios sombra**: cuánto aumentaría el
    objetivo (kg) con una unidad más de cada recurso. Se obtienen de
    ``res.ineqlin.marginals`` (HiGHS). Como `linprog` minimiza, se resuelve
    ``min -c·x`` y se cambia el signo de los marginales.
    """
    c = np.asarray(coeficientes, dtype=float)
    b = np.asarray(recursos, dtype=float)
    res = linprog(-c, A_ub=A, b_ub=b, bounds=cotas, method="highs")
    if not res.success:
        return {"exito": False, "mensaje": res.message, "estado": res.status}

    x = res.x
    uso = A @ x
    return {
        "exito": True,
        "x": x,
        "objetivo": float(-res.fun),
        "uso": uso,
        "holgura": b - uso,
        # +0.0 evita mostrar "-0" cuando el marginal es cero
        "precios_sombra": -res.ineqlin.marginals + 0.0,
        # valor de un barco más disponible de cada tipo (cota superior)
        "valor_barco_extra": -res.upper.marginals + 0.0,
    }


def sensibilidad_recurso(indice, valores, coeficientes=RENDIMIENTO_NOMINAL,
                         recursos=RECURSOS_BASE):
    """Re-resuelve la PL variando un recurso y devuelve el objetivo en cada valor."""
    filas = []
    for v in valores:
        b = np.array(recursos, dtype=float)
        b[indice] = v
        sol = resolver_lp(coeficientes, b)
        filas.append({
            "valor": float(v),
            "objetivo": sol["objetivo"] if sol["exito"] else np.nan,
            "precio_sombra": sol["precios_sombra"][indice] if sol["exito"] else np.nan,
        })
    return pd.DataFrame(filas)


def rango_precio_sombra(indice, coeficientes=RENDIMIENTO_NOMINAL,
                        recursos=RECURSOS_BASE, paso=None, factor_max=3.0):
    """Intervalo (aprox.) del recurso en el que su precio sombra no cambia.

    Se explora numéricamente hacia abajo y hacia arriba desde el valor base.
    Es la versión "experimental" del análisis de rango del lado derecho.
    """
    base = float(recursos[indice])
    paso = paso or base / 1000.0
    ps_base = resolver_lp(coeficientes, recursos)["precios_sombra"][indice]

    def mismo(v):
        b = np.array(recursos, dtype=float)
        b[indice] = v
        sol = resolver_lp(coeficientes, b)
        return sol["exito"] and np.isclose(sol["precios_sombra"][indice], ps_base, atol=1e-9)

    inf = base
    while inf - paso >= 0 and mismo(inf - paso):
        inf -= paso
    sup = base
    while sup + paso <= base * factor_max and mismo(sup + paso):
        sup += paso
    # Si el precio no cambió en todo el tramo explorado, no hay límite superior
    # (un recurso que sobra sigue sobrando por mucho que se aumente).
    if sup + paso > base * factor_max:
        sup = np.inf
    return inf, sup, ps_base


# ---------------------------------------------------------------------------
# 2. Mundo sintético: cómo se generan los datos
# ---------------------------------------------------------------------------

#: El viento perjudica más a los barcos pequeños que a los grandes.
#: factor_viento_i = max(MINIMO_i, 1 - viento / ESCALA_i)
VIENTO_MINIMO = np.array([0.1, 0.3, 0.6])
VIENTO_ESCALA = np.array([25.0, 40.0, 80.0])

VARIABLES_CLIMA = ["temperatura_agua", "salinidad", "velocidad_viento",
                   "altura_olas", "mes", "fase_lunar"]


def rendimiento_verdadero(temperatura, viento, mes, fase_lunar):
    """Rendimiento real (kg por barco) de cada tipo según el clima.

    rend_i = nominal_i · f_temp · f_lunar · f_estación · f_viento_i

    * f_temp     = 1 + 0,1·sen((T − 20)·π/10)
    * f_lunar    = 1 + 0,2·sen(2π·fase)
    * f_estación = 1 + 0,3·sen((mes − 6)·π/6)
    * f_viento_i = max(mín_i, 1 − viento/escala_i)  (distinto por tipo)

    La salinidad y la altura de olas NO intervienen: son variables
    irrelevantes a propósito, para ver si los modelos las ignoran.
    Devuelve una matriz (n, 3).
    """
    T = np.atleast_1d(np.asarray(temperatura, dtype=float))
    v = np.atleast_1d(np.asarray(viento, dtype=float))
    m = np.atleast_1d(np.asarray(mes, dtype=float))
    f = np.atleast_1d(np.asarray(fase_lunar, dtype=float))

    comun = ((1 + 0.1 * np.sin((T - 20) * np.pi / 10))
             * (1 + 0.2 * np.sin(f * 2 * np.pi))
             * (1 + 0.3 * np.sin((m - 6) * np.pi / 6)))
    f_viento = np.maximum(VIENTO_MINIMO, 1 - v[:, None] / VIENTO_ESCALA)
    return RENDIMIENTO_NOMINAL * comun[:, None] * f_viento


def generar_condiciones(n, rng):
    """Condiciones aleatorias: clima, recursos y mercado (distribuciones uniformes)."""
    return pd.DataFrame({
        "combustible_disponible": rng.uniform(1000, 5000, n),
        "tripulacion_disponible": rng.integers(50, 300, n),
        "presupuesto": rng.uniform(50000, 200000, n),
        "temperatura_agua": rng.uniform(15, 25, n),
        "salinidad": rng.uniform(30, 40, n),
        "velocidad_viento": rng.uniform(0, 30, n),
        "altura_olas": rng.uniform(0.5, 4.0, n),
        "precio_pescado": rng.uniform(2, 8, n),
        "demanda_mercado": rng.uniform(0.5, 1.5, n),
        "competencia_zona": rng.uniform(0.1, 1.0, n),
        "mes": rng.integers(1, 13, n),
        "fase_lunar": rng.uniform(0, 1, n),
    })


def _rend(df):
    return rendimiento_verdadero(df["temperatura_agua"], df["velocidad_viento"],
                                 df["mes"], df["fase_lunar"])


def generar_datos_flota(n=1000, semilla=42, ruido=0.10):
    """Escenarios para la parte de PREDICCIÓN.

    Cada fila es una salida de una flota con un número aleatorio de barcos de
    cada tipo. La captura es

        capturas_totales = Σ_i barcos_i · rend_i(clima) · (1 + ε),  ε ~ N(0, ruido)

    Es lineal en el número de barcos pero multiplicativa en el clima.
    Se guarda también la captura sin ruido (`capturas_sin_ruido`) para medir el
    techo de R² que puede alcanzar cualquier modelo.
    """
    rng = np.random.default_rng(semilla)
    barcos = pd.DataFrame({
        "barcos_pequenos": rng.integers(0, 21, n),
        "barcos_medianos": rng.integers(0, 16, n),
        "barcos_grandes": rng.integers(0, 11, n),
    })
    df = pd.concat([barcos, generar_condiciones(n, rng)], axis=1)
    sin_ruido = (barcos.to_numpy() * _rend(df)).sum(axis=1)
    df["capturas_sin_ruido"] = sin_ruido
    df["capturas_totales"] = np.maximum(0, sin_ruido * (1 + rng.normal(0, ruido, n)))
    return df


VARIABLES_PREDICCION = [
    "barcos_pequenos", "barcos_medianos", "barcos_grandes",
    "combustible_disponible", "tripulacion_disponible", "presupuesto",
    "temperatura_agua", "salinidad", "velocidad_viento", "altura_olas",
    "precio_pescado", "demanda_mercado", "competencia_zona",
    "mes", "fase_lunar",
]


def generar_historico_salidas(n_por_tipo=1000, semilla=7, ruido=0.15):
    """Histórico para el enfoque HÍBRIDO: captura observada por barco y salida.

    Una fila por salida: tipo de barco, condiciones climáticas y kg capturados
    por ese barco = rend_tipo(clima) · (1 + ε).
    """
    rng = np.random.default_rng(semilla)
    partes = []
    for t, tipo in enumerate(TIPOS_BARCO):
        cond = generar_condiciones(n_por_tipo, rng)[VARIABLES_CLIMA]
        rend = _rend(cond)[:, t]
        cond = cond.assign(tipo=tipo,
                           captura_por_barco=rend * (1 + rng.normal(0, ruido, n_por_tipo)),
                           rendimiento_real=rend)
        partes.append(cond)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Métricas de predicción (sin recortes)
# ---------------------------------------------------------------------------

def metricas_regresion(y_real, y_pred):
    """R², MSE y MAE. El R² se devuelve tal cual, aunque sea negativo:
    un R² < 0 significa que el modelo predice peor que la media."""
    return {
        "r2": float(r2_score(y_real, y_pred)),
        "mse": float(mean_squared_error(y_real, y_pred)),
        "mae": float(mean_absolute_error(y_real, y_pred)),
    }


# ---------------------------------------------------------------------------
# Utilidades de formato (decimales con coma)
# ---------------------------------------------------------------------------

def fmt(valor, decimales=2, miles=True):
    """Formatea un número al estilo español: 1.234,56"""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "—"
    s = f"{valor:,.{decimales}f}" if miles else f"{valor:.{decimales}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")
