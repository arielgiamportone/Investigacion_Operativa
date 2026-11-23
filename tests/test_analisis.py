# -*- coding: utf-8 -*-
"""Tests mínimos del material: la PL base y las métricas de predicción."""

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

import nucleo_io as nio
from analisis_comparativo_io import AnalizadorComparativoIO, NOMBRE_MEDIA

RAIZ = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Cálculos de referencia, independientes de nucleo_io.resolver_lp
# ---------------------------------------------------------------------------

def _restricciones():
    """Todas las restricciones del modelo base como filas G·x <= h."""
    A, b = nio.A_RECURSOS, nio.RECURSOS_BASE
    G = [*A]
    h = [*b]
    for j, (lo, hi) in enumerate(nio.COTAS_BARCOS):
        e = np.zeros(3)
        e[j] = 1
        G += [e, -e]
        h += [hi, -lo]
    return np.array(G, dtype=float), np.array(h, dtype=float)


def optimo_por_vertices(c):
    """Enumera todos los vértices del poliedro (3 restricciones activas) y
    devuelve el de mayor objetivo. Es el razonamiento geométrico del Simplex,
    hecho por fuerza bruta."""
    G, h = _restricciones()
    mejor_x, mejor_z = None, -np.inf
    for filas in itertools.combinations(range(len(G)), 3):
        M = G[list(filas)]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        x = np.linalg.solve(M, h[list(filas)])
        if np.all(G @ x <= h + 1e-7):
            z = float(np.dot(c, x))
            if z > mejor_z + 1e-9:
                mejor_x, mejor_z = x, z
    return mejor_x, mejor_z


def precios_sombra_por_dual(c):
    """Resuelve explícitamente el problema dual:
        min b·y + u·z   s.a.  Aᵀy + z >= c,  y, z >= 0
    donde y son los precios sombra de los recursos y z los de las cotas superiores."""
    A, b = nio.A_RECURSOS, nio.RECURSOS_BASE
    u = np.array([hi for _, hi in nio.COTAS_BARCOS], dtype=float)
    obj = np.concatenate([b, u])
    A_dual = -np.hstack([A.T, np.eye(3)])          # -(Aᵀy + z) <= -c
    res = linprog(obj, A_ub=A_dual, b_ub=-np.asarray(c), bounds=[(0, None)] * 6,
                  method="highs-ds")
    assert res.success
    return res.x[:3], res.x[3:], res.fun


def precios_sombra_por_diferencias(c, h=1.0):
    """Derivada numérica del objetivo óptimo respecto a cada recurso."""
    base = optimo_por_vertices(c)[1]
    precios = []
    for i in range(3):
        b = nio.RECURSOS_BASE.copy()
        b[i] += h
        r = linprog(-np.asarray(c), A_ub=nio.A_RECURSOS, b_ub=b,
                    bounds=nio.COTAS_BARCOS, method="highs-ipm")
        precios.append((-r.fun - base) / h)
    return np.array(precios)


# ---------------------------------------------------------------------------
# Programación lineal
# ---------------------------------------------------------------------------

def test_lp_base_misma_solucion_que_referencia():
    sol = nio.resolver_lp()
    x_ref, z_ref = optimo_por_vertices(nio.RENDIMIENTO_NOMINAL)
    assert sol["exito"]
    np.testing.assert_allclose(sol["x"], x_ref, atol=1e-7)
    assert sol["objetivo"] == pytest.approx(z_ref)
    # Cifras del informe: 20 pequeños, 6 medianos, 10 grandes -> 3720 kg
    np.testing.assert_allclose(sol["x"], [20, 6, 10], atol=1e-7)
    assert sol["objetivo"] == pytest.approx(3720.0)


def test_lp_base_mismos_precios_sombra_que_referencia():
    sol = nio.resolver_lp()
    y_dual, z_dual, obj_dual = precios_sombra_por_dual(nio.RENDIMIENTO_NOMINAL)
    np.testing.assert_allclose(sol["precios_sombra"], y_dual, atol=1e-7)
    np.testing.assert_allclose(sol["valor_barco_extra"], z_dual, atol=1e-7)
    # Dualidad fuerte: el valor del dual coincide con el del primal
    assert obj_dual == pytest.approx(sol["objetivo"])
    # Y coinciden con la derivada numérica (la solución no es degenerada)
    np.testing.assert_allclose(sol["precios_sombra"],
                               precios_sombra_por_diferencias(nio.RENDIMIENTO_NOMINAL),
                               atol=1e-6)
    # Solo la tripulación limita: 24 kg por persona adicional
    np.testing.assert_allclose(sol["precios_sombra"], [0, 24, 0], atol=1e-9)


def test_holgura_complementaria():
    sol = nio.resolver_lp()
    # Un recurso con holgura tiene precio sombra cero
    assert np.all(np.abs(sol["holgura"] * sol["precios_sombra"]) < 1e-6)


def test_lp_otro_escenario_coincide_con_referencia():
    c = nio.rendimiento_verdadero(18, 22, 3, 0.7)[0]
    sol = nio.resolver_lp(c)
    _, z_ref = optimo_por_vertices(c)
    y_dual, _, _ = precios_sombra_por_dual(c)
    assert sol["objetivo"] == pytest.approx(z_ref)
    np.testing.assert_allclose(sol["precios_sombra"], y_dual, atol=1e-7)


# ---------------------------------------------------------------------------
# Ningún R² se recorta
# ---------------------------------------------------------------------------

def test_metricas_no_recortan_r2_negativo():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    malo = np.array([4.0, 3.0, 2.0, 1.0])      # peor que la media
    assert nio.metricas_regresion(y, malo)["r2"] == pytest.approx(-3.0)


@pytest.fixture(scope="module")
def analizador():
    a = AnalizadorComparativoIO(semilla=42, n_escenarios=300, n_historico_por_tipo=200,
                                n_escenarios_decision=30, verbose=False)
    a.ejecutar_analisis_completo(generar_archivos=False)
    return a


def test_r2_del_analisis_igual_a_sklearn(analizador):
    from sklearn.metrics import r2_score
    y_te, preds = analizador._pred_test
    for fila in analizador.resultados["prediccion"]["tabla"]:
        assert fila["r2_test"] == pytest.approx(r2_score(y_te, preds[fila["modelo"]]))
    # La media aplicada al test da un R² negativo y debe aparecer como tal
    media = next(f for f in analizador.resultados["prediccion"]["tabla"]
                 if f["modelo"] == NOMBRE_MEDIA)
    assert media["r2_test"] < 0


def test_codigo_no_recorta_r2():
    """Ningún script aplica max(0, ...) ni clip(...) a un R²."""
    for archivo in RAIZ.glob("*.py"):
        arbol = ast.parse(archivo.read_text(encoding="utf-8"))
        for nodo in ast.walk(arbol):
            if not isinstance(nodo, ast.Call):
                continue
            nombre = getattr(nodo.func, "id", getattr(nodo.func, "attr", ""))
            if nombre not in {"max", "clip", "maximum"}:
                continue
            # Un recorte compara con una constante: max(0, r2), np.clip(r2, 0, 1)...
            # (max(..., key=lambda x: x['r2']) elige el mejor modelo y no recorta)
            args = list(nodo.args) + [k.value for k in nodo.keywords if k.arg != "key"]
            con_constante = any(isinstance(a, ast.Constant) for a in args)
            texto = " ".join(ast.unparse(a) for a in args).lower()
            assert not (con_constante and ("r2" in texto or "r²" in texto)), \
                f"{archivo.name}: posible recorte de R²: {ast.unparse(nodo)}"


def test_hibrido_se_evalua_por_calidad_de_decision(analizador):
    tabla = analizador.resultados["hibrido"]["tabla"]
    for fila in tabla:
        assert 0 < fila["pct_optimo_media"] <= 100 + 1e-9
        assert "r2_capturas" not in fila
