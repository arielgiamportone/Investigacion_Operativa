# -*- coding: utf-8 -*-
"""Prueba de humo de la app Streamlit: recorre el flujo completo sin excepciones."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parent.parent / "app_streamlit_io.py"


def _sin_excepciones(at):
    assert not at.exception, [e.value for e in at.exception]


def test_app_flujo_completo():
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    _sin_excepciones(at)

    botones = {b.label: b for b in at.sidebar.button}
    botones["🔄 Generar Datos"].click().run()
    _sin_excepciones(at)
    {b.label: b for b in at.sidebar.button}["🎯 Entrenar Modelos ML"].click().run()
    _sin_excepciones(at)

    # Botones de las pestañas: optimizar con Simplex y predecir con ML
    for boton in at.button:
        boton.click().run()
        _sin_excepciones(at)

    tablas = [df.value for df in at.dataframe]

    # Pestaña Simplex: precios sombra del escenario base (solo limita la tripulación)
    recursos = next(t for t in tablas if "Precio sombra (kg/unidad)" in t.columns)
    assert list(recursos["Precio sombra (kg/unidad)"]) == pytest.approx([0, 24, 0])

    # Predecir: incluye las líneas base y los R² no se recortan (la media sale negativa)
    modelos = next(t for t in tablas if "R² test" in t.columns).set_index("Modelo")
    assert {"Media (línea base)", "Regresión lineal"} <= set(modelos.index)
    assert modelos.loc["Media (línea base)", "R² test"] < 0

    # Predecir y optimizar: los rendimientos verdaderos dan el 100 % del óptimo
    decision = next(t for t in tablas if "% del óptimo" in t.columns).set_index(
        "Origen de los coeficientes")
    assert decision.loc["Rendimientos verdaderos (referencia)", "% del óptimo"] == pytest.approx(100)
    assert (decision["% del óptimo"] <= 100 + 1e-6).all()
