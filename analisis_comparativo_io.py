#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis comparativo: decidir (Simplex) y predecir (Machine Learning)
=====================================================================

Curso: CONIPE 2025 - Alfabetización en Ciencia de Datos

El Simplex y el Machine Learning NO compiten en la misma carrera:

1. **Decidir (programación lineal / Simplex).** Con unos rendimientos y unos
   recursos dados, ¿cuántos barcos de cada tipo conviene enviar? Se evalúa con
   la solución óptima, los recursos que limitan, los precios sombra y el
   análisis de sensibilidad. No tiene sentido medirlo con R² o MSE.
2. **Predecir (Machine Learning).** A partir de las condiciones, ¿cuánto se va
   a capturar? Se evalúa con R², MSE y validación cruzada, frente a líneas
   base honestas (la media y una regresión lineal).
3. **Predecir y luego optimizar (híbrido).** El ML estima el rendimiento de
   cada tipo de barco según el clima y la PL decide con esas estimaciones.
   Se evalúa por la *calidad de la decisión*: captura obtenida frente a la
   que se habría obtenido decidiendo con los rendimientos verdaderos.

⚠️ Todos los datos son SINTÉTICOS (ver `nucleo_io.py`): se generan con una
fórmula conocida más ruido. Un buen R² solo demuestra que el modelo recupera
esa fórmula; no dice nada sobre pesquerías reales.

Salidas:
- reporte_ejecutivo_analisis_io.html   (informe para leer)
- analisis_comparativo_io_completo.html (gráficos interactivos)
- resultados_analisis_io.json           (todas las cifras del informe)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import nucleo_io as nio
from nucleo_io import fmt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DIRECTORIO = Path(__file__).resolve().parent

NOMBRE_MEDIA = "Media (línea base)"
NOMBRE_LINEAL = "Regresión lineal"


def modelos_prediccion(semilla=42):
    """Modelos a comparar. Los que necesitan escalado llevan el escalador dentro
    de un pipeline, para que la validación cruzada no "vea" los datos de test."""
    return {
        NOMBRE_MEDIA: DummyRegressor(strategy="mean"),
        NOMBRE_LINEAL: make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=semilla),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=semilla),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale")),
        "Red neuronal (MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=semilla)),
    }


class AnalizadorComparativoIO:
    def __init__(self, semilla=42, n_escenarios=1000, n_historico_por_tipo=1000,
                 n_escenarios_decision=300, verbose=True):
        self.semilla = semilla
        self.n_escenarios = n_escenarios
        self.n_historico_por_tipo = n_historico_por_tipo
        self.n_escenarios_decision = n_escenarios_decision
        self.verbose = verbose
        self.resultados = {}

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # ------------------------------------------------------------------
    # 1. DECIDIR
    # ------------------------------------------------------------------
    def analizar_decision(self):
        """PL del modelo base: solución, recursos limitantes, precios sombra y sensibilidad."""
        self._log("🏛️ 1. DECIDIR: programación lineal (HiGHS)")
        sol = nio.resolver_lp(nio.RENDIMIENTO_NOMINAL, nio.RECURSOS_BASE)
        if not sol["exito"]:
            raise RuntimeError(f"La PL base no tiene solución: {sol['mensaje']}")

        recursos = []
        for i, nombre in enumerate(nio.RECURSOS):
            inf, sup, ps = nio.rango_precio_sombra(i)
            recursos.append({
                "recurso": nombre,
                "unidad": nio.UNIDADES_RECURSOS[i],
                "unidad_singular": nio.UNIDAD_SINGULAR[i],
                "disponible": float(nio.RECURSOS_BASE[i]),
                "usado": float(sol["uso"][i]),
                "holgura": float(sol["holgura"][i]),
                "limita": bool(np.isclose(sol["holgura"][i], 0, atol=1e-7)),
                "precio_sombra": float(ps),
                "rango_validez": [float(inf), None if np.isinf(sup) else float(sup)],
            })

        # Sensibilidad: objetivo al variar cada recurso entre 50 % y 150 % del base
        factores = np.linspace(0.5, 1.5, 41)
        sensibilidad = {}
        for i, nombre in enumerate(nio.RECURSOS):
            df = nio.sensibilidad_recurso(i, nio.RECURSOS_BASE[i] * factores)
            sensibilidad[nombre] = {"factor": factores.tolist(),
                                    "objetivo": df["objetivo"].tolist()}

        self.resultados["decision"] = {
            "coeficientes": nio.RENDIMIENTO_NOMINAL.tolist(),
            "solucion": dict(zip(nio.TIPOS_BARCO, map(float, sol["x"]))),
            "objetivo_kg": sol["objetivo"],
            "valor_barco_extra": dict(zip(nio.TIPOS_BARCO, map(float, sol["valor_barco_extra"]))),
            "recursos": recursos,
            "sensibilidad": sensibilidad,
        }
        self._log(f"   Solución: {sol['x']}  ->  {sol['objetivo']:.1f} kg")
        for r in recursos:
            self._log(f"   {r['recurso']:<12} holgura={r['holgura']:>10.2f}  "
                      f"precio sombra={r['precio_sombra']:.3f} kg/{r['unidad']}")
        return self.resultados["decision"]

    # ------------------------------------------------------------------
    # 2. PREDECIR
    # ------------------------------------------------------------------
    def evaluar_prediccion(self):
        """Modelos de ML frente a la media y la regresión lineal. R² sin recortar."""
        self._log("🧠 2. PREDECIR: capturas a partir de las condiciones")
        datos = nio.generar_datos_flota(self.n_escenarios, self.semilla)
        X = datos[nio.VARIABLES_PREDICCION]
        y = datos["capturas_totales"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=self.semilla)

        filas, predicciones = [], {}
        for nombre, modelo in modelos_prediccion(self.semilla).items():
            cv = cross_val_score(modelo, X_tr, y_tr, cv=5, scoring="r2")
            modelo.fit(X_tr, y_tr)
            y_pred = modelo.predict(X_te)
            m = nio.metricas_regresion(y_te, y_pred)
            filas.append({"modelo": nombre, "r2_test": m["r2"], "mse_test": m["mse"],
                          "mae_test": m["mae"], "cv_r2_media": float(cv.mean()),
                          "cv_r2_desv": float(cv.std())})
            predicciones[nombre] = y_pred
            self._log(f"   {nombre:<22} R²={m['r2']:.3f}  CV={cv.mean():.3f}±{cv.std():.3f}")

        # Techo: la fórmula verdadera sin ruido. No es un modelo: es lo máximo
        # que puede explicar cualquier modelo, porque el resto es ruido.
        techo = nio.metricas_regresion(y_te, datos.loc[y_te.index, "capturas_sin_ruido"])
        self._log(f"   Techo (fórmula verdadera) R²={techo['r2']:.3f}")

        df = pd.DataFrame(filas)
        ml = df[~df["modelo"].isin([NOMBRE_MEDIA, NOMBRE_LINEAL])]
        mejor = ml.loc[ml["r2_test"].idxmax()]
        self.resultados["prediccion"] = {
            "tabla": filas,
            "techo": techo,
            "mejor_modelo": mejor["modelo"],
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        }
        self._pred_test = (y_te.to_numpy(), predicciones)
        return self.resultados["prediccion"]

    # ------------------------------------------------------------------
    # 3. PREDECIR Y LUEGO OPTIMIZAR
    # ------------------------------------------------------------------
    def evaluar_hibrido(self):
        """ML estima el rendimiento por tipo de barco; la PL decide con esas
        estimaciones. Se mide la calidad de la decisión, no el R²."""
        self._log("🚀 3. PREDECIR Y LUEGO OPTIMIZAR")
        hist = nio.generar_historico_salidas(self.n_historico_por_tipo, self.semilla + 1)
        estimadores = {
            NOMBRE_LINEAL: lambda: make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=self.semilla),
            "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, random_state=self.semilla),
        }

        # Escenarios nuevos (clima + recursos) en los que hay que decidir
        rng = np.random.default_rng(self.semilla + 2)
        escenarios = nio.generar_condiciones(self.n_escenarios_decision, rng)
        rend_real = nio.rendimiento_verdadero(escenarios["temperatura_agua"],
                                              escenarios["velocidad_viento"],
                                              escenarios["mes"], escenarios["fase_lunar"])
        recursos = escenarios[["combustible_disponible", "tripulacion_disponible",
                               "presupuesto"]].to_numpy(dtype=float)

        # Rendimientos estimados por cada estrategia, matriz (n, 3)
        estimaciones = {"Coeficientes nominales (sin ML)":
                        np.tile(nio.RENDIMIENTO_NOMINAL, (len(escenarios), 1))}
        r2_rendimiento = {}
        for nombre, crear in estimadores.items():
            est = np.zeros((len(escenarios), 3))
            r2s = []
            for t, tipo in enumerate(nio.TIPOS_BARCO):
                h = hist[hist["tipo"] == tipo]
                X_tr, X_te, y_tr, y_te = train_test_split(
                    h[nio.VARIABLES_CLIMA], h["captura_por_barco"],
                    test_size=0.3, random_state=self.semilla)
                modelo = crear().fit(X_tr, y_tr)
                r2s.append(nio.metricas_regresion(y_te, modelo.predict(X_te))["r2"])
                est[:, t] = modelo.predict(escenarios[nio.VARIABLES_CLIMA])
            estimaciones[f"{nombre} + PL"] = est
            r2_rendimiento[f"{nombre} + PL"] = dict(zip(nio.TIPOS_BARCO, r2s))

        # Óptimo con los rendimientos verdaderos (referencia, 100 %)
        optimo = np.array([nio.resolver_lp(rend_real[i], recursos[i])["objetivo"]
                           for i in range(len(escenarios))])

        filas, ratios = [], {}
        for nombre, est in estimaciones.items():
            obtenido = np.array([rend_real[i] @ nio.resolver_lp(est[i], recursos[i])["x"]
                                 for i in range(len(escenarios))])
            ratio = 100 * obtenido / optimo
            ratios[nombre] = ratio
            filas.append({
                "estrategia": nombre,
                "pct_optimo_media": float(ratio.mean()),
                "pct_optimo_mediana": float(np.median(ratio)),
                "pct_optimo_peor": float(ratio.min()),
                "pct_escenarios_optimos": float(100 * np.mean(ratio >= 99.9)),
                "kg_perdidos_media": float((optimo - obtenido).mean()),
                "r2_rendimiento": r2_rendimiento.get(nombre),
            })
            self._log(f"   {nombre:<34} {ratio.mean():6.2f} % del óptimo "
                      f"(peor caso {ratio.min():.1f} %)")

        self.resultados["hibrido"] = {
            "tabla": filas,
            "n_escenarios": int(len(escenarios)),
            "n_historico": int(len(hist)),
            "optimo_medio_kg": float(optimo.mean()),
        }
        self._ratios = ratios
        return self.resultados["hibrido"]

    # ------------------------------------------------------------------
    # Conclusiones (se derivan de los resultados)
    # ------------------------------------------------------------------
    def conclusiones(self):
        d, p, h = (self.resultados[k] for k in ("decision", "prediccion", "hibrido"))
        textos = {}

        limitantes = [r for r in d["recursos"] if r["limita"]]
        no_limitan = [r for r in d["recursos"] if not r["limita"]]
        frase = "; ".join(
            f"<strong>{r['recurso']}</strong> (precio sombra {fmt(r['precio_sombra'], 2)} kg por "
            f"{r['unidad_singular']}, válido aproximadamente "
            + (f"entre {fmt(r['rango_validez'][0], 0)} y {fmt(r['rango_validez'][1], 0)})"
               if r["rango_validez"][1] is not None
               else f"desde {fmt(r['rango_validez'][0], 0)} sin límite superior)")
            for r in limitantes) or "ningún recurso"
        textos["decidir"] = (
            f"Con los recursos del escenario base, el plan óptimo captura "
            f"{fmt(d['objetivo_kg'], 1)} kg. El recurso que limita es {frase}. "
            + ("Sobran " + " y ".join(f"{fmt(r['holgura'], 0)} {r['unidad']} de {r['recurso']}"
                                      for r in no_limitan)
               + ": conseguir más de esos recursos no aumentaría la captura (precio sombra 0). "
               if no_limitan else "")
        )

        tabla = {f["modelo"]: f for f in p["tabla"]}
        mejor = tabla[p["mejor_modelo"]]
        lineal = tabla[NOMBRE_LINEAL]
        media = tabla[NOMBRE_MEDIA]
        dif = mejor["r2_test"] - lineal["r2_test"]
        negativos = [f["modelo"] for f in p["tabla"]
                     if f["r2_test"] < 0 and f["modelo"] != NOMBRE_MEDIA]
        txt = (f"El mejor modelo de ML es <strong>{p['mejor_modelo']}</strong> "
               f"(R² = {fmt(mejor['r2_test'], 3)} en test). La regresión lineal obtiene "
               f"{fmt(lineal['r2_test'], 3)} y la media {fmt(media['r2_test'], 3)}. ")
        if dif <= 0.01:
            txt += ("La regresión lineal empata o gana: con estos datos, la complejidad "
                    "adicional del ML no aporta. ")
        else:
            txt += (f"La regresión lineal no empata (queda {fmt(dif, 3)} por debajo): la captura "
                    "es lineal en el número de barcos pero se multiplica por factores "
                    "climáticos, y un modelo lineal en las variables originales no puede "
                    "representar ese producto. ")
        txt += (f"El techo alcanzable es R² = {fmt(p['techo']['r2'], 3)} (la fórmula "
                "verdadera sin ruido): el ML no descubre nada nuevo, solo se acerca a la "
                "fórmula con la que se generaron los datos. ")
        if media["r2_test"] < 0:
            txt += ("La media sale ligeramente negativa porque usa la media del entrenamiento, "
                    "no la del test: es el R² que corresponde a «no saber nada». ")
        if negativos:
            txt += ("Hay R² negativos (" + ", ".join(negativos) + "): esos modelos predicen "
                    "peor que la media y se muestran tal cual.")
        textos["predecir"] = txt

        th = {f["estrategia"]: f for f in h["tabla"]}
        nominal = th["Coeficientes nominales (sin ML)"]
        mejor_h = max((f for f in h["tabla"] if f is not nominal),
                      key=lambda f: f["pct_optimo_media"])
        lineal_h = th[f"{NOMBRE_LINEAL} + PL"]
        mejora = mejor_h["pct_optimo_media"] - nominal["pct_optimo_media"]
        txt = (f"Decidir con los rendimientos nominales (los de un día en calma) logra en "
               f"promedio el {fmt(nominal['pct_optimo_media'], 1)} % de la captura óptima, "
               f"con un peor caso del {fmt(nominal['pct_optimo_peor'], 1)} %. ")
        if mejora > 0.5:
            txt += (f"Estimar los rendimientos con ML y decidir con la PL eleva la media al "
                    f"{fmt(mejor_h['pct_optimo_media'], 1)} % ({mejor_h['estrategia']}; peor caso "
                    f"{fmt(mejor_h['pct_optimo_peor'], 1)} %). Con la regresión lineal se llega al "
                    f"{fmt(lineal_h['pct_optimo_media'], 1)} %. ")
        else:
            txt += "Estimar los rendimientos con ML apenas cambia la calidad de la decisión. "
        txt += ("La mejora existe porque, en este mundo sintético, el viento afecta de forma "
                "distinta a cada tipo de barco y eso cambia cuál es la mejor combinación. Si "
                "el clima escalara por igual todos los rendimientos, la decisión óptima no "
                "cambiaría y la predicción no aportaría nada a la decisión.")
        textos["hibrido"] = txt
        textos["mejora_hibrido_pp"] = mejora
        self.resultados["conclusiones"] = textos
        return textos

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------
    def crear_visualizaciones(self, archivo="analisis_comparativo_io_completo.html"):
        self._log("🎨 Creando visualizaciones...")
        d, p, h = (self.resultados[k] for k in ("decision", "prediccion", "hibrido"))
        fig = make_subplots(
            rows=3, cols=2, vertical_spacing=0.12, horizontal_spacing=0.12,
            subplot_titles=[
                "1. Decidir: plan óptimo (barcos por tipo)",
                "1. Decidir: uso de cada recurso (% de lo disponible)",
                "1. Decidir: captura óptima al variar cada recurso",
                "2. Predecir: R² en test (con desviación de la validación cruzada)",
                "2. Predecir: predicción vs. real (test)",
                "3. Predecir y optimizar: % de la captura óptima por escenario",
            ])

        tipos = list(d["solucion"].keys())
        fig.add_trace(go.Bar(x=tipos, y=list(d["solucion"].values()),
                             text=[fmt(v, 1) for v in d["solucion"].values()],
                             textposition="auto", marker_color="#4C78A8",
                             name="Barcos", showlegend=False), row=1, col=1)

        rec = d["recursos"]
        fig.add_trace(go.Bar(
            x=[f"{r['recurso']}<br>(precio sombra {fmt(r['precio_sombra'], 2)})" for r in rec],
            y=[100 * r["usado"] / r["disponible"] for r in rec],
            text=[f"{fmt(100 * r['usado'] / r['disponible'], 0)} %" for r in rec],
            textposition="outside", textangle=0, showlegend=False,
            marker_color=["#E45756" if r["limita"] else "#9DA5B4" for r in rec],
            name="Uso"), row=1, col=2)

        colores = {"combustible": "#4C78A8", "tripulación": "#E45756", "presupuesto": "#54A24B"}
        for nombre, s in d["sensibilidad"].items():
            fig.add_trace(go.Scatter(x=[100 * f for f in s["factor"]], y=s["objetivo"],
                                     mode="lines", name=nombre,
                                     line=dict(color=colores[nombre], width=2)),
                          row=2, col=1)

        tabla = p["tabla"]
        fig.add_trace(go.Bar(
            x=[f["modelo"] for f in tabla], y=[f["r2_test"] for f in tabla],
            error_y=dict(type="data", array=[f["cv_r2_desv"] for f in tabla]),
            text=[fmt(f["r2_test"], 3) for f in tabla], textposition="outside",
            marker_color=["#9DA5B4" if f["modelo"] in (NOMBRE_MEDIA, NOMBRE_LINEAL)
                          else "#4C78A8" for f in tabla],
            name="R² test", showlegend=False), row=2, col=2)
        fig.add_hline(y=p["techo"]["r2"], line_dash="dash", line_color="#54A24B",
                      annotation_text="techo (fórmula verdadera)", row=2, col=2)

        y_te, preds = self._pred_test
        for nombre, color in ((p["mejor_modelo"], "#4C78A8"), (NOMBRE_LINEAL, "#F58518")):
            fig.add_trace(go.Scatter(x=y_te, y=preds[nombre], mode="markers",
                                     marker=dict(size=4, opacity=0.5, color=color),
                                     name=nombre), row=3, col=1)
        lim = [0, float(max(y_te.max(), max(v.max() for v in preds.values())))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="predicción perfecta",
                                 line=dict(color="black", dash="dash")), row=3, col=1)

        for nombre, ratio in self._ratios.items():
            fig.add_trace(go.Box(y=ratio, name=nombre.replace(" + PL", "<br>+ PL").replace("Coeficientes nominales (sin ML)", "Nominales<br>(sin ML)"),
                                 boxpoints="outliers", showlegend=False), row=3, col=2)

        fig.update_yaxes(title_text="barcos", row=1, col=1)
        fig.update_yaxes(title_text="% usado", range=[0, 115], row=1, col=2)
        fig.update_xaxes(title_text="% del valor base del recurso", row=2, col=1)
        fig.update_yaxes(title_text="captura óptima (kg)", row=2, col=1)
        fig.update_yaxes(title_text="R²", row=2, col=2)
        fig.update_xaxes(title_text="captura real (kg)", row=3, col=1)
        fig.update_yaxes(title_text="captura predicha (kg)", row=3, col=1)
        fig.update_yaxes(title_text="% del óptimo", row=3, col=2)
        fig.update_layout(
            height=1400,
            title_text=("Decidir y predecir: dos preguntas distintas "
                        "<br><sup>Datos sintéticos generados con una fórmula conocida "
                        "(ver nucleo_io.py)</sup>"),
            title_x=0.5, showlegend=True, legend=dict(orientation="h", y=-0.05))

        ruta = DIRECTORIO / archivo
        pyo.plot(fig, filename=str(ruta), auto_open=False)
        self._log(f"✅ Visualización guardada: {archivo}")
        return str(ruta)

    # ------------------------------------------------------------------
    # Informe HTML
    # ------------------------------------------------------------------
    def generar_reporte_ejecutivo(self, archivo="reporte_ejecutivo_analisis_io.html"):
        self._log("📋 Generando reporte ejecutivo...")
        d, p, h = (self.resultados[k] for k in ("decision", "prediccion", "hibrido"))
        c = self.resultados.get("conclusiones") or self.conclusiones()

        filas_sol = "".join(
            f"<tr><td>{t}</td><td>{fmt(nio.RENDIMIENTO_NOMINAL[i], 0)}</td>"
            f"<td>{fmt(d['solucion'][t], 2)}</td><td>{nio.COTAS_BARCOS[i][1]}</td>"
            f"<td>{fmt(d['valor_barco_extra'][t], 2)}</td></tr>"
            for i, t in enumerate(nio.TIPOS_BARCO))
        filas_rec = "".join(
            f"<tr class='{'limita' if r['limita'] else ''}'><td>{r['recurso']} ({r['unidad']})</td>"
            f"<td>{fmt(r['disponible'], 0)}</td><td>{fmt(r['usado'], 1)}</td>"
            f"<td>{fmt(r['holgura'], 1)}</td><td>{'sí' if r['limita'] else 'no'}</td>"
            f"<td>{fmt(r['precio_sombra'], 3)}</td>"
            f"<td>{fmt(r['rango_validez'][0], 0)} – "
            f"{fmt(r['rango_validez'][1], 0) if r['rango_validez'][1] is not None else '∞'}</td></tr>"
            for r in d["recursos"])
        filas_pred = "".join(
            f"<tr class='{'base' if f['modelo'] in (NOMBRE_MEDIA, NOMBRE_LINEAL) else ''}"
            f"{' neg' if f['r2_test'] < 0 else ''}'>"
            f"<td>{f['modelo']}</td><td>{fmt(f['r2_test'], 3)}</td>"
            f"<td>{fmt(f['cv_r2_media'], 3)} ± {fmt(f['cv_r2_desv'], 3)}</td>"
            f"<td>{fmt(f['mse_test'], 0)}</td><td>{fmt(f['mae_test'], 1)}</td></tr>"
            for f in p["tabla"])
        filas_pred += (f"<tr class='techo'><td>Techo: fórmula verdadera sin ruido (no es un modelo)</td>"
                       f"<td>{fmt(p['techo']['r2'], 3)}</td><td>—</td>"
                       f"<td>{fmt(p['techo']['mse'], 0)}</td><td>{fmt(p['techo']['mae'], 1)}</td></tr>")

        def r2_rend(f):
            if not f["r2_rendimiento"]:
                return "—"
            return " / ".join(fmt(v, 3) for v in f["r2_rendimiento"].values())

        filas_h = "".join(
            f"<tr><td>{f['estrategia']}</td><td>{fmt(f['pct_optimo_media'], 2)} %</td>"
            f"<td>{fmt(f['pct_optimo_mediana'], 2)} %</td><td>{fmt(f['pct_optimo_peor'], 1)} %</td>"
            f"<td>{fmt(f['pct_escenarios_optimos'], 1)} %</td>"
            f"<td>{fmt(f['kg_perdidos_media'], 1)}</td><td>{r2_rend(f)}</td></tr>"
            for f in h["tabla"])

        mejora = c["mejora_hibrido_pp"]
        recomendacion_hibrido = (
            "Cuando un coeficiente del modelo de decisión es incierto y depende de "
            "condiciones observables (aquí, el rendimiento de cada tipo de barco según el "
            "clima), conviene estimarlo con datos y luego optimizar. En este análisis eso "
            f"mejora la decisión en {fmt(mejora, 1)} puntos porcentuales de media."
            if mejora > 0.5 else
            "En este análisis, estimar los coeficientes con ML no mejora la decisión de forma "
            "apreciable: la complejidad extra no se justifica.")

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Decidir y predecir en IO</title>
<style>
  :root {{ --fondo:#f5f5f5; --papel:#fff; --texto:#1f2933; --suave:#52606d; --borde:#d9dee3;
          --acento:#2f6db5; --ok:#e7f4ea; --ok-borde:#2e8540; --aviso:#fff4d6; --aviso-borde:#c78a00;
          --limita:#fde8e8; --neg:#b42318; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fondo:#111418; --papel:#1b2027; --texto:#e4e7eb; --suave:#9aa5b1; --borde:#323a45;
            --acento:#7fb0f0; --ok:#173323; --ok-borde:#4caf6e; --aviso:#3a2f12; --aviso-borde:#d9a520;
            --limita:#3d1f1f; --neg:#ff8a80; }}
  }}
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', Arial, sans-serif; margin:0; padding:24px 16px;
         background:var(--fondo); color:var(--texto); line-height:1.55; }}
  .container {{ max-width:1000px; margin:0 auto; background:var(--papel); padding:32px;
               border-radius:10px; border:1px solid var(--borde); }}
  h1 {{ margin-top:0; border-bottom:3px solid var(--acento); padding-bottom:16px; }}
  h2 {{ margin-top:36px; }}
  .nota {{ color:var(--suave); font-size:.92em; }}
  .aviso {{ background:var(--aviso); border-left:4px solid var(--aviso-borde); padding:14px 18px; margin:16px 0; }}
  .conclusion {{ background:var(--ok); border-left:4px solid var(--ok-borde); padding:14px 18px; margin:16px 0; }}
  .tabla {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:.95em; }}
  th, td {{ border:1px solid var(--borde); padding:8px 10px; text-align:left; }}
  th {{ background:var(--acento); color:#fff; }}
  td:not(:first-child) {{ text-align:right; font-variant-numeric: tabular-nums; }}
  tr.limita {{ background:var(--limita); font-weight:600; }}
  tr.base td:first-child {{ font-style:italic; }}
  tr.neg td:nth-child(2) {{ color:var(--neg); font-weight:700; }}
  tr.techo {{ color:var(--suave); }}
  code {{ background:var(--fondo); padding:1px 4px; border-radius:3px; }}
  .dos {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .dos > div {{ border:1px solid var(--borde); border-radius:8px; padding:12px 16px; }}
  @media (max-width: 640px) {{ .dos {{ grid-template-columns:1fr; }} .container {{ padding:18px; }} }}
</style>
</head>
<body>
<div class="container">
<h1>Decidir y predecir: Simplex y Machine Learning en una flota pesquera</h1>
<p class="nota">Material docente · CONIPE 2025 · Generado el {datetime.now().strftime('%d/%m/%Y')} con
<code>analisis_comparativo_io.py</code> (semilla {self.semilla}). Todas las cifras salen de ejecutar ese script.</p>

<div class="aviso">
<strong>⚠️ Los datos son sintéticos.</strong> No proceden de ninguna pesquería real. Se generan con
<code>nucleo_io.py</code>: el rendimiento de cada tipo de barco (50, 120 y 200 kg por salida en calma) se
multiplica por factores de temperatura, fase lunar, estación y viento (el viento penaliza más a los barcos
pequeños), y a la captura se le suma un ruido aleatorio del 10 % (15 % en el histórico por barco). La salinidad,
la altura de olas y las variables de mercado no influyen en la captura. Un buen R² solo demuestra que un modelo
recupera esa fórmula; no dice nada sobre el mundo real.
</div>

<h2>Dos preguntas distintas</h2>
<div class="dos">
  <div><h3>🏛️ Decidir (Simplex / PL)</h3>
  <p>Con unos rendimientos y unos recursos dados, ¿cuántos barcos de cada tipo enviar? Se evalúa por la
  solución óptima, los recursos que limitan y los precios sombra. <strong>No se mide con R² ni MSE</strong>:
  no predice nada.</p></div>
  <div><h3>🧠 Predecir (Machine Learning)</h3>
  <p>A partir de las condiciones, ¿cuánto se va a capturar? Se evalúa con R², MSE y validación cruzada,
  frente a líneas base honestas. <strong>No decide nada</strong> por sí solo.</p></div>
</div>
<p>Por eso no tiene sentido ponerlos en la misma tabla de precisión. Lo que sí se puede comparar es
<em>cómo de buena es una decisión</em> según de dónde salgan sus coeficientes (sección 3).</p>

<h2>1. Decidir: programación lineal</h2>
<p>Maximizar la captura esperada <code>50·P + 120·M + 200·G</code> (kg), sujeto a combustible
(<code>20P + 50M + 100G ≤ {fmt(nio.RECURSOS_BASE[0], 0)}</code> L), tripulación
(<code>2P + 5M + 8G ≤ {fmt(nio.RECURSOS_BASE[1], 0)}</code>), presupuesto
(<code>1000P + 2500M + 5000G ≤ {fmt(nio.RECURSOS_BASE[2], 0)}</code> $) y barcos disponibles
(P ≤ 20, M ≤ 15, G ≤ 10). Se resuelve con HiGHS (<code>scipy.optimize.linprog</code>).</p>
<div class="tabla"><table>
<tr><th>Tipo de barco</th><th>kg por barco</th><th>Barcos en el óptimo</th><th>Disponibles</th>
<th>kg extra por un barco más</th></tr>
{filas_sol}
<tr><td><strong>Captura óptima</strong></td><td></td><td><strong>{fmt(d['objetivo_kg'], 1)} kg</strong></td><td></td><td></td></tr>
</table></div>
<div class="tabla"><table>
<tr><th>Recurso</th><th>Disponible</th><th>Usado</th><th>Holgura</th><th>¿Limita?</th>
<th>Precio sombra (kg por unidad)</th><th>Rango donde vale ese precio</th></tr>
{filas_rec}
</table></div>
<p class="nota">Precio sombra: cuántos kg más se capturarían con una unidad más del recurso
(<code>res.ineqlin.marginals</code> de HiGHS, con el signo cambiado porque <code>linprog</code> minimiza).
El rango se obtiene re-resolviendo la PL con el recurso desplazado hasta que el precio cambia. En este
modelo el presupuesto por barco es exactamente 50 veces el combustible por barco, así que ambas restricciones
son proporcionales: solo puede limitar la más estricta de las dos. La solución admite barcos fraccionarios
(relajación lineal); exigir enteros sería un problema de programación entera.</p>
<div class="conclusion">{c['decidir']}</div>

<h2>2. Predecir: capturas a partir de las condiciones</h2>
<p>{p['n_train']} escenarios de entrenamiento y {p['n_test']} de test. Variables de entrada: barcos de cada tipo,
recursos, clima, mercado, mes y fase lunar. R² de validación cruzada con 5 particiones sobre el conjunto de
entrenamiento. Los R² <strong>no se recortan</strong>: un valor negativo indica un modelo peor que predecir la media.</p>
<div class="tabla"><table>
<tr><th>Modelo</th><th>R² test</th><th>R² validación cruzada</th><th>MSE test (kg²)</th><th>MAE test (kg)</th></tr>
{filas_pred}
</table></div>
<div class="conclusion">{c['predecir']}</div>

<h2>3. Predecir y luego optimizar (enfoque híbrido)</h2>
<p>El rendimiento de cada tipo de barco depende del clima y no se conoce de antemano. Se entrena un modelo por
tipo de barco con un histórico de {h['n_historico']} salidas (captura por barco según el clima). Después, en
{h['n_escenarios']} escenarios nuevos (clima y recursos aleatorios), la PL decide con los rendimientos
estimados. La decisión se evalúa con los rendimientos <em>verdaderos</em>: captura obtenida frente a la
captura óptima que se habría logrado conociéndolos (100 %).</p>
<div class="tabla"><table>
<tr><th>Origen de los coeficientes de la PL</th><th>% del óptimo (media)</th><th>Mediana</th><th>Peor caso</th>
<th>Escenarios con decisión óptima</th><th>kg perdidos (media)</th><th>R² del rendimiento (peq./med./gr.)</th></tr>
{filas_h}
</table></div>
<p class="nota">Captura óptima media con rendimientos verdaderos: {fmt(h['optimo_medio_kg'], 1)} kg por escenario.
«Decisión óptima» = al menos el 99,9 % del óptimo.</p>
<div class="conclusion">{c['hibrido']}</div>

<h2>Recomendaciones</h2>
<ul>
<li><strong>Optimizar y predecir responden preguntas distintas.</strong> La PL dice qué hacer con unos
coeficientes dados y qué recurso conviene ampliar (precios sombra); el ML estima cantidades inciertas. No se
elige uno «en lugar» del otro.</li>
<li><strong>Para decidir</strong> con un modelo lineal bien planteado, la PL da el óptimo y además explica la
decisión: qué recursos limitan, cuánto vale cada unidad adicional y en qué rango.</li>
<li><strong>Para predecir</strong>, compare siempre con la media y con una regresión lineal antes de usar un
modelo más complejo, y desconfíe de un R² alto en datos sintéticos: puede ser solo la fórmula de generación.</li>
<li><strong>El ML aporta a la decisión</strong> cuando hay que estimar parámetros inciertos que luego alimentan
la optimización. {recomendacion_hibrido}</li>
</ul>

<h2>Limitaciones</h2>
<ul>
<li>Datos sintéticos con una estructura elegida para el curso; con otra estructura las cifras cambiarían.</li>
<li>La PL usa variables continuas y un único periodo; no modela riesgo ni incertidumbre explícita
(programación estocástica u optimización robusta serían los siguientes pasos).</li>
<li>Los hiperparámetros de los modelos no se han ajustado; el objetivo es docente, no competir en precisión.</li>
</ul>
<p class="nota" style="text-align:center;margin-top:32px">Gráficos interactivos:
<code>analisis_comparativo_io_completo.html</code> · Cifras en bruto: <code>resultados_analisis_io.json</code></p>
</div>
</body>
</html>
"""
        ruta = DIRECTORIO / archivo
        ruta.write_text(html, encoding="utf-8")
        self._log(f"✅ Reporte ejecutivo guardado: {archivo}")
        return str(ruta)

    def guardar_json(self, archivo="resultados_analisis_io.json"):
        datos = {k: v for k, v in self.resultados.items()}
        datos["semilla"] = self.semilla
        ruta = DIRECTORIO / archivo
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log(f"✅ Resultados guardados: {archivo}")
        return str(ruta)

    def ejecutar_analisis_completo(self, generar_archivos=True):
        print("🚀 ANÁLISIS: DECIDIR (PL) · PREDECIR (ML) · PREDECIR Y OPTIMIZAR")
        print("⚠️  Datos sintéticos generados con una fórmula conocida (nucleo_io.py)")
        print("=" * 60)
        self.analizar_decision()
        self.evaluar_prediccion()
        self.evaluar_hibrido()
        self.conclusiones()
        if generar_archivos:
            self.crear_visualizaciones()
            self.generar_reporte_ejecutivo()
            self.guardar_json()
        return self.resultados


def main():
    analizador = AnalizadorComparativoIO(semilla=42)
    resultados = analizador.ejecutar_analisis_completo()
    print("\n🎓 Conclusiones:")
    for clave in ("decidir", "predecir", "hibrido"):
        texto = resultados["conclusiones"][clave]
        print(f"\n[{clave}] " + texto.replace("<strong>", "").replace("</strong>", ""))
    return resultados


if __name__ == "__main__":
    main()
