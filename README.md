# 🚀 Investigación Operativa: del Simplex al Machine Learning

Material docente del curso **CONIPE 2025 – Alfabetización en Ciencia de Datos**. Cuenta la evolución de la
Investigación Operativa, desde el método Simplex (1947) hasta el Machine Learning, con un caso de
asignación de barcos en una flota pesquera.

La idea central del material:

> **Optimizar y predecir responden preguntas distintas.**
> El Simplex (programación lineal, PL) **decide**: con unos coeficientes y recursos dados, ¿cuántos barcos
> de cada tipo enviar? El Machine Learning (ML) **predice**: con unas condiciones dadas, ¿cuánto se va a
> capturar? El ML aporta a la decisión cuando estima parámetros inciertos que luego alimentan la optimización.

## ⚠️ Los datos son sintéticos

Ningún dato procede de una pesquería real. Todos se generan con fórmulas conocidas más ruido aleatorio:

- **`nucleo_io.py`** (análisis comparativo y app Streamlit): el rendimiento de cada tipo de barco
  (50, 120 y 200 kg por salida en calma) se multiplica por factores de temperatura, fase lunar, mes y
  viento. El viento penaliza más a los barcos pequeños que a los grandes. La captura de una flota es
  `Σ barcos_i · rendimiento_i(clima) · (1 + ε)`, con ε normal de desviación 10 %. La salinidad, la altura de
  olas y las variables de mercado no influyen, a propósito.
- **`simplex_ml_optimizacion_pesquera.py`** y **`ml_prediccion_rutas_optimas.py`** generan sus propios datos
  con fórmulas descritas en cada script.

Consecuencia: un R² alto solo demuestra que un modelo recupera la fórmula de generación, no que funcione con
datos reales. Por eso el análisis muestra también el **techo** (el R² de la fórmula verdadera sin ruido).

## 📁 Estructura

```
Investigacion_Operativa/
├── nucleo_io.py                          # Mundo sintético, PL (HiGHS) y métricas comunes
├── analisis_comparativo_io.py            # Análisis principal: decidir, predecir, predecir y optimizar
│   ├── reporte_ejecutivo_analisis_io.html    # → informe (generado)
│   ├── analisis_comparativo_io_completo.html # → gráficos interactivos (generado)
│   └── resultados_analisis_io.json           # → todas las cifras del informe (generado)
├── app_streamlit_io.py                   # Aplicación interactiva
├── simplex_ml_optimizacion_pesquera.py   # El problema de "Don Carlos": PL con precios sombra + ML con líneas base
├── ml_prediccion_rutas_optimas.py        # Modelos de ML para rutas (predicción)
├── animacion_evolucion_simplex_ml.py     # GIF de la evolución histórica
├── storytelling_evolucion_io.md          # Narrativa docente (personajes ficticios)
├── presentacion_storytelling_io.py       # Genera presentacion_interactiva_io.html
├── tests/                                # Tests (pytest): PL, métricas y app Streamlit
├── .github/workflows/tests.yml           # CI: tests y scripts con versiones mínimas y recientes
├── GUIA_STREAMLIT.md · GUIA_INSTALACION.md · PROMPT_VIDEO_STREAMLIT.md
└── requirements.txt
```

## 🛠️ Instalación

Requiere Python 3.10 o superior (lo exige Streamlit 1.51).

```bash
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Cómo ejecutarlo

```bash
# 1. Análisis principal (≈20 s): regenera el informe, los gráficos y el JSON
python analisis_comparativo_io.py

# 2. Presentación HTML (lee las cifras del JSON del paso 1)
python presentacion_storytelling_io.py

# 3. Aplicación interactiva
streamlit run app_streamlit_io.py

# 4. Scripts complementarios
python simplex_ml_optimizacion_pesquera.py   # genera comparacion_metodos_io.png
python ml_prediccion_rutas_optimas.py        # genera visualizaciones_ml_rutas.html
python animacion_evolucion_simplex_ml.py     # genera evolucion_io_simplex_ml.gif

# 5. Tests (≈20 s): PL frente a cálculos de referencia, R² sin recortar y recorrido completo de la app
pytest
```

## 🔎 Qué demuestra cada parte

### 1. Decidir: programación lineal (`analisis_comparativo_io.py`, sección 1)

Maximizar `50·P + 120·M + 200·G` kg sujeto a combustible (≤ 3.000 L), tripulación (≤ 150 personas),
presupuesto (≤ 125.000 $) y barcos disponibles (P ≤ 20, M ≤ 15, G ≤ 10). Se resuelve con HiGHS.
Se evalúa por la solución, los recursos que limitan, los **precios sombra** (`res.ineqlin.marginals`) y la
sensibilidad, **no con R² ni MSE**: la PL no predice nada.

| Resultado | Valor |
|-----------|-------|
| Plan óptimo (pequeños / medianos / grandes) | 20 / 6 / 10 barcos |
| Captura óptima | 3.720 kg |
| Recurso que limita | Tripulación: 24 kg por persona adicional, válido aprox. entre 120 y 195 personas |
| Recursos con holgura (precio sombra 0) | Combustible (sobran 1.300 L) y presupuesto (sobran 40.000 $) |
| Valor de un barco más disponible | Pequeño: 2 kg · Grande: 8 kg · Mediano: 0 |

En este modelo el presupuesto por barco es exactamente 50 veces el combustible por barco, así que ambas
restricciones son proporcionales y solo puede limitar la más estricta.

### 2. Predecir: Machine Learning (sección 2)

Predicción de la captura total de 1.000 escenarios (700 de entrenamiento, 300 de test). R² **sin recortar**:
un valor negativo significa que el modelo predice peor que la media.

| Modelo | R² test | R² validación cruzada | MSE test | MAE test (kg) |
|--------|---------|-----------------------|----------|---------------|
| Media (línea base) | −0,003 | −0,011 ± 0,010 | 906.277 | 734,7 |
| Regresión lineal (línea base) | 0,791 | 0,784 ± 0,034 | 189.058 | 330,8 |
| Random Forest | 0,826 | 0,789 ± 0,033 | 157.257 | 293,0 |
| **Gradient Boosting** | **0,892** | 0,862 ± 0,036 | 97.224 | 217,4 |
| SVM (RBF) | 0,673 | 0,655 ± 0,039 | 295.608 | 353,6 |
| Red neuronal (MLP) | 0,809 | 0,789 ± 0,044 | 173.009 | 320,9 |
| *Techo: fórmula verdadera sin ruido* | *0,968* | — | *28.958* | *119,9* |

La regresión lineal **no empata** con el mejor modelo: la captura es lineal en el número de barcos pero se
multiplica por factores climáticos, y un modelo lineal en las variables originales no representa ese
producto. Ningún modelo supera el techo: el ML solo se acerca a la fórmula con la que se generaron los datos.

### 3. Predecir y luego optimizar (sección 3)

Un modelo por tipo de barco estima el rendimiento según el clima (histórico de 3.000 salidas). En 300
escenarios nuevos, la PL decide con esos rendimientos estimados y la decisión se evalúa con los rendimientos
**verdaderos**, como porcentaje de la captura óptima. Se mide la **calidad de la decisión**, no el R².

| Origen de los coeficientes de la PL | % del óptimo (media) | Mediana | Peor caso | Escenarios con decisión óptima | kg perdidos (media) |
|-------------------------------------|---------------------:|--------:|----------:|-------------------------------:|--------------------:|
| Coeficientes nominales (sin ML) | 91,16 % | 96,74 % | 43,5 % | 38,0 % | 156,2 |
| Regresión lineal + PL | 99,74 % | 100,00 % | 91,7 % | 91,0 % | 4,1 |
| Random Forest + PL | 99,74 % | 100,00 % | 89,1 % | 90,3 % | 5,0 |
| Gradient Boosting + PL | 99,79 % | 100,00 % | 88,2 % | 93,3 % | 3,8 |

La mejora existe porque, en este mundo sintético, el viento afecta de forma distinta a cada tipo de barco y
cambia cuál es la mejor combinación. Si el clima escalara por igual todos los rendimientos, la decisión
óptima no cambiaría y la predicción no aportaría nada a la decisión.

*(Cifras obtenidas con `python analisis_comparativo_io.py`, semilla 42, scikit-learn 1.9 y SciPy 1.18. Con las
versiones mínimas de `requirements.txt` (scikit-learn 1.3, SciPy 1.11) la PL da exactamente lo mismo y solo
cambian en la tercera decimal Random Forest y Gradient Boosting.)*

### Otros scripts

- **`app_streamlit_io.py`**: lo mismo, de forma interactiva. La pestaña *Decidir vs. predecir* muestra, para
  el clima elegido en la barra lateral, el plan con coeficientes nominales, con coeficientes estimados y con
  los verdaderos.
- **`simplex_ml_optimizacion_pesquera.py`**: el problema de «Don Carlos» del relato. La PL da 15 barcos del
  tipo 1 y 12.000 $; limita el combustible (4 $ por litro adicional) y hay óptimos alternativos (10 barcos
  del tipo 2 dan lo mismo). Por separado, predice el beneficio con ML frente a la media y la regresión lineal.
- **`ml_prediccion_rutas_optimas.py`**: predice captura, beneficio y eficiencia de rutas con ML, frente a la
  media y una regresión lineal. La regresión lineal queda cerca en beneficio (R² 0,89 frente a 0,94) y lejos en
  captura y eficiencia. Las «mejores rutas» son candidatas con mayor beneficio *predicho*, no resultados.
- **`storytelling_evolucion_io.md`** y la presentación: narrativa histórica con personajes ficticios; las
  cifras técnicas que citan salen de los scripts.

## 🎓 Recomendaciones que se derivan del análisis

- Optimizar y predecir responden preguntas distintas; no se ponen en la misma tabla de precisión.
- Para decidir con un modelo lineal, la PL da el óptimo y además lo explica (qué limita y cuánto vale ampliarlo).
- Para predecir, compare siempre con la media y una regresión lineal, y desconfíe de un R² alto en datos sintéticos.
- El ML mejora la decisión cuando estima coeficientes inciertos que dependen de condiciones observables.

## ✅ Comprobaciones

`pytest` ejecuta 9 tests: la solución y los precios sombra de la PL base frente a una enumeración de vértices,
el problema dual y diferencias finitas; que ningún R² se recorte; y un recorrido completo de la app Streamlit.
El flujo de GitHub Actions (`.github/workflows/tests.yml`) repite los tests y ejecuta todos los scripts con las
versiones mínimas de `requirements.txt` (Python 3.10) y con las más recientes (Python 3.13).

## ⚠️ Limitaciones

- Datos sintéticos con una estructura elegida para el curso: con otra estructura, las cifras cambiarían.
- La PL usa variables continuas (barcos fraccionarios) y un único periodo; no modela el riesgo.
- Hiperparámetros sin ajustar: el objetivo es docente.

## 📚 Referencias

- Dantzig, G. B. (1963). *Linear Programming and Extensions*.
- Hillier, F. S., & Lieberman, G. J. *Introduction to Operations Research*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Elmachtoub, A. N., & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1).
- [SciPy `linprog` (HiGHS)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) ·
  [scikit-learn](https://scikit-learn.org/stable/)

---

*Proyecto del CONIPE 2025 – Curso de Alfabetización en Ciencia de Datos.*
