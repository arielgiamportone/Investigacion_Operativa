#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplex y Machine Learning en una flota pesquera: dos herramientas, dos preguntas
================================================================================

Este script acompaña la narrativa del curso (el problema de "Don Carlos"):

1. **Decidir con programación lineal (Simplex).** ¿Cuántos barcos de cada tipo
   enviar con el combustible, la tripulación y los barcos disponibles? Se
   muestran la solución óptima, los recursos que limitan, los precios sombra y
   si hay óptimos alternativos.
2. **Predecir con Machine Learning.** ¿Qué beneficio cabe esperar de una
   operación según sus condiciones? Se compara con dos líneas base honestas:
   la media y una regresión lineal. Los R² se muestran tal cual (también si son
   negativos).

Las dos partes NO se comparan entre sí: responden preguntas distintas. La
combinación correcta ("predecir y luego optimizar", con la decisión evaluada
frente al óptimo) está en `analisis_comparativo_io.py`, sección 3.

⚠️ Datos SINTÉTICOS: el histórico se genera con fórmulas lineales más ruido
(ver `generar_datos_historicos`). No proceden de ninguna pesquería real.

Autor: Curso CONIPE 2025 - Alfabetización en Ciencia de Datos Pesquera
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linprog
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore', category=UserWarning)

DIRECTORIO = Path(__file__).resolve().parent

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Modelo lineal de Don Carlos
TIPOS = ['Tipo 1', 'Tipo 2', 'Tipo 3']
BENEFICIO_POR_BARCO = np.array([800.0, 1200.0, 1800.0])   # $ por barco y salida
A_RECURSOS = np.array([
    [200.0, 300.0, 500.0],   # combustible (L por barco)
    [8.0, 12.0, 18.0],       # tripulación (personas por barco)
    [1.0, 1.0, 1.0],         # barcos (unidades)
])
NOMBRES_RECURSOS = ['Combustible (L)', 'Tripulación (personas)', 'Barcos disponibles']


class OptimizadorFlotaPesquera:
    """Resuelve el problema de decisión (PL) y, por separado, el de predicción (ML)."""

    def __init__(self):
        self.datos_historicos = None
        self.resultados_simplex = None
        self.resultados_ml = None

    def generar_datos_historicos(self, n_samples=1000, semilla=42):
        """
        Genera un histórico SINTÉTICO de operaciones pesqueras.

        captura  = 150·barcos + 0,8·combustible + 5·tripulación + 2000·clima + ruido N(0, 500)
        beneficio = captura·precio − 1,2·combustible − 50·tripulación − 2·distancia·barcos

        La captura es lineal; el beneficio no, porque multiplica captura × precio
        y distancia × barcos.
        """
        rng = np.random.default_rng(semilla)

        n_barcos = rng.integers(5, 20, n_samples)
        combustible_disponible = rng.uniform(1000, 5000, n_samples)
        tripulacion_disponible = rng.integers(50, 200, n_samples)
        condiciones_climaticas = rng.uniform(0.3, 1.0, n_samples)  # 0.3=malo, 1.0=excelente
        precio_pescado = rng.uniform(2.5, 8.0, n_samples)  # USD/kg
        distancia_promedio = rng.uniform(50, 300, n_samples)  # km

        captura_total = (
            n_barcos * 150
            + combustible_disponible * 0.8
            + tripulacion_disponible * 5
            + condiciones_climaticas * 2000
            + rng.normal(0, 500, n_samples)
        )
        beneficio_total = (
            captura_total * precio_pescado
            - combustible_disponible * 1.2
            - tripulacion_disponible * 50
            - distancia_promedio * n_barcos * 2
        )

        self.datos_historicos = pd.DataFrame({
            'n_barcos': n_barcos,
            'combustible_disponible': combustible_disponible,
            'tripulacion_disponible': tripulacion_disponible,
            'condiciones_climaticas': condiciones_climaticas,
            'precio_pescado': precio_pescado,
            'distancia_promedio': distancia_promedio,
            'captura_total': np.maximum(captura_total, 0),
            'beneficio_total': beneficio_total
        })

        print(f"✅ Datos sintéticos generados: {len(self.datos_historicos)} registros")
        return self.datos_historicos

    # ------------------------------------------------------------------
    # 1. DECIDIR
    # ------------------------------------------------------------------
    def optimizacion_simplex_clasica(self, combustible_max=3000, tripulacion_max=150,
                                     barcos_max=15):
        """
        Maximizar 800·x1 + 1200·x2 + 1800·x3 ($) con combustible, tripulación y
        número de barcos limitados. Se resuelve con HiGHS (variante moderna del
        Simplex) y se leen los precios sombra de `res.ineqlin.marginals`.
        """
        print("\n🔧 1. DECIDIR - PROGRAMACIÓN LINEAL (SIMPLEX)")
        print("=" * 50)

        b = np.array([combustible_max, tripulacion_max, barcos_max], dtype=float)
        # linprog minimiza: se cambia el signo del objetivo y de los marginales
        res = linprog(-BENEFICIO_POR_BARCO, A_ub=A_RECURSOS, b_ub=b,
                      bounds=[(0, None)] * 3, method='highs')

        if not res.success:
            print(f"❌ No hay solución factible: {res.message}")
            print("   (Esto también es información: los recursos no alcanzan para las restricciones.)")
            return None

        precios_sombra = -res.ineqlin.marginals + 0.0
        costes_reducidos = res.lower.marginals + 0.0   # cuánto empeora forzar 1 barco que no se usa
        # Óptimos alternativos: una variable que vale 0 con coste reducido 0
        alternativos = [TIPOS[j] for j in range(3)
                        if res.x[j] < 1e-9 and abs(costes_reducidos[j]) < 1e-9]

        self.resultados_simplex = {
            'x': res.x + 0.0,
            'beneficio_total': -res.fun,
            'uso': A_RECURSOS @ res.x,
            'holgura': res.ineqlin.residual,
            'precios_sombra': precios_sombra,
            'costes_reducidos': costes_reducidos,
            'optimos_alternativos': alternativos,
        }

        print("✅ Solución óptima:")
        for t, v in zip(TIPOS, res.x):
            print(f"   Barcos {t}: {v:.1f}")
        print(f"   Beneficio máximo: ${-res.fun:,.2f}")
        print("\n   Recurso                    Usado     Holgura   Precio sombra")
        for nombre, u, h, ps in zip(NOMBRES_RECURSOS, self.resultados_simplex['uso'],
                                    res.ineqlin.residual, precios_sombra):
            print(f"   {nombre:<24} {u:>8.1f}  {h:>8.1f}   {ps:>8.2f} $ por unidad")
        if alternativos:
            print(f"\n   ℹ️ Hay óptimos alternativos: usar {', '.join(alternativos)} "
                  "daría el mismo beneficio (coste reducido 0).")
        print("   Precio sombra = $ adicionales por una unidad más del recurso.")
        return self.resultados_simplex

    # ------------------------------------------------------------------
    # 2. PREDECIR
    # ------------------------------------------------------------------
    def entrenar_modelo_ml(self, semilla=42):
        """
        Predice el beneficio de una operación a partir de sus condiciones y
        compara con la media y una regresión lineal. R² sin recortar.
        """
        print("\n🤖 2. PREDECIR - MACHINE LEARNING")
        print("=" * 50)

        if self.datos_historicos is None:
            self.generar_datos_historicos()

        X = self.datos_historicos[[
            'n_barcos', 'combustible_disponible', 'tripulacion_disponible',
            'condiciones_climaticas', 'precio_pescado', 'distancia_promedio'
        ]]
        y = self.datos_historicos['beneficio_total']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=semilla)

        modelos = {
            'Media (línea base)': DummyRegressor(),
            'Regresión lineal': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=semilla),
            'SVM': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale')),
        }

        self.resultados_ml = {}
        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            self.resultados_ml[nombre] = {
                'modelo': modelo,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),   # puede ser negativo: se informa tal cual
                'y_pred': y_pred,
            }
            r = self.resultados_ml[nombre]
            print(f"📊 {nombre:<20} R² = {r['r2']:7.4f}   MSE = {r['mse']:,.0f}")
        self._y_test = y_test.to_numpy()

        lineal = self.resultados_ml['Regresión lineal']['r2']
        mejor = max(self.resultados_ml.items(), key=lambda kv: kv[1]['r2'])
        print(f"\n🏆 Mejor modelo: {mejor[0]} (R² = {mejor[1]['r2']:.4f}); "
              f"regresión lineal: R² = {lineal:.4f}")
        print("   Recuerde: los datos salen de una fórmula conocida; un R² alto indica que el")
        print("   modelo la recupera, no que sirva para una flota real.")
        print("   El ML no decide cuántos barcos enviar: para eso está la parte 1.")
        return self.resultados_ml

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------
    def crear_visualizaciones(self, archivo='comparacion_metodos_io.png'):
        print("\n📊 GENERANDO VISUALIZACIONES")
        print("=" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Decidir (PL) y predecir (ML): dos preguntas distintas\n'
                     '(datos sintéticos)', fontsize=16, fontweight='bold')

        # 1. Distribución del histórico sintético
        axes[0, 0].hist(self.datos_historicos['beneficio_total'], bins=50,
                        alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de beneficios (histórico sintético)')
        axes[0, 0].set_xlabel('Beneficio ($)')
        axes[0, 0].set_ylabel('Frecuencia')

        # 2. Correlación entre variables
        corr = self.datos_historicos[[
            'n_barcos', 'combustible_disponible', 'condiciones_climaticas',
            'precio_pescado', 'beneficio_total']].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    ax=axes[0, 1], square=True)
        axes[0, 1].set_title('Matriz de correlación')

        # 3. Decidir: uso de recursos y precios sombra
        if self.resultados_simplex:
            s = self.resultados_simplex
            disponible = s['uso'] + s['holgura']
            uso_pct = 100 * s['uso'] / disponible
            barras = axes[1, 0].bar(range(3), uso_pct,
                                    color=['#E45756' if h < 1e-9 else '#9DA5B4'
                                           for h in s['holgura']])
            axes[1, 0].set_xticks(range(3))
            axes[1, 0].set_xticklabels([n.replace(' (', '\n(') for n in NOMBRES_RECURSOS])
            for bar, ps in zip(barras, s['precios_sombra']):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                                f'precio sombra\n{ps:.2f} $', ha='center', va='bottom')
            plan = ', '.join(f'{t}: {abs(v):.0f}' for t, v in zip(TIPOS, s['x']))
            axes[1, 0].set_title(f'Decidir: plan óptimo ({plan}) → ${s["beneficio_total"]:,.0f}')
            axes[1, 0].set_ylabel('% del recurso usado')
            axes[1, 0].set_ylim(0, 125)

        # 4. Predecir: R² frente a las líneas base
        if self.resultados_ml:
            nombres = list(self.resultados_ml)
            r2 = [self.resultados_ml[n]['r2'] for n in nombres]
            barras = axes[1, 1].bar(range(len(nombres)), r2,
                                    color=['#9DA5B4' if 'Media' in n or 'lineal' in n
                                           else '#4C78A8' for n in nombres])
            axes[1, 1].set_xticks(range(len(nombres)))
            axes[1, 1].set_xticklabels([n.replace(' (', '\n(') for n in nombres])
            axes[1, 1].axhline(0, color='black', linewidth=0.8)
            for bar, v in zip(barras, r2):
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.02,
                                f'{v:.3f}', ha='center', va='bottom')
            axes[1, 1].set_title('Predecir: R² en test (sin recortar)')
            axes[1, 1].set_ylabel('R²')

        plt.tight_layout()
        ruta = DIRECTORIO / archivo
        plt.savefig(ruta, dpi=150, bbox_inches='tight')
        if matplotlib.get_backend().lower() != 'agg':
            plt.show()
        plt.close(fig)
        print(f"✅ Visualizaciones guardadas en '{archivo}'")
        return ruta


def main():
    print("🌊 SIMPLEX Y MACHINE LEARNING EN UNA FLOTA PESQUERA 🌊")
    print("=" * 60)
    print("⚠️  Datos sintéticos: generados con fórmulas conocidas más ruido")
    print("=" * 60)

    optimizador = OptimizadorFlotaPesquera()
    optimizador.generar_datos_historicos(1500)
    optimizador.optimizacion_simplex_clasica()
    optimizador.entrenar_modelo_ml()
    optimizador.crear_visualizaciones()

    print("\n🎉 ANÁLISIS COMPLETADO")
    print("=" * 30)
    print("La PL decide qué hacer con unos coeficientes dados; el ML estima cantidades")
    print("inciertas. No compiten: se combinan cuando el ML estima los coeficientes que")
    print("luego usa la optimización (ver analisis_comparativo_io.py, sección 3).")
    return optimizador


if __name__ == "__main__":
    optimizador = main()
