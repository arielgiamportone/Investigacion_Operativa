#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación Interactiva de Investigación Operativa
Streamlit App: decidir (Simplex), predecir (ML) y predecir-y-optimizar

Curso: CONIPE 2025 - Alfabetización en Ciencia de Datos
Autor: Sistema de IA Educativa
Fecha: 2025

Esta aplicación permite:
- Resolver la PL de asignación de barcos y ver precios sombra (decidir)
- Entrenar modelos de ML para estimar capturas, frente a líneas base (predecir)
- Ver cómo cambia la decisión si los rendimientos se estiman con ML
- Explorar los datos SINTÉTICOS (generados con nucleo_io.py; no son reales)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import warnings

import nucleo_io as nio

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="IO Interactive: Simplex vs ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .method-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptimizadorInteractivo:
    """Estado de la app. El mundo sintético, la PL y las métricas vienen de nucleo_io."""

    def __init__(self):
        self.datos_generados = None
        self.modelos_entrenados = {}
        self.modelos_rendimiento = {}
        self.resultados = {}

    def generar_datos_sinteticos(self, n_escenarios, semilla):
        """Datos SINTÉTICOS: captura = Σ barcos_i · rendimiento_i(clima) · (1 + ruido 10 %)."""
        datos = nio.generar_datos_flota(n_escenarios, semilla)
        total = datos[['barcos_pequenos', 'barcos_medianos', 'barcos_grandes']].sum(axis=1)
        datos['capturas_por_barco'] = np.where(total > 0, datos['capturas_totales'] / total.where(total > 0, 1), 0)
        self.datos_generados = datos.drop(columns=['capturas_sin_ruido'])
        self.semilla = semilla
        return self.datos_generados

    def optimizar_simplex(self, combustible, tripulacion, presupuesto, coeficientes=None):
        """PL: max Σ rendimiento_i · x_i  s.a. combustible, tripulación, presupuesto y barcos disponibles."""
        inicio = time.time()
        coef = nio.RENDIMIENTO_NOMINAL if coeficientes is None else coeficientes
        sol = nio.resolver_lp(coef, [combustible, tripulacion, presupuesto])
        if not sol['exito']:
            return {'exito': False, 'mensaje': f"No hay solución factible ({sol['mensaje']})"}
        x1, x2, x3 = sol['x']
        return {
            'exito': True,
            'barcos_pequenos': x1,
            'barcos_medianos': x2,
            'barcos_grandes': x3,
            'x': sol['x'],
            'capturas_esperadas': sol['objetivo'],
            'holgura': sol['holgura'],
            'uso': sol['uso'],
            'precios_sombra': sol['precios_sombra'],
            'valor_barco_extra': sol['valor_barco_extra'],
            'tiempo_ejecucion': time.time() - inicio,
        }

    def entrenar_modelos_ml(self):
        """Entrena modelos de predicción de capturas (con líneas base) y, aparte,
        modelos del rendimiento por tipo de barco para el enfoque híbrido."""
        if self.datos_generados is None:
            return False

        X = self.datos_generados[nio.VARIABLES_PREDICCION]
        y = self.datos_generados['capturas_totales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelos = {
            'Media (línea base)': DummyRegressor(),
            'Regresión lineal': make_pipeline(StandardScaler(), LinearRegression()),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale')),
            'Neural Network': make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)),
        }

        resultados_ml = {}
        for nombre, modelo in modelos.items():
            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo_entrenamiento = time.time() - inicio
            y_pred = modelo.predict(X_test)
            m = nio.metricas_regresion(y_test, y_pred)   # R² sin recortar
            resultados_ml[nombre] = {
                'modelo': modelo,
                'mse': m['mse'],
                'r2': m['r2'],
                'mae': m['mae'],
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'predicciones_test': y_pred,
                'valores_reales': y_test.values,
            }
        self.modelos_entrenados = resultados_ml

        # Rendimiento por tipo de barco según el clima (para "predecir y optimizar")
        hist = nio.generar_historico_salidas(1000, getattr(self, 'semilla', 42) + 1)
        self.modelos_rendimiento = {}
        for nombre, crear in {
            'Regresión lineal': lambda: make_pipeline(StandardScaler(), LinearRegression()),
            'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
        }.items():
            self.modelos_rendimiento[nombre] = [
                crear().fit(hist.loc[hist['tipo'] == t, nio.VARIABLES_CLIMA],
                            hist.loc[hist['tipo'] == t, 'captura_por_barco'])
                for t in nio.TIPOS_BARCO]
        return True

    def estimar_rendimientos(self, nombre, clima):
        """Rendimiento estimado (kg por barco) de cada tipo para unas condiciones."""
        entrada = pd.DataFrame([clima])[nio.VARIABLES_CLIMA]
        return np.array([m.predict(entrada)[0] for m in self.modelos_rendimiento[nombre]])

    def predecir_ml(self, modelo_nombre, parametros):
        """Predicción de capturas con un modelo entrenado."""
        if modelo_nombre not in self.modelos_entrenados:
            return None
        modelo = self.modelos_entrenados[modelo_nombre]['modelo']
        entrada = pd.DataFrame([parametros])[nio.VARIABLES_PREDICCION]
        inicio = time.time()
        prediccion = modelo.predict(entrada)[0]
        return {
            'capturas_predichas': prediccion,
            'tiempo_prediccion': time.time() - inicio,
        }

# Inicializar optimizador
if 'optimizador' not in st.session_state:
    st.session_state.optimizador = OptimizadorInteractivo()

# Título principal
st.markdown('<h1 class="main-header">🚀 IO Interactive: Simplex vs Machine Learning</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Comparación Interactiva de Métodos de Investigación Operativa</p>', unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Generar datos sintéticos
st.sidebar.subheader("📊 Datos Sintéticos")
st.sidebar.caption("⚠️ Los datos no son reales: se generan con una fórmula conocida "
                   "(rendimiento por tipo de barco × factores de clima) más un 10 % de ruido. "
                   "Ver nucleo_io.py.")
n_escenarios = st.sidebar.slider("Número de escenarios", 100, 2000, 1000, 100)
semilla = st.sidebar.number_input("Semilla aleatoria", 1, 1000, 42)

if st.sidebar.button("🔄 Generar Datos"):
    with st.spinner("Generando datos sintéticos..."):
        st.session_state.optimizador.generar_datos_sinteticos(n_escenarios, semilla)
    st.sidebar.success(f"✅ {n_escenarios} escenarios generados")

# Entrenar modelos ML
st.sidebar.subheader("🧠 Machine Learning")
if st.sidebar.button("🎯 Entrenar Modelos ML"):
    if st.session_state.optimizador.datos_generados is not None:
        with st.spinner("Entrenando modelos de Machine Learning..."):
            exito = st.session_state.optimizador.entrenar_modelos_ml()
        if exito:
            st.sidebar.success("✅ Modelos entrenados")
        else:
            st.sidebar.error("❌ Error al entrenar modelos")
    else:
        st.sidebar.warning("⚠️ Primero genera los datos")

# Parámetros del problema
st.sidebar.subheader("🎛️ Parámetros del Problema")

# Recursos disponibles
st.sidebar.write("**Recursos Disponibles:**")
combustible = st.sidebar.slider("Combustible (L)", 1000, 5000, 3000, 100)
tripulacion = st.sidebar.slider("Tripulación", 50, 300, 150, 10)
presupuesto = st.sidebar.slider("Presupuesto ($)", 50000, 200000, 125000, 5000)

# Condiciones ambientales
st.sidebar.write("**Condiciones Ambientales:**")
temperatura = st.sidebar.slider("Temperatura del agua (°C)", 15.0, 25.0, 20.0, 0.5)
salinidad = st.sidebar.slider("Salinidad (ppt)", 30.0, 40.0, 35.0, 0.5)
viento = st.sidebar.slider("Velocidad del viento (km/h)", 0.0, 30.0, 15.0, 1.0)
olas = st.sidebar.slider("Altura de olas (m)", 0.5, 4.0, 2.0, 0.1)

# Factores de mercado
st.sidebar.write("**Factores de Mercado:**")
precio = st.sidebar.slider("Precio del pescado ($/kg)", 2.0, 8.0, 5.0, 0.1)
demanda = st.sidebar.slider("Demanda del mercado", 0.5, 1.5, 1.0, 0.1)
competencia = st.sidebar.slider("Competencia en la zona", 0.1, 1.0, 0.5, 0.1)

# Factores temporales
st.sidebar.write("**Factores Temporales:**")
mes = st.sidebar.selectbox("Mes", list(range(1, 13)), 6)
fase_lunar = st.sidebar.slider("Fase lunar", 0.0, 1.0, 0.5, 0.1)

# Variables de decisión para entrada manual
st.sidebar.write("**Variables de Decisión (Manual):**")
barcos_p = st.sidebar.slider("Barcos pequeños", 0, 20, 5, 1)
barcos_m = st.sidebar.slider("Barcos medianos", 0, 15, 3, 1)
barcos_g = st.sidebar.slider("Barcos grandes", 0, 10, 2, 1)

# Contenido principal
tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Simplex", "🧠 Machine Learning", "🔀 Decidir vs. predecir", "📈 Análisis"])

# Tab 1: Simplex
with tab1:
    st.header("🏛️ Optimización con Método Simplex")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Configuración del Problema")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recursos Disponibles</h4>
            <ul>
                <li><strong>Combustible:</strong> {combustible:,} L</li>
                <li><strong>Tripulación:</strong> {tripulacion} personas</li>
                <li><strong>Presupuesto:</strong> ${presupuesto:,}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-card">
            <h4>🎯 Función Objetivo</h4>
            <p>Maximizar: 50×P + 120×M + 200×G (kg esperados en un día de calma)</p>
            <p>Donde P, M, G son barcos pequeños, medianos y grandes (máx. 20, 15 y 10)</p>
            <p>La PL <strong>decide</strong>; no predice nada, así que no se mide con R².</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Optimizar con Simplex", key="simplex_opt"):
            resultado = st.session_state.optimizador.optimizar_simplex(combustible, tripulacion, presupuesto)
            
            if resultado['exito']:
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ Solución Óptima Encontrada</h4>
                    <ul>
                        <li><strong>Barcos Pequeños:</strong> {resultado['barcos_pequenos']:.1f}</li>
                        <li><strong>Barcos Medianos:</strong> {resultado['barcos_medianos']:.1f}</li>
                        <li><strong>Barcos Grandes:</strong> {resultado['barcos_grandes']:.1f}</li>
                        <li><strong>Captura esperada (según los coeficientes):</strong> {resultado['capturas_esperadas']:.1f} kg</li>
                        <li><strong>Ingreso esperado a ${precio:.2f}/kg:</strong> ${resultado['capturas_esperadas'] * precio:,.2f}</li>
                        <li><strong>Tiempo de Ejecución:</strong> {resultado['tiempo_ejecucion']*1000:.2f} ms</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Recursos y precios sombra** (kg adicionales por una unidad más del recurso)")
                st.dataframe(pd.DataFrame({
                    'Recurso': [f"{r} ({u})" for r, u in zip(nio.RECURSOS, nio.UNIDADES_RECURSOS)],
                    'Disponible': [combustible, tripulacion, presupuesto],
                    'Usado': np.round(resultado['uso'], 2),
                    'Holgura': np.round(resultado['holgura'], 2),
                    'Precio sombra (kg/unidad)': np.round(resultado['precios_sombra'], 4),
                }), use_container_width=True, hide_index=True)
                st.caption("Un recurso con holgura tiene precio sombra 0: tener más no aumenta la captura. "
                           "Combustible y presupuesto son proporcionales en este modelo (presupuesto = 50 × combustible "
                           "por barco), así que solo puede limitar el más estricto de los dos.")
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>⚠️ No se encontró solución</h4>
                    <p>{resultado['mensaje']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Visualización del Espacio de Soluciones")
        
        # Crear gráfico 3D del espacio de soluciones
        fig = go.Figure()
        
        # Generar puntos factibles
        puntos_x, puntos_y, puntos_z = [], [], []
        valores_objetivo = []
        
        for p in range(0, 21):
            for m in range(0, 16):
                for g in range(0, 11):
                    # Verificar restricciones
                    if (20*p + 50*m + 100*g <= combustible and
                        2*p + 5*m + 8*g <= tripulacion and
                        1000*p + 2500*m + 5000*g <= presupuesto):
                        puntos_x.append(p)
                        puntos_y.append(m)
                        puntos_z.append(g)
                        valores_objetivo.append(50*p + 120*m + 200*g)
        
        if puntos_x:
            fig.add_trace(go.Scatter3d(
                x=puntos_x,
                y=puntos_y,
                z=puntos_z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=valores_objetivo,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Valor Objetivo")
                ),
                text=[f'Objetivo: {v:.0f}' for v in valores_objetivo],
                name='Puntos Factibles'
            ))
            
            fig.update_layout(
                title="Espacio de Soluciones Factibles",
                scene=dict(
                    xaxis_title="Barcos Pequeños",
                    yaxis_title="Barcos Medianos",
                    zaxis_title="Barcos Grandes"
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay puntos factibles con las restricciones actuales")

# Tab 2: Machine Learning
with tab2:
    st.header("🧠 Predicción con Machine Learning")
    
    if st.session_state.optimizador.modelos_entrenados:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Selección de Modelo")
            
            modelo_seleccionado = st.selectbox(
                "Selecciona un modelo:",
                list(st.session_state.optimizador.modelos_entrenados.keys())
            )
            
            # Mostrar métricas del modelo
            modelo_info = st.session_state.optimizador.modelos_entrenados[modelo_seleccionado]
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Métricas del Modelo: {modelo_seleccionado}</h4>
                <ul>
                    <li><strong>R² Score (sin recortar):</strong> {modelo_info['r2']:.3f}</li>
                    <li><strong>MSE:</strong> {modelo_info['mse']:.2f}</li>
                    <li><strong>MAE:</strong> {modelo_info['mae']:.2f}</li>
                    <li><strong>Tiempo de Entrenamiento:</strong> {modelo_info['tiempo_entrenamiento']:.3f}s</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Parámetros para predicción
            parametros_ml = {
                'barcos_pequenos': barcos_p,
                'barcos_medianos': barcos_m,
                'barcos_grandes': barcos_g,
                'combustible_disponible': combustible,
                'tripulacion_disponible': tripulacion,
                'presupuesto': presupuesto,
                'temperatura_agua': temperatura,
                'salinidad': salinidad,
                'velocidad_viento': viento,
                'altura_olas': olas,
                'precio_pescado': precio,
                'demanda_mercado': demanda,
                'competencia_zona': competencia,
                'mes': mes,
                'fase_lunar': fase_lunar
            }
            
            if st.button("🔮 Predecir con ML", key="ml_pred"):
                resultado_ml = st.session_state.optimizador.predecir_ml(modelo_seleccionado, parametros_ml)
                
                if resultado_ml:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>🎯 Predicción Realizada</h4>
                        <ul>
                            <li><strong>Capturas Predichas:</strong> {resultado_ml['capturas_predichas']:.1f} kg</li>
                            <li><strong>Tiempo de Predicción:</strong> {resultado_ml['tiempo_prediccion']*1000:.2f} ms</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📈 Rendimiento de Modelos")
            
            # Crear gráfico de comparación de modelos
            modelos_nombres = list(st.session_state.optimizador.modelos_entrenados.keys())
            r2_scores = [st.session_state.optimizador.modelos_entrenados[m]['r2'] for m in modelos_nombres]
            mse_scores = [st.session_state.optimizador.modelos_entrenados[m]['mse'] for m in modelos_nombres]
            
            fig_comp = make_subplots(
                rows=1, cols=2,
                subplot_titles=['R² Score', 'MSE'],
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            fig_comp.add_trace(
                go.Bar(x=modelos_nombres, y=r2_scores, name='R²', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig_comp.add_trace(
                go.Bar(x=modelos_nombres, y=mse_scores, name='MSE', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig_comp.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Gráfico de predicciones vs reales
            st.subheader("🎯 Predicciones vs Valores Reales")
            
            modelo_info = st.session_state.optimizador.modelos_entrenados[modelo_seleccionado]
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=modelo_info['valores_reales'],
                y=modelo_info['predicciones_test'],
                mode='markers',
                name='Predicciones',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            # Línea de referencia perfecta
            min_val = min(modelo_info['valores_reales'].min(), modelo_info['predicciones_test'].min())
            max_val = max(modelo_info['valores_reales'].max(), modelo_info['predicciones_test'].max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(color='red', dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f'Predicciones vs Reales - {modelo_seleccionado}',
                xaxis_title='Valores Reales',
                yaxis_title='Predicciones',
                height=400
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
    
    else:
        st.warning("⚠️ Primero entrena los modelos ML en la barra lateral")

# Tab 3: Decidir vs. predecir
with tab3:
    st.header("🔀 Decidir vs. predecir")
    st.markdown("""
    El **Simplex (PL)** y el **Machine Learning** responden preguntas distintas, así que no se
    ponen en la misma tabla de precisión ni de velocidad:

    * **Decidir:** con unos rendimientos y recursos dados, ¿cuántos barcos enviar? → PL.
    * **Predecir:** con unas condiciones dadas, ¿cuánto se capturará? → ML.

    Se combinan cuando el ML **estima los coeficientes inciertos** (el rendimiento de cada tipo de
    barco según el clima) y la PL **decide** con esas estimaciones.
    """)

    if st.session_state.optimizador.modelos_entrenados:
        st.subheader("🧠 Predecir: modelos frente a líneas base")
        df_precision = pd.DataFrame([
            {'Modelo': nombre, 'R² test': info['r2'], 'MSE': info['mse'], 'MAE': info['mae']}
            for nombre, info in st.session_state.optimizador.modelos_entrenados.items()
        ])
        st.dataframe(df_precision, use_container_width=True, hide_index=True)
        ml = df_precision[~df_precision['Modelo'].isin(['Media (línea base)', 'Regresión lineal'])]
        mejor = ml.loc[ml['R² test'].idxmax()]
        r2_lineal = df_precision.set_index('Modelo').loc['Regresión lineal', 'R² test']
        st.markdown(f"""
        <div class="method-card">
            <p>Mejor modelo de ML: <strong>{mejor['Modelo']}</strong> (R² = {mejor['R² test']:.3f}).
            Regresión lineal: R² = {r2_lineal:.3f}. Un R² negativo significaría predecir peor que la media;
            se muestra tal cual.</p>
            <p>Como los datos son sintéticos, un R² alto solo indica que el modelo se acerca a la fórmula
            con la que se generaron.</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🚀 Predecir y luego optimizar, con las condiciones de la barra lateral")
        clima = {'temperatura_agua': temperatura, 'salinidad': salinidad, 'velocidad_viento': viento,
                 'altura_olas': olas, 'mes': mes, 'fase_lunar': fase_lunar}
        # En este mundo sintético conocemos el rendimiento verdadero: sirve de referencia
        rend_real = nio.rendimiento_verdadero(temperatura, viento, mes, fase_lunar)[0]
        estrategias = {'Coeficientes nominales (sin ML)': nio.RENDIMIENTO_NOMINAL}
        for nombre in st.session_state.optimizador.modelos_rendimiento:
            estrategias[f'{nombre} + PL'] = st.session_state.optimizador.estimar_rendimientos(nombre, clima)
        estrategias['Rendimientos verdaderos (referencia)'] = rend_real

        opt = st.session_state.optimizador.optimizar_simplex(combustible, tripulacion, presupuesto, rend_real)
        if opt['exito']:
            filas = []
            for nombre, coef in estrategias.items():
                plan = st.session_state.optimizador.optimizar_simplex(combustible, tripulacion, presupuesto, coef)
                obtenido = float(rend_real @ plan['x'])
                filas.append({
                    'Origen de los coeficientes': nombre,
                    'kg/barco (P / M / G)': ' / '.join(f'{c:.0f}' for c in coef),
                    'Plan (P / M / G)': ' / '.join(f'{v:.1f}' for v in plan['x']),
                    'Captura real obtenida (kg)': round(obtenido, 1),
                    '% del óptimo': round(100 * obtenido / opt['capturas_esperadas'], 2),
                })
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
            st.caption("La decisión se evalúa con los rendimientos verdaderos (conocidos porque los datos son "
                       "sintéticos). Prueba a subir el viento: castiga más a los barcos pequeños y la combinación "
                       "óptima cambia. El resultado agregado en muchos escenarios está en "
                       "reporte_ejecutivo_analisis_io.html.")
        else:
            st.warning(opt['mensaje'])
    else:
        st.warning("⚠️ Primero entrena los modelos ML en la barra lateral")

# Tab 4: Análisis
with tab4:
    st.header("📈 Análisis Exploratorio de Datos")
    
    if st.session_state.optimizador.datos_generados is not None:
        datos = st.session_state.optimizador.datos_generados
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Estadísticas Descriptivas")
            st.dataframe(datos.describe(), use_container_width=True)
            
            st.subheader("🔗 Matriz de Correlación")
            
            # Seleccionar variables numéricas principales
            vars_principales = [
                'capturas_totales', 'capturas_por_barco',
                'temperatura_agua', 'velocidad_viento', 'precio_pescado',
                'combustible_disponible', 'presupuesto'
            ]
            
            corr_matrix = datos[vars_principales].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matriz de Correlación",
                color_continuous_scale='RdBu'
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.subheader("📈 Distribuciones")
            
            variable_analisis = st.selectbox(
                "Selecciona variable para análisis:",
                ['capturas_totales', 'capturas_por_barco']
            )
            
            fig_dist = px.histogram(
                datos,
                x=variable_analisis,
                nbins=30,
                title=f'Distribución de {variable_analisis}',
                marginal='box'
            )
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.subheader("🌊 Análisis por Condiciones Ambientales")
            
            # Crear una muestra para mejor rendimiento
            muestra_datos = datos.sample(min(500, len(datos)), random_state=0)
            
            
            fig_scatter = px.scatter(
                muestra_datos,
                x='temperatura_agua',
                y='capturas_totales',
                color='velocidad_viento',
                title='Capturas vs Temperatura (Color: Viento)',
                hover_data=['salinidad', 'altura_olas']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis de sensibilidad
        st.subheader("🔍 Capturas según las condiciones (exploratorio)")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            # Impacto de la temperatura
            temp_groups = pd.cut(datos['temperatura_agua'], bins=5, labels=['Muy Fría', 'Fría', 'Templada', 'Cálida', 'Muy Cálida'])
            temp_analysis = datos.groupby(temp_groups)['capturas_totales'].mean()
            
            fig_temp = go.Figure(data=[
                go.Bar(x=temp_analysis.index, y=temp_analysis.values, marker_color='orange')
            ])
            fig_temp.update_layout(
                title='Capturas Promedio por Temperatura del Agua',
                xaxis_title='Categoría de Temperatura',
                yaxis_title='Capturas Promedio (kg)',
                height=400
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col4:
            # Impacto del viento
            viento_groups = pd.cut(datos['velocidad_viento'], bins=5, labels=['Calma', 'Brisa Suave', 'Brisa Moderada', 'Brisa Fuerte', 'Viento Fuerte'])
            viento_analysis = datos.groupby(viento_groups)['capturas_por_barco'].mean()
            
            fig_viento = go.Figure(data=[
                go.Bar(x=viento_analysis.index, y=viento_analysis.values, marker_color='lightblue')
            ])
            fig_viento.update_layout(
                title='Captura media por barco según el viento',
                xaxis_title='Categoría de Viento',
                yaxis_title='kg por barco',
                height=400
            )
            st.plotly_chart(fig_viento, use_container_width=True)
        
        # Insights automáticos
        st.subheader("🧠 Insights Automáticos")
        
        # Calcular correlaciones más fuertes
        corr_capturas = datos.drop(columns=['capturas_por_barco']).corr()['capturas_totales'].abs().sort_values(ascending=False)
        factor_mas_importante = corr_capturas.index[1]  # Excluir la correlación consigo mismo
        correlacion_valor = corr_capturas.iloc[1]
        
        st.markdown(f"""
        <div class="success-card">
            <h4>🔍 Factor Más Influyente en las Capturas</h4>
            <p><strong>{factor_mas_importante}</strong> con correlación de {correlacion_valor:.3f}</p>
        </div>
        
        <div class="metric-card">
            <h4>📊 Estadísticas Clave</h4>
            <ul>
                <li><strong>Capturas Promedio:</strong> {datos['capturas_totales'].mean():.1f} kg</li>
                <li><strong>Captura media por barco:</strong> {datos['capturas_por_barco'].mean():.1f} kg</li>
                <li><strong>Mejor Mes:</strong> {datos.groupby('mes')['capturas_totales'].mean().idxmax()}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Primero genera los datos sintéticos para realizar el análisis")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚀 <strong>IO Interactive</strong> - Desarrollado para CONIPE 2025</p>
    <p>Comparación Interactiva de Métodos de Investigación Operativa</p>
    <p><em>Decidir (Simplex) · Predecir (ML) · Predecir y optimizar — con datos sintéticos</em></p>
</div>
""", unsafe_allow_html=True)