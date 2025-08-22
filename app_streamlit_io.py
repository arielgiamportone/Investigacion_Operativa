#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación Interactiva de Investigación Operativa
Streamlit App: Simplex vs Machine Learning vs Híbrido

Curso: CONIPE 2025 - Alfabetización en Ciencia de Datos
Autor: Sistema de IA Educativa
Fecha: 2025

Esta aplicación permite:
- Comparar métodos de optimización en tiempo real
- Ajustar parámetros interactivamente
- Visualizar resultados dinámicamente
- Explorar diferentes escenarios
- Entender las diferencias entre enfoques
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
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
    def __init__(self):
        self.datos_generados = None
        self.modelos_entrenados = {}
        self.resultados = {}
        
    def generar_datos_sinteticos(self, n_escenarios, semilla):
        """Genera datos sintéticos para la optimización"""
        np.random.seed(semilla)
        
        datos = {
            # Variables de decisión
            'barcos_pequenos': np.random.randint(0, 20, n_escenarios),
            'barcos_medianos': np.random.randint(0, 15, n_escenarios),
            'barcos_grandes': np.random.randint(0, 10, n_escenarios),
            
            # Recursos disponibles
            'combustible_disponible': np.random.uniform(1000, 5000, n_escenarios),
            'tripulacion_disponible': np.random.randint(50, 300, n_escenarios),
            'presupuesto': np.random.uniform(50000, 200000, n_escenarios),
            
            # Condiciones ambientales
            'temperatura_agua': np.random.uniform(15, 25, n_escenarios),
            'salinidad': np.random.uniform(30, 40, n_escenarios),
            'velocidad_viento': np.random.uniform(0, 30, n_escenarios),
            'altura_olas': np.random.uniform(0.5, 4.0, n_escenarios),
            
            # Factores de mercado
            'precio_pescado': np.random.uniform(2, 8, n_escenarios),
            'demanda_mercado': np.random.uniform(0.5, 1.5, n_escenarios),
            'competencia_zona': np.random.uniform(0.1, 1.0, n_escenarios),
            
            # Factores temporales
            'mes': np.random.randint(1, 13, n_escenarios),
            'fase_lunar': np.random.uniform(0, 1, n_escenarios),
        }
        
        # Calcular variables objetivo
        datos['capturas_totales'] = self._calcular_capturas_complejas(datos)
        datos['beneficio_neto'] = self._calcular_beneficio_complejo(datos)
        datos['eficiencia_operativa'] = self._calcular_eficiencia(datos)
        
        self.datos_generados = pd.DataFrame(datos)
        return self.datos_generados
    
    def _calcular_capturas_complejas(self, datos):
        """Modelo complejo para calcular capturas"""
        base_pequenos = datos['barcos_pequenos'] * 50
        base_medianos = datos['barcos_medianos'] * 120
        base_grandes = datos['barcos_grandes'] * 200
        
        # Factores ambientales
        factor_temp = 1 + 0.1 * np.sin((datos['temperatura_agua'] - 20) * np.pi / 10)
        factor_viento = np.maximum(0.3, 1 - datos['velocidad_viento'] / 40)
        factor_lunar = 1 + 0.2 * np.sin(datos['fase_lunar'] * 2 * np.pi)
        
        # Factores estacionales
        factor_estacional = 1 + 0.3 * np.sin((datos['mes'] - 6) * np.pi / 6)
        
        # Interacciones no lineales
        sinergia_flota = 1 + 0.1 * np.log1p(datos['barcos_pequenos'] + datos['barcos_medianos'] + datos['barcos_grandes'])
        
        capturas = (base_pequenos + base_medianos + base_grandes) * \
                  factor_temp * factor_viento * factor_lunar * factor_estacional * sinergia_flota
        
        # Añadir ruido realista
        ruido = np.random.normal(0, 0.1, len(capturas))
        return np.maximum(0, capturas * (1 + ruido))
    
    def _calcular_beneficio_complejo(self, datos):
        """Modelo complejo para calcular beneficio"""
        # Costos operativos
        costo_pequenos = datos['barcos_pequenos'] * 1000
        costo_medianos = datos['barcos_medianos'] * 2500
        costo_grandes = datos['barcos_grandes'] * 5000
        
        # Costos variables
        costo_combustible = (datos['barcos_pequenos'] * 20 + 
                           datos['barcos_medianos'] * 50 + 
                           datos['barcos_grandes'] * 100) * \
                          (1 + datos['velocidad_viento'] / 30)
        
        # Ingresos
        capturas = self._calcular_capturas_complejas(datos)
        ingresos = capturas * datos['precio_pescado'] * datos['demanda_mercado']
        
        # Penalizaciones por competencia
        factor_competencia = 1 - 0.3 * datos['competencia_zona']
        
        beneficio = (ingresos * factor_competencia) - \
                   (costo_pequenos + costo_medianos + costo_grandes + costo_combustible)
        
        return beneficio
    
    def _calcular_eficiencia(self, datos):
        """Calcula eficiencia operativa"""
        total_barcos = datos['barcos_pequenos'] + datos['barcos_medianos'] + datos['barcos_grandes']
        capturas = self._calcular_capturas_complejas(datos)
        
        eficiencia = np.where(total_barcos > 0, capturas / total_barcos, 0)
        return eficiencia / np.maximum(1, datos['velocidad_viento'] / 10)
    
    def optimizar_simplex(self, combustible, tripulacion, presupuesto):
        """Optimización usando Simplex"""
        inicio = time.time()
        
        # Definir problema de optimización lineal
        # Maximizar: 50*x1 + 120*x2 + 200*x3 (capturas esperadas)
        c = [-50, -120, -200]  # Negativo para maximización
        
        # Restricciones
        A_ub = [
            [20, 50, 100],    # Combustible
            [2, 5, 8],        # Tripulación
            [1000, 2500, 5000] # Presupuesto
        ]
        
        b_ub = [combustible, tripulacion, presupuesto]
        
        # Límites de variables
        bounds = [(0, 20), (0, 15), (0, 10)]
        
        try:
            resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if resultado.success:
                x1, x2, x3 = resultado.x
                capturas_predichas = 50*x1 + 120*x2 + 200*x3
                beneficio_predicho = capturas_predichas * 5 - (1000*x1 + 2500*x2 + 5000*x3)
                tiempo_ejecucion = time.time() - inicio
                
                return {
                    'exito': True,
                    'barcos_pequenos': x1,
                    'barcos_medianos': x2,
                    'barcos_grandes': x3,
                    'capturas_predichas': capturas_predichas,
                    'beneficio_predicho': beneficio_predicho,
                    'tiempo_ejecucion': tiempo_ejecucion
                }
            else:
                return {'exito': False, 'mensaje': 'No se encontró solución factible'}
                
        except Exception as e:
            return {'exito': False, 'mensaje': f'Error: {str(e)}'}
    
    def entrenar_modelos_ml(self):
        """Entrena modelos de Machine Learning"""
        if self.datos_generados is None:
            return False
        
        # Preparar datos
        caracteristicas = [
            'barcos_pequenos', 'barcos_medianos', 'barcos_grandes',
            'combustible_disponible', 'tripulacion_disponible', 'presupuesto',
            'temperatura_agua', 'salinidad', 'velocidad_viento', 'altura_olas',
            'precio_pescado', 'demanda_mercado', 'competencia_zona',
            'mes', 'fase_lunar'
        ]
        
        X = self.datos_generados[caracteristicas]
        y = self.datos_generados['capturas_totales']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelos a entrenar
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf', C=100, gamma='scale'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        resultados_ml = {}
        
        for nombre, modelo in modelos.items():
            inicio = time.time()
            
            # Entrenar modelo
            if nombre in ['SVM', 'Neural Network']:
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            
            tiempo_entrenamiento = time.time() - inicio
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            resultados_ml[nombre] = {
                'modelo': modelo,
                'scaler': scaler if nombre in ['SVM', 'Neural Network'] else None,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'tiempo_entrenamiento': tiempo_entrenamiento,
                'predicciones_test': y_pred,
                'valores_reales': y_test.values
            }
        
        self.modelos_entrenados = resultados_ml
        return True
    
    def predecir_ml(self, modelo_nombre, parametros):
        """Realiza predicción con modelo ML"""
        if modelo_nombre not in self.modelos_entrenados:
            return None
        
        modelo_info = self.modelos_entrenados[modelo_nombre]
        modelo = modelo_info['modelo']
        scaler = modelo_info['scaler']
        
        # Preparar datos de entrada
        entrada = np.array([[
            parametros['barcos_pequenos'],
            parametros['barcos_medianos'], 
            parametros['barcos_grandes'],
            parametros['combustible_disponible'],
            parametros['tripulacion_disponible'],
            parametros['presupuesto'],
            parametros['temperatura_agua'],
            parametros['salinidad'],
            parametros['velocidad_viento'],
            parametros['altura_olas'],
            parametros['precio_pescado'],
            parametros['demanda_mercado'],
            parametros['competencia_zona'],
            parametros['mes'],
            parametros['fase_lunar']
        ]])
        
        inicio = time.time()
        
        # Realizar predicción
        if scaler is not None:
            entrada_escalada = scaler.transform(entrada)
            prediccion = modelo.predict(entrada_escalada)[0]
        else:
            prediccion = modelo.predict(entrada)[0]
        
        tiempo_prediccion = time.time() - inicio
        
        return {
            'capturas_predichas': prediccion,
            'tiempo_prediccion': tiempo_prediccion
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
tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Simplex", "🧠 Machine Learning", "📊 Comparación", "📈 Análisis"])

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
            <p>Maximizar: 50×P + 120×M + 200×G</p>
            <p>Donde P, M, G son barcos pequeños, medianos y grandes</p>
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
                        <li><strong>Capturas Predichas:</strong> {resultado['capturas_predichas']:.1f} kg</li>
                        <li><strong>Beneficio Predicho:</strong> ${resultado['beneficio_predicho']:.2f}</li>
                        <li><strong>Tiempo de Ejecución:</strong> {resultado['tiempo_ejecucion']*1000:.2f} ms</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
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
                    <li><strong>R² Score:</strong> {modelo_info['r2']:.3f}</li>
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

# Tab 3: Comparación
with tab3:
    st.header("📊 Comparación de Métodos")
    
    if st.session_state.optimizador.modelos_entrenados:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("⚡ Comparación de Velocidad")
            
            # Ejecutar Simplex
            resultado_simplex = st.session_state.optimizador.optimizar_simplex(combustible, tripulacion, presupuesto)
            
            # Ejecutar ML
            parametros_comp = {
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
            
            tiempos = []
            metodos = []
            
            if resultado_simplex['exito']:
                tiempos.append(resultado_simplex['tiempo_ejecucion'] * 1000)
                metodos.append('Simplex')
            
            for modelo_nombre in st.session_state.optimizador.modelos_entrenados.keys():
                resultado_ml = st.session_state.optimizador.predecir_ml(modelo_nombre, parametros_comp)
                if resultado_ml:
                    tiempos.append(resultado_ml['tiempo_prediccion'] * 1000)
                    metodos.append(modelo_nombre)
            
            if tiempos:
                fig_tiempo = go.Figure(data=[
                    go.Bar(x=metodos, y=tiempos, marker_color='lightgreen')
                ])
                fig_tiempo.update_layout(
                    title='Tiempo de Ejecución (ms)',
                    xaxis_title='Método',
                    yaxis_title='Tiempo (ms)',
                    height=400
                )
                st.plotly_chart(fig_tiempo, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Comparación de Precisión")
            
            # Mostrar métricas de precisión
            precision_data = []
            
            for modelo_nombre, modelo_info in st.session_state.optimizador.modelos_entrenados.items():
                precision_data.append({
                    'Método': modelo_nombre,
                    'R²': modelo_info['r2'],
                    'MSE': modelo_info['mse'],
                    'MAE': modelo_info['mae']
                })
            
            df_precision = pd.DataFrame(precision_data)
            
            # Gráfico de R²
            fig_r2 = go.Figure(data=[
                go.Bar(x=df_precision['Método'], y=df_precision['R²'], marker_color='lightblue')
            ])
            fig_r2.update_layout(
                title='R² Score por Método',
                xaxis_title='Método',
                yaxis_title='R² Score',
                height=400
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Tabla comparativa
        st.subheader("📋 Tabla Comparativa Completa")
        
        # Agregar Simplex a la comparación
        if resultado_simplex['exito']:
            df_precision = pd.concat([
                pd.DataFrame([{
                    'Método': 'Simplex',
                    'R²': 'N/A (Determinístico)',
                    'MSE': 'N/A',
                    'MAE': 'N/A'
                }]),
                df_precision
            ], ignore_index=True)
        
        st.dataframe(df_precision, use_container_width=True)
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        
        mejor_r2 = df_precision.loc[df_precision['R²'] != 'N/A (Determinístico)', 'R²'].astype(float).idxmax()
        mejor_modelo = df_precision.loc[mejor_r2, 'Método']
        
        st.markdown(f"""
        <div class="success-card">
            <h4>🏆 Mejor Modelo por Precisión</h4>
            <p><strong>{mejor_modelo}</strong> con R² = {df_precision.loc[mejor_r2, 'R²']:.3f}</p>
        </div>
        
        <div class="method-card">
            <h4>🎯 Recomendaciones de Uso</h4>
            <ul>
                <li><strong>Simplex:</strong> Problemas lineales con solución óptima garantizada</li>
                <li><strong>Random Forest:</strong> Problemas complejos con datos abundantes</li>
                <li><strong>Gradient Boosting:</strong> Máxima precisión predictiva</li>
                <li><strong>SVM:</strong> Problemas de alta dimensionalidad</li>
                <li><strong>Neural Network:</strong> Patrones muy complejos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Primero entrena los modelos ML para realizar comparaciones")

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
                'capturas_totales', 'beneficio_neto', 'eficiencia_operativa',
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
                ['capturas_totales', 'beneficio_neto', 'eficiencia_operativa']
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
            muestra_datos = datos.sample(500)
            
            # Normalizar beneficio_neto para el tamaño (valores positivos)
            beneficio_normalizado = muestra_datos['beneficio_neto'] - muestra_datos['beneficio_neto'].min() + 1
            
            fig_scatter = px.scatter(
                muestra_datos,
                x='temperatura_agua',
                y='capturas_totales',
                color='velocidad_viento',
                size=beneficio_normalizado,
                title='Capturas vs Temperatura (Color: Viento, Tamaño: Beneficio)',
                hover_data=['salinidad', 'altura_olas']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis de sensibilidad
        st.subheader("🔍 Análisis de Sensibilidad")
        
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
            viento_analysis = datos.groupby(viento_groups)['eficiencia_operativa'].mean()
            
            fig_viento = go.Figure(data=[
                go.Bar(x=viento_analysis.index, y=viento_analysis.values, marker_color='lightblue')
            ])
            fig_viento.update_layout(
                title='Eficiencia Promedio por Velocidad del Viento',
                xaxis_title='Categoría de Viento',
                yaxis_title='Eficiencia Promedio',
                height=400
            )
            st.plotly_chart(fig_viento, use_container_width=True)
        
        # Insights automáticos
        st.subheader("🧠 Insights Automáticos")
        
        # Calcular correlaciones más fuertes
        corr_capturas = datos.corr()['capturas_totales'].abs().sort_values(ascending=False)
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
                <li><strong>Beneficio Promedio:</strong> ${datos['beneficio_neto'].mean():.2f}</li>
                <li><strong>Eficiencia Promedio:</strong> {datos['eficiencia_operativa'].mean():.2f}</li>
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
    <p><em>Simplex vs Machine Learning vs Enfoques Híbridos</em></p>
</div>
""", unsafe_allow_html=True)