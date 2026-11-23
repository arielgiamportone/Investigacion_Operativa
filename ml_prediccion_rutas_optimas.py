#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelos de Machine Learning para Predicción de Rutas Óptimas
============================================================

Este script implementa modelos avanzados de Machine Learning para
predicción de rutas pesqueras. El ML estima cantidades inciertas (capturas,
beneficio, eficiencia); no sustituye a la programación lineal, que decide con
esas estimaciones. ⚠️ Los datos son SINTÉTICOS: se generan dentro del script.

Autor: Curso CONIPE 2025 - Alfabetización en Ciencia de Datos Pesquera
Fecha: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

DIRECTORIO = Path(__file__).resolve().parent

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

class PredictorRutasML:
    """
    Clase principal para predicción de rutas óptimas usando Machine Learning
    """
    
    def __init__(self):
        self.datos_rutas = None
        self.modelos = {}
        self.scaler = StandardScaler()
        self.mejor_modelo = None
        self.resultados_evaluacion = {}
        
    def generar_datos_rutas_pesqueras(self, n_rutas=2000):
        """
        Genera dataset sintético de rutas pesqueras con múltiples variables
        """
        np.random.seed(42)
        
        # Variables geográficas y temporales
        latitud = np.random.uniform(-60, -30, n_rutas)  # Latitudes del Atlántico Sur
        longitud = np.random.uniform(-70, -40, n_rutas)  # Longitudes del Atlántico Sur
        profundidad_agua = np.random.uniform(50, 2000, n_rutas)  # metros
        distancia_costa = np.random.uniform(10, 500, n_rutas)  # km
        
        # Variables temporales
        mes = np.random.randint(1, 13, n_rutas)
        hora_salida = np.random.randint(0, 24, n_rutas)
        duracion_viaje = np.random.uniform(4, 48, n_rutas)  # horas
        
        # Variables ambientales
        temperatura_agua = 15 + 10 * np.sin(2 * np.pi * mes / 12) + np.random.normal(0, 2, n_rutas)
        velocidad_viento = np.random.exponential(15, n_rutas)  # km/h
        altura_olas = np.random.gamma(2, 0.5, n_rutas)  # metros
        
        # Variables de flota
        tipo_barco = np.random.choice(['Pequeño', 'Mediano', 'Grande'], n_rutas, 
                                    p=[0.4, 0.4, 0.2])
        capacidad_bodega = np.where(tipo_barco == 'Pequeño', 
                                  np.random.uniform(5, 15, n_rutas),
                                  np.where(tipo_barco == 'Mediano',
                                         np.random.uniform(15, 40, n_rutas),
                                         np.random.uniform(40, 100, n_rutas)))
        
        combustible_disponible = np.random.uniform(500, 5000, n_rutas)  # litros
        experiencia_capitan = np.random.randint(1, 30, n_rutas)  # años
        
        # Variables objetivo (simuladas con relaciones complejas)
        # Captura total (kg)
        captura_base = (
            1000 * (profundidad_agua / 1000) * 
            (1 + 0.1 * np.sin(2 * np.pi * mes / 12)) *  # estacionalidad
            (1 - distancia_costa / 1000) *  # proximidad a costa
            (1 + experiencia_capitan / 50) *  # experiencia
            np.where(tipo_barco == 'Grande', 1.5, 
                   np.where(tipo_barco == 'Mediano', 1.2, 1.0))
        )
        
        # Factores de reducción por condiciones adversas
        factor_clima = np.maximum(0.3, 1 - (velocidad_viento / 50) - (altura_olas / 5))
        factor_temperatura = np.maximum(0.5, 1 - np.abs(temperatura_agua - 18) / 20)
        
        captura_total = np.maximum(0, 
            captura_base * factor_clima * factor_temperatura + 
            np.random.normal(0, 200, n_rutas)
        )
        
        # Costo total de la ruta
        costo_combustible = combustible_disponible * 1.2 * (duracion_viaje / 24)
        costo_tripulacion = duracion_viaje * 50 * np.where(tipo_barco == 'Grande', 8, 
                                                          np.where(tipo_barco == 'Mediano', 5, 3))
        costo_mantenimiento = distancia_costa * 2 + duracion_viaje * 100
        
        costo_total = costo_combustible + costo_tripulacion + costo_mantenimiento
        
        # Beneficio neto
        precio_kg = np.random.uniform(3, 8, n_rutas)  # USD/kg
        beneficio_neto = captura_total * precio_kg - costo_total
        
        # Eficiencia de la ruta (kg capturados por hora)
        eficiencia_ruta = captura_total / duracion_viaje
        
        # Crear DataFrame
        self.datos_rutas = pd.DataFrame({
            'latitud': latitud,
            'longitud': longitud,
            'profundidad_agua': profundidad_agua,
            'distancia_costa': distancia_costa,
            'mes': mes,
            'hora_salida': hora_salida,
            'duracion_viaje': duracion_viaje,
            'temperatura_agua': temperatura_agua,
            'velocidad_viento': velocidad_viento,
            'altura_olas': altura_olas,
            'tipo_barco': tipo_barco,
            'capacidad_bodega': capacidad_bodega,
            'combustible_disponible': combustible_disponible,
            'experiencia_capitan': experiencia_capitan,
            'precio_kg': precio_kg,
            'captura_total': captura_total,
            'costo_total': costo_total,
            'beneficio_neto': beneficio_neto,
            'eficiencia_ruta': eficiencia_ruta
        })
        
        print(f"✅ Dataset de rutas generado: {len(self.datos_rutas)} rutas")
        print(f"📊 Variables: {len(self.datos_rutas.columns)} características")
        
        return self.datos_rutas
    
    def analisis_exploratorio(self):
        """
        Realiza análisis exploratorio de los datos de rutas
        """
        print("\n🔍 ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 40)
        
        # Estadísticas descriptivas
        print("\n📈 Estadísticas Descriptivas:")
        print(self.datos_rutas[['captura_total', 'beneficio_neto', 'eficiencia_ruta']].describe())
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis Exploratorio de Rutas Pesqueras', fontsize=16, fontweight='bold')
        
        # 1. Distribución de capturas por tipo de barco
        sns.boxplot(data=self.datos_rutas, x='tipo_barco', y='captura_total', ax=axes[0,0])
        axes[0,0].set_title('Captura por Tipo de Barco')
        axes[0,0].set_ylabel('Captura Total (kg)')
        
        # 2. Relación profundidad vs captura
        axes[0,1].scatter(self.datos_rutas['profundidad_agua'], 
                         self.datos_rutas['captura_total'], alpha=0.6)
        axes[0,1].set_title('Profundidad vs Captura')
        axes[0,1].set_xlabel('Profundidad (m)')
        axes[0,1].set_ylabel('Captura Total (kg)')
        
        # 3. Estacionalidad de capturas
        captura_por_mes = self.datos_rutas.groupby('mes')['captura_total'].mean()
        axes[0,2].plot(captura_por_mes.index, captura_por_mes.values, 'o-', linewidth=2)
        axes[0,2].set_title('Estacionalidad de Capturas')
        axes[0,2].set_xlabel('Mes')
        axes[0,2].set_ylabel('Captura Promedio (kg)')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Mapa de calor de correlaciones
        variables_numericas = ['profundidad_agua', 'distancia_costa', 'temperatura_agua',
                              'velocidad_viento', 'altura_olas', 'experiencia_capitan',
                              'captura_total', 'beneficio_neto']
        
        corr_matrix = self.datos_rutas[variables_numericas].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Matriz de Correlación')
        
        # 5. Distribución de beneficios
        axes[1,1].hist(self.datos_rutas['beneficio_neto'], bins=50, alpha=0.7, 
                      color='green', edgecolor='black')
        axes[1,1].set_title('Distribución de Beneficios')
        axes[1,1].set_xlabel('Beneficio Neto ($)')
        axes[1,1].set_ylabel('Frecuencia')
        
        # 6. Eficiencia vs experiencia del capitán
        axes[1,2].scatter(self.datos_rutas['experiencia_capitan'], 
                         self.datos_rutas['eficiencia_ruta'], alpha=0.6, color='orange')
        axes[1,2].set_title('Experiencia vs Eficiencia')
        axes[1,2].set_xlabel('Experiencia Capitán (años)')
        axes[1,2].set_ylabel('Eficiencia (kg/hora)')
        
        plt.tight_layout()
        plt.savefig(DIRECTORIO / 'analisis_exploratorio_rutas.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Análisis exploratorio completado")
    
    def entrenar_modelos_avanzados(self):
        """
        Entrena múltiples modelos de ML para predicción de rutas óptimas
        """
        print("\n🤖 ENTRENAMIENTO DE MODELOS AVANZADOS")
        print("=" * 45)
        
        if self.datos_rutas is None:
            self.generar_datos_rutas_pesqueras()
        
        # Preparar datos
        # Codificar variables categóricas
        le = LabelEncoder()
        datos_ml = self.datos_rutas.copy()
        datos_ml['tipo_barco_encoded'] = le.fit_transform(datos_ml['tipo_barco'])
        
        # Características para el modelo
        caracteristicas = [
            'latitud', 'longitud', 'profundidad_agua', 'distancia_costa',
            'mes', 'hora_salida', 'duracion_viaje', 'temperatura_agua',
            'velocidad_viento', 'altura_olas', 'tipo_barco_encoded',
            'capacidad_bodega', 'combustible_disponible', 'experiencia_capitan'
        ]
        
        X = datos_ml[caracteristicas]
        
        # Múltiples variables objetivo
        objetivos = {
            'captura_total': datos_ml['captura_total'],
            'beneficio_neto': datos_ml['beneficio_neto'],
            'eficiencia_ruta': datos_ml['eficiencia_ruta']
        }
        
        # Modelos a entrenar. Las dos primeras son líneas base: cualquier modelo
        # de ML debe compararse con ellas. Los modelos que necesitan escalado lo
        # llevan dentro de un pipeline, para que la validación cruzada no use
        # información de las particiones de validación.
        modelos_config = {
            'Media (línea base)': DummyRegressor(strategy='mean'),
            'Regresión lineal': make_pipeline(StandardScaler(), LinearRegression()),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale')),
            'Neural Network': make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
        }
        lineas_base = ('Media (línea base)', 'Regresión lineal')
        
        # Entrenar para cada objetivo
        for objetivo_nombre, y in objetivos.items():
            print(f"\n🎯 Entrenando modelos para: {objetivo_nombre}")
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            resultados_objetivo = {}
            
            for nombre_modelo, modelo in modelos_config.items():
                print(f"   Entrenando {nombre_modelo}...")
                
                # Validación cruzada (R²) sobre el entrenamiento y evaluación en test
                cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                
                # Métricas (el R² se informa tal cual, aunque sea negativo)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                resultados_objetivo[nombre_modelo] = {
                    'modelo': modelo,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"     R²: {r2:.4f} | CV: {cv_mean:.4f}±{cv_std:.4f}")
            
            self.resultados_evaluacion[objetivo_nombre] = resultados_objetivo
            
            # Mejor modelo de ML (las líneas base no compiten) y comparación con ellas.
            # Se elige por la validación cruzada, no por el test, para no
            # sobreajustar la elección al conjunto de test.
            candidatos = {k: v for k, v in resultados_objetivo.items() if k not in lineas_base}
            mejor = max(candidatos.items(), key=lambda x: x[1]['cv_mean'])
            r2_lineal = resultados_objetivo['Regresión lineal']['r2']
            print(f"   🏆 Mejor modelo de ML (por validación cruzada): {mejor[0]} "
                  f"(R² test = {mejor[1]['r2']:.4f}; regresión lineal = {r2_lineal:.4f})")
            if mejor[1]['r2'] - r2_lineal <= 0.01:
                print("   ℹ️ La regresión lineal empata o gana: el ML no aporta en este objetivo.")
        
        print("\n✅ Entrenamiento de modelos completado")
        return self.resultados_evaluacion
    
    def optimizacion_hiperparametros(self, objetivo='beneficio_neto'):
        """
        Optimiza hiperparámetros del mejor modelo
        """
        print(f"\n⚙️ OPTIMIZACIÓN DE HIPERPARÁMETROS - {objetivo.upper()}")
        print("=" * 50)
        
        # Preparar datos
        le = LabelEncoder()
        datos_ml = self.datos_rutas.copy()
        datos_ml['tipo_barco_encoded'] = le.fit_transform(datos_ml['tipo_barco'])
        
        caracteristicas = [
            'latitud', 'longitud', 'profundidad_agua', 'distancia_costa',
            'mes', 'hora_salida', 'duracion_viaje', 'temperatura_agua',
            'velocidad_viento', 'altura_olas', 'tipo_barco_encoded',
            'capacidad_bodega', 'combustible_disponible', 'experiencia_capitan'
        ]
        
        X = datos_ml[caracteristicas]
        y = datos_ml[objetivo]
        
        # Grid search para Random Forest (generalmente el mejor)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        print("🔍 Ejecutando Grid Search...")
        grid_search.fit(X, y)
        
        self.mejor_modelo = grid_search.best_estimator_
        
        print(f"✅ Mejores parámetros: {grid_search.best_params_}")
        print(f"🎯 Mejor puntuación CV: {grid_search.best_score_:.4f}")
        
        return self.mejor_modelo
    
    def prediccion_rutas_optimas(self, n_predicciones=10):
        """
        Ordena rutas candidatas por el beneficio PREDICHO con el mejor modelo.

        Ojo: es una selección entre 1000 candidatas aleatorias, no una
        optimización, y el beneficio que se muestra es una predicción. Al elegir
        las predicciones más altas se tienden a elegir también los mayores
        errores al alza (la "maldición del ganador").
        """
        print(f"\n🎯 PREDICCIÓN DE {n_predicciones} RUTAS ÓPTIMAS")
        print("=" * 40)
        
        if self.mejor_modelo is None:
            self.optimizacion_hiperparametros()
        
        # Generar escenarios candidatos
        np.random.seed(123)
        escenarios = []
        
        for _ in range(1000):  # Generar muchos candidatos
            escenario = {
                'latitud': np.random.uniform(-60, -30),
                'longitud': np.random.uniform(-70, -40),
                'profundidad_agua': np.random.uniform(100, 1500),
                'distancia_costa': np.random.uniform(20, 300),
                'mes': np.random.randint(1, 13),
                'hora_salida': np.random.randint(4, 8),  # Horas óptimas
                'duracion_viaje': np.random.uniform(12, 36),
                'temperatura_agua': np.random.uniform(12, 22),
                'velocidad_viento': np.random.uniform(5, 25),
                'altura_olas': np.random.uniform(0.5, 3),
                'tipo_barco_encoded': np.random.randint(0, 3),
                'capacidad_bodega': np.random.uniform(20, 80),
                'combustible_disponible': np.random.uniform(2000, 4000),
                'experiencia_capitan': np.random.randint(10, 25)
            }
            escenarios.append(escenario)
        
        # Convertir a DataFrame y predecir
        X_candidatos = pd.DataFrame(escenarios)
        predicciones = self.mejor_modelo.predict(X_candidatos)
        
        # Seleccionar las mejores rutas
        indices_mejores = np.argsort(predicciones)[-n_predicciones:]
        
        rutas_optimas = []
        for i, idx in enumerate(indices_mejores):
            ruta = escenarios[idx].copy()
            ruta['beneficio_predicho'] = predicciones[idx]
            ruta['ranking'] = n_predicciones - i
            rutas_optimas.append(ruta)
        
        # Mostrar resultados
        print("🏆 CANDIDATAS CON MAYOR BENEFICIO PREDICHO (predicciones, no resultados):")
        print("=" * 35)
        
        for ruta in reversed(rutas_optimas[-5:]):  # Top 5
            tipo_barco_str = ['Pequeño', 'Mediano', 'Grande'][int(ruta['tipo_barco_encoded'])]
            print(f"\n#{ruta['ranking']} - Beneficio Predicho: ${ruta['beneficio_predicho']:,.2f}")
            print(f"   📍 Ubicación: ({ruta['latitud']:.2f}, {ruta['longitud']:.2f})")
            print(f"   🌊 Profundidad: {ruta['profundidad_agua']:.0f}m")
            print(f"   🚢 Tipo Barco: {tipo_barco_str}")
            print(f"   ⏱️ Duración: {ruta['duracion_viaje']:.1f}h")
            print(f"   🌡️ Temp. Agua: {ruta['temperatura_agua']:.1f}°C")
        
        return rutas_optimas
    
    def crear_visualizaciones_avanzadas(self):
        """
        Crea visualizaciones avanzadas de los resultados
        """
        print("\n📊 CREANDO VISUALIZACIONES AVANZADAS")
        print("=" * 40)
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Comparación de Modelos por Objetivo',
                'Importancia de Características',
                'Beneficio: predicción vs. real (Random Forest, test)',
                'Rutas históricas sintéticas (color: beneficio)'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Comparación de modelos
        objetivos = list(self.resultados_evaluacion.keys())
        modelos = list(self.resultados_evaluacion[objetivos[0]].keys())
        
        for i, objetivo in enumerate(objetivos):
            r2_scores = [self.resultados_evaluacion[objetivo][modelo]['r2'] 
                        for modelo in modelos]
            
            fig.add_trace(
                go.Bar(name=objetivo, x=modelos, y=r2_scores),
                row=1, col=1
            )
        
        # 2. Importancia de características (Random Forest)
        if self.mejor_modelo is not None:
            caracteristicas = [
                'latitud', 'longitud', 'profundidad_agua', 'distancia_costa',
                'mes', 'hora_salida', 'duracion_viaje', 'temperatura_agua',
                'velocidad_viento', 'altura_olas', 'tipo_barco_encoded',
                'capacidad_bodega', 'combustible_disponible', 'experiencia_capitan'
            ]
            
            importancias = self.mejor_modelo.feature_importances_
            
            fig.add_trace(
                go.Bar(x=caracteristicas, y=importancias, 
                      name='Importancia', showlegend=False),
                row=1, col=2
            )
        
        # 3. Predicciones vs Reales (Random Forest, el modelo que se ajusta en la búsqueda de hiperparámetros)
        mejor_resultado = self.resultados_evaluacion['beneficio_neto']['Random Forest']
        
        fig.add_trace(
            go.Scatter(
                x=mejor_resultado['y_test'], 
                y=mejor_resultado['y_pred'],
                mode='markers',
                name='Predicciones',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Línea de referencia perfecta
        min_val = min(mejor_resultado['y_test'].min(), mejor_resultado['y_pred'].min())
        max_val = max(mejor_resultado['y_test'].max(), mejor_resultado['y_pred'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(dash='dash', color='red'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Mapa de rutas (muestra de datos)
        muestra = self.datos_rutas.sample(200)
        
        fig.add_trace(
            go.Scatter(
                x=muestra['longitud'], 
                y=muestra['latitud'],
                mode='markers',
                marker=dict(
                    size=muestra['captura_total']/100,
                    color=muestra['beneficio_neto'],
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Rutas Históricas',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Análisis Avanzado de Modelos ML para Rutas Pesqueras",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Guardar como HTML interactivo
        fig.write_html(DIRECTORIO / 'visualizaciones_ml_rutas.html')
        
        print("✅ Visualizaciones interactivas guardadas en 'visualizaciones_ml_rutas.html'")
        
        return fig

def main():
    """
    Función principal que ejecuta todo el pipeline de ML
    """
    print("🌊 PREDICCIÓN DE RUTAS ÓPTIMAS CON MACHINE LEARNING 🌊")
    print("=" * 65)
    print("El ML estima capturas y beneficios de rutas (datos sintéticos).")
    print("Predice; no sustituye a la optimización, que decide con esas estimaciones.")
    print("=" * 65)
    
    # Crear instancia del predictor
    predictor = PredictorRutasML()
    
    # 1. Generar datos sintéticos
    predictor.generar_datos_rutas_pesqueras(2500)
    
    # 2. Análisis exploratorio
    predictor.analisis_exploratorio()
    
    # 3. Entrenar modelos avanzados
    predictor.entrenar_modelos_avanzados()
    
    # 4. Optimizar hiperparámetros
    predictor.optimizacion_hiperparametros('beneficio_neto')
    
    # 5. Generar predicciones óptimas
    rutas_optimas = predictor.prediccion_rutas_optimas(15)
    
    # 6. Crear visualizaciones avanzadas
    predictor.crear_visualizaciones_avanzadas()
    
    print("\n🎉 ANÁLISIS DE ML COMPLETADO")
    print("=" * 30)
    print("Los R² anteriores miden lo bien que los modelos recuperan la fórmula")
    print("sintética con la que se generaron los datos; no son evidencia sobre")
    print("rutas reales. Elegir rutas con estas predicciones es una decisión que")
    print("debe evaluarse por su resultado, no por el R².")
    
    print("\n🚀 VENTAJAS DEL ENFOQUE ML:")
    print("=" * 30)
    print("• Manejo de relaciones no-lineales complejas")
    print("• Adaptación a patrones estacionales")
    print("• Incorporación de múltiples variables ambientales")
    print("• Aprendizaje continuo con nuevos datos")
    
    return predictor

if __name__ == "__main__":
    predictor = main()