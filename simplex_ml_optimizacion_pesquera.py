#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimización Híbrida de Flotas Pesqueras: Simplex + Machine Learning
=====================================================================

Este script demuestra la evolución de la Investigación Operativa clásica
hacia la era de la Inteligencia Artificial, combinando el método simplex
con técnicas avanzadas de Machine Learning para optimización de flotas pesqueras.

Autor: Curso CONIPE 2025 - Alfabetización en Ciencia de Datos Pesquera
Fecha: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class OptimizadorFlotaPesquera:
    """
    Clase principal que implementa optimización híbrida de flotas pesqueras
    combinando programación lineal (Simplex) con Machine Learning.
    """
    
    def __init__(self):
        self.datos_historicos = None
        self.modelo_ml = None
        self.scaler = StandardScaler()
        self.resultados_simplex = None
        self.resultados_ml = None
        self.resultados_hibrido = None
        
    def generar_datos_historicos(self, n_samples=1000):
        """
        Genera datos sintéticos históricos de operaciones pesqueras
        """
        np.random.seed(42)
        
        # Variables de entrada
        n_barcos = np.random.randint(5, 20, n_samples)
        combustible_disponible = np.random.uniform(1000, 5000, n_samples)
        tripulacion_disponible = np.random.randint(50, 200, n_samples)
        condiciones_climaticas = np.random.uniform(0.3, 1.0, n_samples)  # 0.3=malo, 1.0=excelente
        precio_pescado = np.random.uniform(2.5, 8.0, n_samples)  # USD/kg
        distancia_promedio = np.random.uniform(50, 300, n_samples)  # km
        
        # Variables objetivo (simuladas con relaciones realistas)
        captura_total = (
            n_barcos * 150 +  # base por barco
            combustible_disponible * 0.8 +  # eficiencia combustible
            tripulacion_disponible * 5 +  # productividad tripulación
            condiciones_climaticas * 2000 +  # factor climático
            np.random.normal(0, 500, n_samples)  # ruido
        )
        
        beneficio_total = (
            captura_total * precio_pescado -  # ingresos
            combustible_disponible * 1.2 -  # costos combustible
            tripulacion_disponible * 50 -  # costos tripulación
            distancia_promedio * n_barcos * 2  # costos operativos
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
        
        print(f"✅ Datos históricos generados: {len(self.datos_historicos)} registros")
        return self.datos_historicos
    
    def optimizacion_simplex_clasica(self, combustible_max=3000, tripulacion_max=150, 
                                    precio_pescado=5.0, costo_combustible=1.2):
        """
        Implementa optimización clásica usando el método simplex
        """
        print("\n🔧 OPTIMIZACIÓN CLÁSICA - MÉTODO SIMPLEX")
        print("="*50)
        
        # Definición del problema de programación lineal
        # Variables: [barco_tipo_1, barco_tipo_2, barco_tipo_3]
        # Maximizar: beneficio = ingresos - costos
        
        # Coeficientes de la función objetivo (beneficio por tipo de barco)
        c = [-800, -1200, -1800]  # Negativo porque linprog minimiza
        
        # Restricciones de desigualdad (Ax <= b)
        A = [
            [200, 300, 500],    # Restricción de combustible
            [8, 12, 18],        # Restricción de tripulación
            [1, 1, 1]           # Restricción de barcos disponibles
        ]
        
        b = [combustible_max, tripulacion_max, 15]  # Límites de recursos
        
        # Límites de variables (no negatividad)
        x_bounds = [(0, None), (0, None), (0, None)]
        
        # Resolver usando simplex
        resultado = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')
        
        if resultado.success:
            self.resultados_simplex = {
                'barcos_tipo_1': resultado.x[0],
                'barcos_tipo_2': resultado.x[1],
                'barcos_tipo_3': resultado.x[2],
                'beneficio_total': -resultado.fun,
                'combustible_usado': np.dot(A[0], resultado.x),
                'tripulacion_usada': np.dot(A[1], resultado.x)
            }
            
            print(f"✅ Solución óptima encontrada:")
            print(f"   Barcos Tipo 1: {self.resultados_simplex['barcos_tipo_1']:.1f}")
            print(f"   Barcos Tipo 2: {self.resultados_simplex['barcos_tipo_2']:.1f}")
            print(f"   Barcos Tipo 3: {self.resultados_simplex['barcos_tipo_3']:.1f}")
            print(f"   Beneficio Total: ${self.resultados_simplex['beneficio_total']:,.2f}")
            print(f"   Combustible Usado: {self.resultados_simplex['combustible_usado']:.1f}L")
            print(f"   Tripulación Usada: {self.resultados_simplex['tripulacion_usada']:.1f} personas")
        else:
            print("❌ No se encontró solución factible")
            
        return self.resultados_simplex
    
    def entrenar_modelo_ml(self):
        """
        Entrena modelos de Machine Learning para predicción de beneficios
        """
        print("\n🤖 ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING")
        print("="*50)
        
        if self.datos_historicos is None:
            self.generar_datos_historicos()
        
        # Preparar datos
        X = self.datos_historicos[[
            'n_barcos', 'combustible_disponible', 'tripulacion_disponible',
            'condiciones_climaticas', 'precio_pescado', 'distancia_promedio'
        ]]
        y = self.datos_historicos['beneficio_total']
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalado de características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar múltiples modelos
        modelos = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        resultados_modelos = {}
        
        for nombre, modelo in modelos.items():
            if nombre == 'SVM':
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            resultados_modelos[nombre] = {
                'modelo': modelo,
                'mse': mse,
                'r2': r2,
                'y_pred': y_pred
            }
            
            print(f"📊 {nombre}:")
            print(f"   MSE: {mse:,.2f}")
            print(f"   R²: {r2:.4f}")
        
        # Seleccionar mejor modelo
        mejor_modelo = max(resultados_modelos.items(), key=lambda x: x[1]['r2'])
        self.modelo_ml = mejor_modelo[1]['modelo']
        
        print(f"\n🏆 Mejor modelo: {mejor_modelo[0]} (R² = {mejor_modelo[1]['r2']:.4f})")
        
        return resultados_modelos
    
    def optimizacion_hibrida(self, escenarios_candidatos=1000):
        """
        Implementa optimización híbrida: Simplex + ML
        """
        print("\n🚀 OPTIMIZACIÓN HÍBRIDA - SIMPLEX + MACHINE LEARNING")
        print("="*50)
        
        if self.modelo_ml is None:
            self.entrenar_modelo_ml()
        
        # Generar escenarios candidatos
        np.random.seed(42)
        escenarios = []
        
        for _ in range(escenarios_candidatos):
            escenario = {
                'n_barcos': np.random.randint(5, 20),
                'combustible_disponible': np.random.uniform(1000, 5000),
                'tripulacion_disponible': np.random.randint(50, 200),
                'condiciones_climaticas': np.random.uniform(0.5, 1.0),
                'precio_pescado': np.random.uniform(3.0, 7.0),
                'distancia_promedio': np.random.uniform(50, 250)
            }
            escenarios.append(escenario)
        
        # Evaluar escenarios con ML
        X_escenarios = pd.DataFrame(escenarios)
        
        if isinstance(self.modelo_ml, SVR):
            X_escenarios_scaled = self.scaler.transform(X_escenarios)
            beneficios_predichos = self.modelo_ml.predict(X_escenarios_scaled)
        else:
            beneficios_predichos = self.modelo_ml.predict(X_escenarios)
        
        # Seleccionar top escenarios
        top_indices = np.argsort(beneficios_predichos)[-10:]
        
        # Aplicar simplex a los mejores escenarios
        mejores_resultados = []
        
        for idx in top_indices:
            escenario = escenarios[idx]
            beneficio_ml = beneficios_predichos[idx]
            
            # Optimización simplex para este escenario específico
            resultado_simplex = self.optimizacion_simplex_clasica(
                combustible_max=escenario['combustible_disponible'],
                tripulacion_max=escenario['tripulacion_disponible'],
                precio_pescado=escenario['precio_pescado']
            )
            
            if resultado_simplex:
                mejores_resultados.append({
                    'escenario': escenario,
                    'beneficio_ml': beneficio_ml,
                    'beneficio_simplex': resultado_simplex['beneficio_total'],
                    'configuracion_optima': resultado_simplex
                })
        
        # Seleccionar la mejor combinación
        if mejores_resultados:
            mejor_resultado = max(mejores_resultados, 
                                key=lambda x: x['beneficio_simplex'])
            
            self.resultados_hibrido = mejor_resultado
            
            print(f"✅ Mejor configuración híbrida encontrada:")
            print(f"   Beneficio ML Predicho: ${mejor_resultado['beneficio_ml']:,.2f}")
            print(f"   Beneficio Simplex Optimizado: ${mejor_resultado['beneficio_simplex']:,.2f}")
            print(f"   Mejora: {((mejor_resultado['beneficio_simplex']/mejor_resultado['beneficio_ml'])-1)*100:.1f}%")
        
        return self.resultados_hibrido
    
    def crear_visualizaciones(self):
        """
        Crea visualizaciones comparativas de los métodos
        """
        print("\n📊 GENERANDO VISUALIZACIONES COMPARATIVAS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimización de Flotas Pesqueras: Simplex vs ML vs Híbrido', 
                     fontsize=16, fontweight='bold')
        
        # 1. Distribución de datos históricos
        axes[0,0].hist(self.datos_historicos['beneficio_total'], bins=50, 
                      alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribución Histórica de Beneficios')
        axes[0,0].set_xlabel('Beneficio Total ($)')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Correlación entre variables
        corr_matrix = self.datos_historicos[[
            'n_barcos', 'combustible_disponible', 'condiciones_climaticas', 
            'precio_pescado', 'beneficio_total'
        ]].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0,1], square=True)
        axes[0,1].set_title('Matriz de Correlación')
        
        # 3. Comparación de métodos
        if all([self.resultados_simplex, self.resultados_hibrido]):
            metodos = ['Simplex\nClásico', 'ML\nPredicción', 'Híbrido\nSimplex+ML']
            beneficios = [
                self.resultados_simplex['beneficio_total'],
                self.resultados_hibrido['beneficio_ml'],
                self.resultados_hibrido['beneficio_simplex']
            ]
            
            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = axes[1,0].bar(metodos, beneficios, color=colores, alpha=0.8)
            axes[1,0].set_title('Comparación de Beneficios por Método')
            axes[1,0].set_ylabel('Beneficio Total ($)')
            
            # Añadir valores en las barras
            for bar, valor in zip(bars, beneficios):
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                              f'${valor:,.0f}', ha='center', va='bottom')
        
        # 4. Evolución conceptual
        años = [1947, 1980, 2000, 2025]
        tecnologias = ['Simplex\nOriginal', 'Programación\nLineal Avanzada', 
                      'Primeros\nAlgoritmos ML', 'IA Híbrida\nModerna']
        
        axes[1,1].plot(años, [1, 2, 4, 8], 'o-', linewidth=3, markersize=10, 
                      color='#FF6B6B')
        axes[1,1].set_title('Evolución de la Investigación Operativa')
        axes[1,1].set_xlabel('Año')
        axes[1,1].set_ylabel('Capacidad de Optimización (Relativa)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Añadir etiquetas
        for i, (año, tech) in enumerate(zip(años, tecnologias)):
            axes[1,1].annotate(tech, (año, [1, 2, 4, 8][i]), 
                              textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('C:\\Users\\Ariel Giamporte\\Desktop\\Conipe2025\\curso_alfabetizacion_CONIPE25-main\\extras\\Investigacion_Operativa\\comparacion_metodos_io.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en 'comparacion_metodos_io.png'")

def main():
    """
    Función principal que ejecuta todo el pipeline de optimización
    """
    print("🌊 OPTIMIZACIÓN HÍBRIDA DE FLOTAS PESQUERAS 🌊")
    print("=" * 60)
    print("Demostrando la evolución de la Investigación Operativa")
    print("desde el método Simplex clásico hasta la IA moderna")
    print("=" * 60)
    
    # Crear instancia del optimizador
    optimizador = OptimizadorFlotaPesquera()
    
    # 1. Generar datos históricos
    optimizador.generar_datos_historicos(1500)
    
    # 2. Optimización clásica con Simplex
    optimizador.optimizacion_simplex_clasica()
    
    # 3. Entrenamiento de modelos ML
    optimizador.entrenar_modelo_ml()
    
    # 4. Optimización híbrida
    optimizador.optimizacion_hibrida()
    
    # 5. Crear visualizaciones
    optimizador.crear_visualizaciones()
    
    print("\n🎉 ANÁLISIS COMPLETADO")
    print("=" * 30)
    print("El proyecto demuestra cómo la Investigación Operativa moderna")
    print("combina la robustez matemática del Simplex con el poder")
    print("predictivo del Machine Learning para crear soluciones híbridas")
    print("que superan a cada método individual.")
    
    return optimizador

if __name__ == "__main__":
    optimizador = main()