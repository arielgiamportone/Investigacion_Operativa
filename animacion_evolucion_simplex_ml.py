#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animación de la Evolución: Del Método Simplex al Machine Learning
================================================================

Este script crea una animación GIF que muestra la evolución histórica
de la Investigación Operativa desde el método simplex clásico hasta
las técnicas modernas de Machine Learning e IA híbrida.

Autor: Curso CONIPE 2025 - Alfabetización en Ciencia de Datos Pesquera
Fecha: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path

DIRECTORIO = Path(__file__).resolve().parent

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

class AnimacionEvolucionIO:
    """
    Clase para crear animación de la evolución de la Investigación Operativa
    """
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.frames = []
        
    def crear_frame_simplex_clasico(self):
        """
        Crea frame mostrando el método simplex clásico (1947)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        
        # Título y año
        ax.text(5, 9.5, '1947: NACIMIENTO DEL MÉTODO SIMPLEX', 
                fontsize=20, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        ax.text(5, 9, 'George Dantzig - Optimización Lineal', 
                fontsize=14, ha='center', style='italic')
        
        # Dibujar región factible (simplex)
        vertices = np.array([[2, 2], [8, 2], [6, 6], [3, 5]])
        simplex = Polygon(vertices, alpha=0.3, facecolor='lightgreen', 
                         edgecolor='darkgreen', linewidth=3)
        ax.add_patch(simplex)
        
        # Punto óptimo
        ax.plot(6, 6, 'ro', markersize=15, label='Solución Óptima')
        ax.text(6.3, 6.3, 'Punto Óptimo\n(Vértice del Simplex)', 
                fontsize=12, ha='left')
        
        # Líneas de nivel de la función objetivo
        x = np.linspace(1, 9, 100)
        for c in [10, 15, 20, 25]:
            y = (c - 2*x) / 3
            mask = (y >= 1) & (y <= 8)
            ax.plot(x[mask], y[mask], '--', alpha=0.6, color='blue')
        
        ax.text(1, 7, 'Líneas de Nivel\nFunción Objetivo', 
                fontsize=10, color='blue')
        
        # Características del método
        caracteristicas = [
            "• Optimización determinística",
            "• Problemas lineales únicamente",
            "• Solución exacta garantizada",
            "• Restricciones rígidas",
            "• No considera incertidumbre"
        ]
        
        for i, carac in enumerate(caracteristicas):
            ax.text(0.5, 4.5-i*0.4, carac, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
        
        ax.set_title('MÉTODO SIMPLEX CLÁSICO', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        return fig
    
    def crear_frame_programacion_avanzada(self):
        """
        Crea frame mostrando programación lineal avanzada (1980s)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        
        # Título y año
        ax.text(5, 9.5, '1980s: PROGRAMACIÓN LINEAL AVANZADA', 
                fontsize=20, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        ax.text(5, 9, 'Métodos de Punto Interior - Karmarkar', 
                fontsize=14, ha='center', style='italic')
        
        # Múltiples regiones factibles
        vertices1 = np.array([[1.5, 1.5], [4, 1.5], [3.5, 4], [2, 3.5]])
        vertices2 = np.array([[5.5, 2], [8.5, 2], [8, 5], [6, 4.5]])
        
        simplex1 = Polygon(vertices1, alpha=0.4, facecolor='lightgreen', 
                          edgecolor='darkgreen', linewidth=2)
        simplex2 = Polygon(vertices2, alpha=0.4, facecolor='lightblue', 
                          edgecolor='darkblue', linewidth=2)
        
        ax.add_patch(simplex1)
        ax.add_patch(simplex2)
        
        # Trayectoria de punto interior
        t = np.linspace(0, 2*np.pi, 50)
        x_traj = 5 + 1.5*np.cos(t)
        y_traj = 5 + 1.5*np.sin(t)
        ax.plot(x_traj, y_traj, 'r-', linewidth=3, alpha=0.7, 
               label='Trayectoria Punto Interior')
        
        # Puntos de la trayectoria
        for i in range(0, len(x_traj), 5):
            ax.plot(x_traj[i], y_traj[i], 'ro', markersize=8, alpha=0.6)
        
        ax.text(5, 3, 'Método de\nPunto Interior', fontsize=12, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Avances
        avances = [
            "• Problemas de gran escala",
            "• Mejor complejidad computacional",
            "• Programación entera",
            "• Optimización no lineal",
            "• Primeras aplicaciones industriales"
        ]
        
        for i, avance in enumerate(avances):
            ax.text(0.5, 7.5-i*0.4, avance, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcyan'))
        
        ax.set_title('PROGRAMACIÓN LINEAL AVANZADA', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        return fig
    
    def crear_frame_primeros_ml(self):
        """
        Crea frame mostrando los primeros algoritmos de ML (2000s)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Título y año
        ax.text(5, 9.5, '2000s: PRIMEROS ALGORITMOS DE MACHINE LEARNING', 
                fontsize=18, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax.text(5, 9, 'SVM, Random Forest, Redes Neuronales', 
                fontsize=14, ha='center', style='italic')
        
        # Datos de entrenamiento
        np.random.seed(42)
        x1 = np.random.normal(3, 0.8, 50)
        y1 = np.random.normal(3, 0.8, 50)
        x2 = np.random.normal(7, 0.8, 50)
        y2 = np.random.normal(7, 0.8, 50)
        
        ax.scatter(x1, y1, c='red', s=60, alpha=0.7, label='Clase A')
        ax.scatter(x2, y2, c='blue', s=60, alpha=0.7, label='Clase B')
        
        # Frontera de decisión (SVM)
        x_line = np.linspace(1, 9, 100)
        y_line = x_line  # Línea diagonal simple
        ax.plot(x_line, y_line, 'k-', linewidth=3, label='Frontera SVM')
        
        # Vectores de soporte
        support_x = [2.5, 4, 6, 7.5]
        support_y = [2.5, 4, 6, 7.5]
        for sx, sy in zip(support_x, support_y):
            circle = Circle((sx, sy), 0.3, fill=False, edgecolor='black', 
                          linewidth=3)
            ax.add_patch(circle)
        
        ax.text(1, 8, 'Vectores de\nSoporte', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
        
        # Características ML
        caracteristicas_ml = [
            "• Aprendizaje de patrones",
            "• Manejo de no linealidad",
            "• Predicción probabilística",
            "• Validación cruzada",
            "• Generalización a nuevos datos"
        ]
        
        for i, carac in enumerate(caracteristicas_ml):
            ax.text(0.5, 6-i*0.4, carac, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightpink'))
        
        ax.set_title('MACHINE LEARNING CLÁSICO', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        return fig
    
    def crear_frame_ia_hibrida(self):
        """
        Crea frame mostrando IA híbrida moderna (2025)
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Título y año
        ax.text(5, 9.5, '2025: INTELIGENCIA ARTIFICIAL HÍBRIDA', 
                fontsize=18, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='gold'))
        
        ax.text(5, 9, 'Simplex + ML + Deep Learning + Optimización Evolutiva', 
                fontsize=12, ha='center', style='italic')
        
        # Diagrama de arquitectura híbrida
        # Capa 1: Datos
        rect1 = FancyBboxPatch((1, 1), 8, 1, boxstyle="round,pad=0.1", 
                              facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect1)
        ax.text(5, 1.5, 'DATOS HISTÓRICOS + TIEMPO REAL', 
               fontsize=12, ha='center', fontweight='bold')
        
        # Capa 2: ML
        rect2 = FancyBboxPatch((1, 3), 3.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect2)
        ax.text(2.75, 3.75, 'MACHINE\nLEARNING', 
               fontsize=11, ha='center', fontweight='bold')
        
        # Capa 3: Simplex
        rect3 = FancyBboxPatch((5.5, 3), 3.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect3)
        ax.text(7.25, 3.75, 'OPTIMIZACIÓN\nSIMPLEX', 
               fontsize=11, ha='center', fontweight='bold')
        
        # Capa 4: Integración
        rect4 = FancyBboxPatch((2.5, 5.5), 5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor='gold', edgecolor='orange')
        ax.add_patch(rect4)
        ax.text(5, 6.25, 'MOTOR DE DECISIÓN\nHÍBRIDO', 
               fontsize=12, ha='center', fontweight='bold')
        
        # Flechas de conexión
        ax.annotate('', xy=(2.75, 3), xytext=(3.5, 2), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.annotate('', xy=(7.25, 3), xytext=(6.5, 2), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.annotate('', xy=(4, 5.5), xytext=(2.75, 4.5), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax.annotate('', xy=(6, 5.5), xytext=(7.25, 4.5), 
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        # Ventajas del enfoque híbrido
        ventajas = [
            "🚀 Optimización global + local",
            "🧠 Aprendizaje continuo",
            "⚡ Adaptación en tiempo real",
            "🎯 Coeficientes estimados con datos",
            "🔄 Retroalimentación automática"
        ]
        
        for i, ventaja in enumerate(ventajas):
            ax.text(0.5, 8.5-i*0.3, ventaja, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
        
        ax.set_title('INTELIGENCIA ARTIFICIAL HÍBRIDA', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def crear_frame_comparacion_final(self):
        """
        Crea frame final con comparación de todos los métodos
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('EVOLUCIÓN DE LA INVESTIGACIÓN OPERATIVA: 1947-2025', 
                    fontsize=18, fontweight='bold')
        
        # Antes este fotograma mostraba "capacidad", "precisión" y puntuaciones 1-10
        # inventadas. Ahora muestra qué pregunta responde cada método y las cifras
        # reales del análisis del curso (datos sintéticos), si ya se ha ejecutado.
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        resultados = None
        ruta_json = DIRECTORIO / 'resultados_analisis_io.json'
        if ruta_json.exists():
            resultados = json.loads(ruta_json.read_text(encoding='utf-8'))

        # Gráfico 1: dos preguntas distintas
        ax1.axis('off')
        ax1.set_title('Dos preguntas distintas')
        ax1.text(0.02, 0.85, 'DECIDIR (Simplex / PL)', fontsize=13, fontweight='bold', color='#FF6B6B')
        ax1.text(0.02, 0.62, '¿Cuántos barcos enviar con estos recursos?\n'
                 'Se mide por: óptimo, recursos que limitan,\nprecios sombra.', fontsize=11)
        ax1.text(0.02, 0.38, 'PREDECIR (Machine Learning)', fontsize=13, fontweight='bold', color='#45B7D1')
        ax1.text(0.02, 0.15, '¿Cuánto se capturará con este clima?\n'
                 'Se mide por: R², MSE, validación cruzada,\nfrente a la media y la regresión lineal.', fontsize=11)

        if resultados:
            # Gráfico 2: calidad de la decisión
            hib = resultados['hibrido']['tabla']
            nombres = [f['estrategia'].replace(' + PL', '\n+ PL').replace(' (sin ML)', '\n(sin ML)') for f in hib]
            valores = [f['pct_optimo_media'] for f in hib]
            barras = ax2.bar(range(len(hib)), valores, color=colores[:len(hib)], alpha=0.8)
            ax2.set_xticks(range(len(hib)))
            ax2.set_xticklabels(nombres, fontsize=9)
            ax2.set_ylim(min(valores) - 5, 101)
            for b, v in zip(barras, valores):
                ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f'{v:.1f} %', ha='center')
            ax2.set_title('Predecir y optimizar: % de la captura óptima\n(datos sintéticos)')

            # Gráfico 3: R² de predicción frente a líneas base
            pred = resultados['prediccion']['tabla']
            ax3.bar(range(len(pred)), [f['r2_test'] for f in pred],
                    color=['#9DA5B4' if f['modelo'].startswith(('Media', 'Regresión')) else '#45B7D1'
                           for f in pred])
            ax3.axhline(resultados['prediccion']['techo']['r2'], ls='--', color='green',
                        label='techo (fórmula verdadera)')
            ax3.axhline(0, color='black', lw=0.8)
            ax3.set_xticks(range(len(pred)))
            ax3.set_xticklabels([f['modelo'] for f in pred], rotation=30, ha='right', fontsize=9)
            ax3.set_ylabel('R² en test (sin recortar)')
            ax3.set_title('Predecir: R² de capturas (datos sintéticos)')
            ax3.legend()
        else:
            for ax in (ax2, ax3):
                ax.axis('off')
                ax.text(0.5, 0.5, 'Ejecute analisis_comparativo_io.py\npara ver las cifras',
                        ha='center', va='center', fontsize=12)

        # Gráfico 4: Timeline con hitos
        ax4.set_xlim(1940, 2030)
        ax4.set_ylim(-1, 4)
        
        eventos = [
            (1947, 'Dantzig inventa\nel Simplex'),
            (1984, 'Algoritmo de\nKarmarkar'),
            (1995, 'SVM (1995) y\nRandom Forest (2001)'),
            (2006, 'Deep Learning\nRenaissance'),
            (2025, 'Predecir y\noptimizar')
        ]
        
        for i, (año, evento) in enumerate(eventos):
            ax4.plot(año, i, 'o', markersize=15, color=colores[min(i, 3)])
            ax4.text(año, i+0.3, evento, ha='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            if i > 0:
                ax4.plot([eventos[i-1][0], año], [i-1, i], '-', 
                        linewidth=3, alpha=0.6, color='gray')
        
        ax4.set_title('Timeline de Innovaciones')
        ax4.set_xlabel('Año')
        ax4.set_yticks([])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def crear_animacion_completa(self):
        """
        Crea la animación completa combinando todos los frames
        """
        print("🎬 Creando animación de evolución de la Investigación Operativa...")
        
        # Crear todos los frames
        frames_funciones = [
            self.crear_frame_simplex_clasico,
            self.crear_frame_programacion_avanzada,
            self.crear_frame_primeros_ml,
            self.crear_frame_ia_hibrida,
            self.crear_frame_comparacion_final
        ]
        
        # Guardar frames individuales
        frame_paths = []
        for i, func in enumerate(frames_funciones):
            fig = func()
            filename = str(DIRECTORIO / f'frame_{i+1}.png')
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            frame_paths.append(filename)
            plt.close(fig)
            print(f"✅ Frame {i+1}/5 creado")
        
        # Crear GIF animado
        # Todos los fotogramas sobre un lienzo blanco del mismo tamaño (con
        # bbox_inches='tight' cada PNG sale de un tamaño distinto y el GIF se
        # recortaba y mezclaba restos del fotograma anterior).
        originales = [Image.open(path).convert('RGB') for path in frame_paths]
        ancho = max(im.width for im in originales)
        alto = max(im.height for im in originales)
        images = []
        for im in originales:
            lienzo = Image.new('RGB', (ancho, alto), 'white')
            lienzo.paste(im, ((ancho - im.width) // 2, (alto - im.height) // 2))
            # Mantener cada frame por 3 segundos (3 x 1000 ms)
            for _ in range(3):
                images.append(lienzo)
        
        # Guardar GIF
        gif_path = str(DIRECTORIO / 'evolucion_io_simplex_ml.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 segundo por frame
            loop=0,
            disposal=2
        )
        
        # Limpiar archivos temporales
        for path in frame_paths:
            os.remove(path)
        
        print(f"🎉 Animación creada exitosamente: {gif_path}")
        return gif_path

def main():
    """
    Función principal para crear la animación
    """
    print("🎬 CREACIÓN DE ANIMACIÓN: EVOLUCIÓN DE LA INVESTIGACIÓN OPERATIVA")
    print("=" * 70)
    print("Del Método Simplex (1947) a la combinación de predicción y optimización (2025)")
    print("=" * 70)
    
    animador = AnimacionEvolucionIO()
    gif_path = animador.crear_animacion_completa()
    
    print("\n📊 RESUMEN DE LA EVOLUCIÓN:")
    print("=" * 30)
    print("1947: Simplex - Optimización lineal determinística")
    print("1980s: Programación avanzada - Problemas de gran escala")
    print("2000s: Machine Learning - Aprendizaje de patrones")
    print("2025: Predecir y optimizar - el ML estima coeficientes, la PL decide")
    
    print("\n🎯 IMPACTO EN LA INDUSTRIA PESQUERA:")
    print("=" * 35)
    print("• Optimización de rutas de pesca")
    print("• Predicción de zonas de alta captura")
    print("• Gestión eficiente de flotas")
    print("• Reducción de costos operativos")
    print("• Sostenibilidad ambiental")
    
    return gif_path

if __name__ == "__main__":
    gif_path = main()