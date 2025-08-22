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
            "🎯 Precisión mejorada",
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
        
        # Gráfico 1: Capacidad de optimización en el tiempo
        años = [1947, 1980, 2000, 2025]
        capacidad = [1, 3, 6, 10]
        metodos = ['Simplex', 'Prog. Avanzada', 'ML Clásico', 'IA Híbrida']
        
        ax1.plot(años, capacidad, 'o-', linewidth=4, markersize=12, color='#FF6B6B')
        ax1.set_title('Evolución de la Capacidad de Optimización')
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Capacidad Relativa')
        ax1.grid(True, alpha=0.3)
        
        for i, (año, cap, met) in enumerate(zip(años, capacidad, metodos)):
            ax1.annotate(met, (año, cap), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=10)
        
        # Gráfico 2: Complejidad vs Precisión
        complejidad = [2, 5, 7, 9]
        precision = [6, 7, 8, 9.5]
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        scatter = ax2.scatter(complejidad, precision, s=[200, 300, 400, 500], 
                             c=colores, alpha=0.7)
        ax2.set_title('Complejidad vs Precisión')
        ax2.set_xlabel('Complejidad Computacional')
        ax2.set_ylabel('Precisión de Resultados')
        ax2.grid(True, alpha=0.3)
        
        for i, met in enumerate(metodos):
            ax2.annotate(met, (complejidad[i], precision[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Gráfico 3: Tipos de problemas resueltos
        problemas = ['Lineales', 'No Lineales', 'Estocásticos', 'Dinámicos', 'Multi-objetivo']
        simplex_score = [10, 2, 1, 2, 3]
        ml_score = [6, 8, 9, 7, 6]
        hibrido_score = [10, 9, 9, 9, 9]
        
        x = np.arange(len(problemas))
        width = 0.25
        
        ax3.bar(x - width, simplex_score, width, label='Simplex', color='#FF6B6B', alpha=0.8)
        ax3.bar(x, ml_score, width, label='ML', color='#4ECDC4', alpha=0.8)
        ax3.bar(x + width, hibrido_score, width, label='Híbrido', color='#96CEB4', alpha=0.8)
        
        ax3.set_title('Capacidad por Tipo de Problema')
        ax3.set_ylabel('Puntuación (1-10)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(problemas, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Timeline con hitos
        ax4.set_xlim(1940, 2030)
        ax4.set_ylim(-1, 4)
        
        eventos = [
            (1947, 'Dantzig inventa\nel Simplex'),
            (1984, 'Algoritmo de\nKarmarkar'),
            (1995, 'SVM y Random\nForest'),
            (2006, 'Deep Learning\nRenaissance'),
            (2025, 'IA Híbrida\nModerna')
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
            filename = f'C:\\Users\\Ariel Giamporte\\Desktop\\Conipe2025\\curso_alfabetizacion_CONIPE25-main\\extras\\Investigacion_Operativa\\frame_{i+1}.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            frame_paths.append(filename)
            plt.close(fig)
            print(f"✅ Frame {i+1}/5 creado")
        
        # Crear GIF animado
        images = []
        for path in frame_paths:
            img = Image.open(path)
            # Mantener cada frame por 3 segundos (3000ms)
            for _ in range(3):
                images.append(img)
        
        # Guardar GIF
        gif_path = 'C:\\Users\\Ariel Giamporte\\Desktop\\Conipe2025\\curso_alfabetizacion_CONIPE25-main\\extras\\Investigacion_Operativa\\evolucion_io_simplex_ml.gif'
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,  # 1 segundo por frame
            loop=0
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
    print("Del Método Simplex (1947) a la Inteligencia Artificial Híbrida (2025)")
    print("=" * 70)
    
    animador = AnimacionEvolucionIO()
    gif_path = animador.crear_animacion_completa()
    
    print("\n📊 RESUMEN DE LA EVOLUCIÓN:")
    print("=" * 30)
    print("1947: Simplex - Optimización lineal determinística")
    print("1980s: Programación avanzada - Problemas de gran escala")
    print("2000s: Machine Learning - Aprendizaje de patrones")
    print("2025: IA Híbrida - Combinación de todas las técnicas")
    
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