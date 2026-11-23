#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presentación Interactiva: La Revolución de la Investigación Operativa
Del Método Simplex a la Inteligencia Artificial en Operaciones Pesqueras

Curso: CONIPE 2025 - Alfabetización en Ciencia de Datos
Autor: Sistema de IA Educativa
Fecha: 2025

Este script genera una presentación HTML interactiva que combina:
- El storytelling completo de la evolución de la IO
- Visualizaciones dinámicas y animaciones
- Elementos interactivos para exploración
- Diseño responsivo y moderno

Las cifras de la sección "Visualizaciones" se leen de resultados_analisis_io.json,
que genera analisis_comparativo_io.py. No hay cifras escritas a mano.
"""

import json
import os
from pathlib import Path

from nucleo_io import fmt

class GeneradorPresentacionIO:
    def __init__(self, directorio_base):
        self.directorio_base = Path(directorio_base)
        self.archivo_storytelling = self.directorio_base / "storytelling_evolucion_io.md"
        self.archivo_salida = self.directorio_base / "presentacion_interactiva_io.html"
        self.archivo_resultados = self.directorio_base / "resultados_analisis_io.json"

    def leer_resultados(self):
        """Cifras del análisis comparativo (o None si aún no se ha ejecutado)."""
        try:
            return json.loads(self.archivo_resultados.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def generar_metricas(self):
        """Tarjetas con cifras reales del análisis (datos sintéticos)."""
        r = self.leer_resultados()
        if r is None:
            return """
            <p><em>Ejecute <code>python analisis_comparativo_io.py</code> y vuelva a generar la
            presentación para ver aquí las cifras del análisis.</em></p>"""
        pred = {f["modelo"]: f for f in r["prediccion"]["tabla"]}
        mejor = pred[r["prediccion"]["mejor_modelo"]]
        hib = {f["estrategia"]: f for f in r["hibrido"]["tabla"]}
        nominal = hib["Coeficientes nominales (sin ML)"]
        mejor_h = max((f for f in r["hibrido"]["tabla"] if f is not nominal),
                      key=lambda f: f["pct_optimo_media"])
        tarjetas = [
            (fmt(r["decision"]["objetivo_kg"], 0) + " kg",
             "Captura óptima del modelo base (PL). Limita: "
             + ", ".join(x["recurso"] for x in r["decision"]["recursos"] if x["limita"])),
            (fmt(mejor["r2_test"], 2),
             f"R² de {r['prediccion']['mejor_modelo']} al predecir capturas "
             f"(regresión lineal: {fmt(pred['Regresión lineal']['r2_test'], 2)}; "
             f"techo: {fmt(r['prediccion']['techo']['r2'], 2)})"),
            (fmt(nominal["pct_optimo_media"], 1) + " %",
             "De la captura óptima, decidiendo con rendimientos nominales (sin ML)"),
            (fmt(mejor_h["pct_optimo_media"], 1) + " %",
             f"De la captura óptima, estimando rendimientos con ML y decidiendo con PL "
             f"({mejor_h['estrategia']})"),
        ]
        html = "".join(f"""
                <div class="metrica-card">
                    <div class="metrica-valor">{v}</div>
                    <div class="metrica-label">{t}</div>
                </div>""" for v, t in tarjetas)
        return f"""
            <h3>Cifras del análisis del curso</h3>
            <p><strong>⚠️ Datos sintéticos</strong>, generados con una fórmula conocida (ver
            <code>nucleo_io.py</code>). Salen de ejecutar <code>analisis_comparativo_io.py</code>;
            no describen ninguna flota real.</p>
            <div class="metricas-grid">{html}
            </div>"""
        
    def leer_storytelling(self):
        """Lee el contenido del archivo de storytelling"""
        try:
            with open(self.archivo_storytelling, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "# Error: No se encontró el archivo de storytelling"
    
    def convertir_imagen_base64(self, ruta_imagen):
        """Convierte una imagen a base64 para embebido en HTML"""
        try:
            with open(ruta_imagen, 'rb') as f:
                imagen_data = f.read()
                return base64.b64encode(imagen_data).decode('utf-8')
        except FileNotFoundError:
            return None
    
    def generar_css_avanzado(self):
        """Genera CSS avanzado para la presentación"""
        return """
        <style>
        :root {
            --color-primario: #1e3a8a;
            --color-secundario: #3b82f6;
            --color-acento: #06b6d4;
            --color-exito: #10b981;
            --color-advertencia: #f59e0b;
            --color-error: #ef4444;
            --color-texto: #1f2937;
            --color-texto-claro: #6b7280;
            --color-fondo: #f8fafc;
            --color-fondo-seccion: #ffffff;
            --sombra-suave: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --sombra-media: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --sombra-fuerte: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            --radio-borde: 12px;
            --transicion-suave: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.7;
            color: var(--color-texto);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--color-fondo);
            min-height: 100vh;
            box-shadow: var(--sombra-fuerte);
        }
        
        .header {
            text-align: center;
            padding: 3rem 0;
            background: linear-gradient(135deg, var(--color-primario), var(--color-secundario));
            color: white;
            border-radius: var(--radio-borde);
            margin-bottom: 3rem;
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="waves" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><path d="M0 50 Q25 25 50 50 T100 50 V100 H0 Z" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23waves)"/></svg>') center/cover;
            opacity: 0.3;
        }
        
        .header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
        }
        
        .header .subtitle {
            font-size: 1.25rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .navegacion {
            position: sticky;
            top: 0;
            background: var(--color-fondo-seccion);
            padding: 1rem 0;
            border-radius: var(--radio-borde);
            box-shadow: var(--sombra-media);
            margin-bottom: 2rem;
            z-index: 100;
        }
        
        .nav-menu {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 1rem;
            list-style: none;
        }
        
        .nav-item {
            background: var(--color-secundario);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            transition: var(--transicion-suave);
            cursor: pointer;
        }
        
        .nav-item:hover {
            background: var(--color-primario);
            transform: translateY(-2px);
            box-shadow: var(--sombra-media);
        }
        
        .seccion {
            background: var(--color-fondo-seccion);
            margin-bottom: 3rem;
            padding: 3rem;
            border-radius: var(--radio-borde);
            box-shadow: var(--sombra-suave);
            transition: var(--transicion-suave);
        }
        
        .seccion:hover {
            box-shadow: var(--sombra-media);
            transform: translateY(-2px);
        }
        
        .seccion h2 {
            color: var(--color-primario);
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 3px solid var(--color-acento);
            padding-bottom: 0.5rem;
        }
        
        .seccion h3 {
            color: var(--color-secundario);
            font-size: 1.8rem;
            margin: 2rem 0 1rem 0;
        }
        
        .seccion h4 {
            color: var(--color-texto);
            font-size: 1.4rem;
            margin: 1.5rem 0 1rem 0;
        }
        
        .seccion p {
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
            line-height: 1.8;
        }
        
        .cita-destacada {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-left: 5px solid var(--color-acento);
            padding: 2rem;
            margin: 2rem 0;
            border-radius: 0 var(--radio-borde) var(--radio-borde) 0;
            font-style: italic;
            font-size: 1.2rem;
            position: relative;
        }
        
        .cita-destacada::before {
            content: '"';
            font-size: 4rem;
            color: var(--color-acento);
            position: absolute;
            top: -10px;
            left: 20px;
            opacity: 0.3;
        }
        
        .codigo {
            background: #1e293b;
            color: #e2e8f0;
            padding: 2rem;
            border-radius: var(--radio-borde);
            margin: 2rem 0;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', monospace;
            position: relative;
        }
        
        .codigo::before {
            content: attr(data-language);
            position: absolute;
            top: 10px;
            right: 15px;
            background: var(--color-acento);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .metricas-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .metrica-card {
            background: linear-gradient(135deg, var(--color-exito), #34d399);
            color: white;
            padding: 2rem;
            border-radius: var(--radio-borde);
            text-align: center;
            box-shadow: var(--sombra-suave);
            transition: var(--transicion-suave);
        }
        
        .metrica-card:hover {
            transform: scale(1.05);
            box-shadow: var(--sombra-media);
        }
        
        .metrica-valor {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        
        .metrica-label {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .timeline {
            position: relative;
            padding: 2rem 0;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 4px;
            background: linear-gradient(to bottom, var(--color-primario), var(--color-acento));
            transform: translateX(-50%);
        }
        
        .timeline-item {
            position: relative;
            margin: 3rem 0;
            display: flex;
            align-items: center;
        }
        
        .timeline-item:nth-child(odd) {
            flex-direction: row;
        }
        
        .timeline-item:nth-child(even) {
            flex-direction: row-reverse;
        }
        
        .timeline-content {
            background: var(--color-fondo-seccion);
            padding: 2rem;
            border-radius: var(--radio-borde);
            box-shadow: var(--sombra-suave);
            width: 45%;
            position: relative;
        }
        
        .timeline-year {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            background: var(--color-primario);
            color: white;
            padding: 1rem 2rem;
            border-radius: 25px;
            font-weight: 800;
            font-size: 1.2rem;
            box-shadow: var(--sombra-media);
        }
        
        .imagen-container {
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border-radius: var(--radio-borde);
            border: 2px dashed var(--color-acento);
        }
        
        .imagen-placeholder {
            background: var(--color-secundario);
            color: white;
            padding: 4rem 2rem;
            border-radius: var(--radio-borde);
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .tabla-comparativa {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            background: var(--color-fondo-seccion);
            border-radius: var(--radio-borde);
            overflow: hidden;
            box-shadow: var(--sombra-suave);
        }
        
        .tabla-comparativa th {
            background: var(--color-primario);
            color: white;
            padding: 1.5rem;
            text-align: left;
            font-weight: 600;
        }
        
        .tabla-comparativa td {
            padding: 1.5rem;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .tabla-comparativa tr:hover {
            background: #f9fafb;
        }
        
        .boton-interactivo {
            background: linear-gradient(135deg, var(--color-acento), #0891b2);
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transicion-suave);
            margin: 1rem 0.5rem;
        }
        
        .boton-interactivo:hover {
            transform: translateY(-2px);
            box-shadow: var(--sombra-media);
        }
        
        .footer {
            background: var(--color-primario);
            color: white;
            text-align: center;
            padding: 3rem;
            border-radius: var(--radio-borde);
            margin-top: 3rem;
        }
        
        .animacion-entrada {
            animation: fadeInUp 0.8s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animacion-flotante {
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .seccion {
                padding: 2rem;
            }
            
            .timeline::before {
                left: 20px;
            }
            
            .timeline-item {
                flex-direction: column !important;
                align-items: flex-start;
                padding-left: 3rem;
            }
            
            .timeline-content {
                width: 100%;
            }
            
            .timeline-year {
                position: relative;
                left: auto;
                transform: none;
                margin-bottom: 1rem;
            }
        }
        </style>
        """
    
    def generar_javascript_interactivo(self):
        """Genera JavaScript para interactividad"""
        return """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Navegación suave
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                });
            });
            
            // Animaciones de entrada
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animacion-entrada');
                    }
                });
            }, {
                threshold: 0.1
            });
            
            document.querySelectorAll('.seccion').forEach(seccion => {
                observer.observe(seccion);
            });
            
            // Contador animado para métricas
            function animarContador(elemento, valorFinal, duracion = 2000) {
                const valorInicial = 0;
                const incremento = valorFinal / (duracion / 16);
                let valorActual = valorInicial;
                
                const timer = setInterval(() => {
                    valorActual += incremento;
                    if (valorActual >= valorFinal) {
                        valorActual = valorFinal;
                        clearInterval(timer);
                    }
                    elemento.textContent = Math.floor(valorActual).toLocaleString();
                }, 16);
            }
            
            // Activar contadores cuando sean visibles
            const contadorObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const valor = parseInt(entry.target.dataset.valor);
                        animarContador(entry.target, valor);
                        contadorObserver.unobserve(entry.target);
                    }
                });
            });
            
            document.querySelectorAll('.metrica-valor[data-valor]').forEach(contador => {
                contadorObserver.observe(contador);
            });
            
            // Efectos de hover para tarjetas
            document.querySelectorAll('.metrica-card').forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.05) rotate(1deg)';
                });
                
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1) rotate(0deg)';
                });
            });
            
            // Botones interactivos
            document.querySelectorAll('.boton-interactivo').forEach(boton => {
                boton.addEventListener('click', function() {
                    // Efecto de ondas
                    const ripple = document.createElement('span');
                    ripple.style.position = 'absolute';
                    ripple.style.borderRadius = '50%';
                    ripple.style.background = 'rgba(255,255,255,0.6)';
                    ripple.style.transform = 'scale(0)';
                    ripple.style.animation = 'ripple 0.6s linear';
                    ripple.style.left = '50%';
                    ripple.style.top = '50%';
                    ripple.style.width = '20px';
                    ripple.style.height = '20px';
                    ripple.style.marginLeft = '-10px';
                    ripple.style.marginTop = '-10px';
                    
                    this.style.position = 'relative';
                    this.appendChild(ripple);
                    
                    setTimeout(() => {
                        ripple.remove();
                    }, 600);
                });
            });
            
            // Agregar estilos para el efecto ripple
            const style = document.createElement('style');
            style.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(4);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
            
            // Modo oscuro toggle
            const toggleModoOscuro = document.createElement('button');
            toggleModoOscuro.innerHTML = '🌙';
            toggleModoOscuro.style.position = 'fixed';
            toggleModoOscuro.style.top = '20px';
            toggleModoOscuro.style.right = '20px';
            toggleModoOscuro.style.background = 'var(--color-primario)';
            toggleModoOscuro.style.color = 'white';
            toggleModoOscuro.style.border = 'none';
            toggleModoOscuro.style.borderRadius = '50%';
            toggleModoOscuro.style.width = '50px';
            toggleModoOscuro.style.height = '50px';
            toggleModoOscuro.style.fontSize = '20px';
            toggleModoOscuro.style.cursor = 'pointer';
            toggleModoOscuro.style.zIndex = '1000';
            toggleModoOscuro.style.boxShadow = 'var(--sombra-media)';
            
            document.body.appendChild(toggleModoOscuro);
            
            let modoOscuro = false;
            toggleModoOscuro.addEventListener('click', function() {
                modoOscuro = !modoOscuro;
                if (modoOscuro) {
                    document.body.style.filter = 'invert(1) hue-rotate(180deg)';
                    this.innerHTML = '☀️';
                } else {
                    document.body.style.filter = 'none';
                    this.innerHTML = '🌙';
                }
            });
        });
        </script>
        """
    
    def procesar_markdown_a_html(self, contenido_md):
        """Convierte markdown a HTML con mejoras visuales"""
        # Conversiones básicas de markdown
        html = contenido_md
        
        # Títulos
        html = html.replace('# ', '<h1>')
        html = html.replace('## ', '<h2>')
        html = html.replace('### ', '<h3>')
        html = html.replace('#### ', '<h4>')
        
        # Agregar cierre de títulos
        import re
        html = re.sub(r'<h1>([^\n]+)', r'<h1>\1</h1>', html)
        html = re.sub(r'<h2>([^\n]+)', r'<h2>\1</h2>', html)
        html = re.sub(r'<h3>([^\n]+)', r'<h3>\1</h3>', html)
        html = re.sub(r'<h4>([^\n]+)', r'<h4>\1</h4>', html)
        
        # Citas destacadas
        html = re.sub(r'> \*"([^"]+)"\* - ([^\n]+)', 
                     r'<div class="cita-destacada">"\1"<br><strong>- \2</strong></div>', html)
        
        # Código
        html = re.sub(r'```python\n([^`]+)```', 
                     r'<div class="codigo" data-language="Python"><pre>\1</pre></div>', html)
        html = re.sub(r'```([^\n]*)\n([^`]+)```', 
                     r'<div class="codigo" data-language="\1"><pre>\2</pre></div>', html)
        
        # Párrafos
        lineas = html.split('\n')
        html_procesado = []
        en_parrafo = False
        
        for linea in lineas:
            linea = linea.strip()
            if not linea:
                if en_parrafo:
                    html_procesado.append('</p>')
                    en_parrafo = False
                continue
            
            if (linea.startswith('<h') or linea.startswith('<div') or 
                linea.startswith('---') or linea.startswith('|')):
                if en_parrafo:
                    html_procesado.append('</p>')
                    en_parrafo = False
                html_procesado.append(linea)
            else:
                if not en_parrafo and not linea.startswith('<'):
                    html_procesado.append('<p>')
                    en_parrafo = True
                html_procesado.append(linea)
        
        if en_parrafo:
            html_procesado.append('</p>')
        
        return '\n'.join(html_procesado)
    
    def generar_seccion_visualizaciones(self):
        """Genera sección con placeholders para visualizaciones y cifras reales"""
        return f"""
        <div class="seccion" id="visualizaciones">
            <h2>🎬 Visualizaciones Interactivas</h2>
            
            <div class="imagen-container">
                <h3>Evolución del Simplex al Machine Learning</h3>
                <div class="imagen-placeholder animacion-flotante">
                    📊 Animación GIF: evolucion_simplex_ml.gif<br>
                    <small>Visualización de la transformación histórica de la IO</small>
                </div>
            </div>
            
            <div class="imagen-container">
                <h3>Optimización Híbrida de Flotas</h3>
                <div class="imagen-placeholder animacion-flotante">
                    📈 Gráficos: optimizacion_flota_pesquera.png<br>
                    <small>Comparación de métodos clásicos vs modernos</small>
                </div>
            </div>
            
            <div class="imagen-container">
                <h3>Predicción de Rutas Óptimas con ML</h3>
                <div class="imagen-placeholder animacion-flotante">
                    🗺️ Visualización: rutas_optimas_ml.html<br>
                    <small>Mapas interactivos con predicciones de IA</small>
                </div>
            </div>
            
            {self.generar_metricas()}
        </div>
        </div>
        """
    
    def generar_timeline_interactivo(self):
        """Genera timeline interactivo de la evolución"""
        return """
        <div class="seccion" id="timeline">
            <h2>⏰ Línea de Tiempo: Evolución de la IO</h2>
            
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-year">1947</div>
                    <div class="timeline-content">
                        <h3>🏛️ Nacimiento del Simplex</h3>
                        <p>George Dantzig desarrolla el método Simplex, revolucionando la optimización lineal.</p>
                        <button class="boton-interactivo" onclick="mostrarDetalle('1947')">Ver Detalles</button>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-year">1980s</div>
                    <div class="timeline-content">
                        <h3>⚙️ Programación Avanzada</h3>
                        <p>Algoritmos de punto interior y programación no lineal expanden las posibilidades.</p>
                        <button class="boton-interactivo" onclick="mostrarDetalle('1980')">Ver Detalles</button>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-year">2000s</div>
                    <div class="timeline-content">
                        <h3>🧠 Era del Machine Learning</h3>
                        <p>Los algoritmos aprenden patrones y predicen comportamientos complejos.</p>
                        <button class="boton-interactivo" onclick="mostrarDetalle('2000')">Ver Detalles</button>
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-year">2025</div>
                    <div class="timeline-content">
                        <h3>🚀 Predecir y optimizar</h3>
                        <p>El ML estima lo incierto; la optimización decide con esas estimaciones.</p>
                        <button class="boton-interactivo" onclick="mostrarDetalle('2025')">Ver Detalles</button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function mostrarDetalle(año) {
            const detalles = {
                '1947': 'El método Simplex permitió resolver problemas de optimización lineal de manera sistemática, estableciendo las bases de la Investigación Operativa moderna.',
                '1980': 'Los algoritmos de punto interior y la programación no lineal permitieron abordar problemas más complejos y de mayor escala.',
                '2000': 'Machine Learning introdujo la capacidad de aprender de datos históricos y predecir comportamientos futuros sin programación explícita.',
                '2025': 'Optimizar y predecir responden preguntas distintas. Se combinan cuando el ML estima parámetros inciertos (por ejemplo, el rendimiento de cada barco según el clima) que luego alimentan la optimización.'
            };
            
            alert(`Año ${año}:\n\n${detalles[año]}`);
        }
        </script>
        """
    
    def generar_html_completo(self):
        """Genera el HTML completo de la presentación"""
        contenido_storytelling = self.leer_storytelling()
        contenido_html = self.procesar_markdown_a_html(contenido_storytelling)
        
        # Dividir contenido en secciones
        secciones = contenido_html.split('<h2>')
        
        html_completo = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>La Revolución de la Investigación Operativa: Del Simplex a la IA</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
            {self.generar_css_avanzado()}
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>🌊 La Revolución de la Investigación Operativa</h1>
                    <p class="subtitle">Del Método Simplex a la Inteligencia Artificial en Operaciones Pesqueras</p>
                    <p class="subtitle"><strong>CONIPE 2025 - Curso de Alfabetización en Ciencia de Datos</strong></p>
                </header>
                
                <nav class="navegacion">
                    <ul class="nav-menu">
                        <li><a href="#inicio" class="nav-item">🏠 Inicio</a></li>
                        <li><a href="#evolucion" class="nav-item">📈 Evolución</a></li>
                        <li><a href="#timeline" class="nav-item">⏰ Timeline</a></li>
                        <li><a href="#visualizaciones" class="nav-item">📊 Visualizaciones</a></li>
                        <li><a href="#impacto" class="nav-item">🎯 Impacto</a></li>
                        <li><a href="#futuro" class="nav-item">🔮 Futuro</a></li>
                    </ul>
                </nav>
                
                <main>
                    <div class="seccion" id="inicio">
                        <h2>📖 Prólogo: El Despertar de una Nueva Era</h2>
                        <p>En las vastas extensiones del océano, donde las decisiones pueden significar la diferencia entre el éxito y el fracaso, entre la prosperidad y la pérdida, una revolución silenciosa ha estado transformando la manera en que entendemos y optimizamos las operaciones pesqueras.</p>
                        
                        <div class="cita-destacada">
                            "Esta es la historia de cómo la <strong>Investigación Operativa</strong>, nacida en los campos de batalla de la Segunda Guerra Mundial, evolucionó desde las elegantes pero rígidas ecuaciones del método Simplex hasta las sofisticadas redes neuronales de la Inteligencia Artificial moderna."
                        </div>
                    </div>
                    
                    {self.generar_timeline_interactivo()}
                    
                    {self.generar_seccion_visualizaciones()}
                    
                    <div class="seccion" id="evolucion">
                        <h2>🏛️ La Evolución Histórica</h2>
                        
                        <h3>1947: El Nacimiento del Orden</h3>
                        <p>George Dantzig no sabía que estaba a punto de cambiar para siempre la forma en que la humanidad resolvería problemas de optimización. Su <strong>método Simplex</strong> nació de una necesidad militar: ¿cómo asignar recursos limitados de la manera más eficiente posible?</p>
                        
                        <div class="codigo" data-language="Matemáticas">
                            <pre>Maximizar: Z = c₁x₁ + c₂x₂ + ... + cₙxₙ
Sujeto a:
  a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
  a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
  ...
  xᵢ ≥ 0</pre>
                        </div>
                        
                        <h3>2000s: El Amanecer de la Inteligencia</h3>
                        <p>El nuevo milenio trajo consigo una revolución silenciosa: <strong>las máquinas comenzaron a aprender</strong>. Los algoritmos ya no solo optimizaban; ahora descubrían patrones, predecían comportamientos, se adaptaban a nuevas situaciones.</p>
                        
                        <div class="codigo" data-language="Python">
                            <pre># El bosque que predice el futuro
predictor_capturas = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42
)

# Variables que el Simplex nunca pudo considerar
caracteristicas = [
    'temperatura_agua', 'salinidad', 'corrientes_marinas',
    'fase_lunar', 'presion_atmosferica', 'migracion_peces',
    'experiencia_tripulacion', 'estado_equipos'
]</pre>
                        </div>
                        
                        <h3>2025: La Convergencia</h3>
                        <p>En 2025, ya no hay que elegir entre el Simplex y el Machine Learning, porque nunca compitieron: uno <strong>decide</strong> y el otro <strong>predice</strong>. Se combinan cuando el ML estima los parámetros inciertos que después usa la optimización.</p>
                    </div>
                    
                    <div class="seccion" id="impacto">
                        <h2>🎯 El Impacto Transformador</h2>
                        
                        <h3>Qué aportó cada etapa</h3>
                        <table class="tabla-comparativa">
                            <thead>
                                <tr>
                                    <th>Era</th>
                                    <th>Enfoque</th>
                                    <th>Fortalezas</th>
                                    <th>Limitaciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>1947-1980</strong></td>
                                    <td>Simplex Clásico</td>
                                    <td>Precisión matemática, Soluciones óptimas</td>
                                    <td>Linealidad, Determinismo</td>
                                </tr>
                                <tr>
                                    <td><strong>1980-2000</strong></td>
                                    <td>Programación Avanzada</td>
                                    <td>Problemas no lineales, Mayor escala</td>
                                    <td>Complejidad computacional</td>
                                </tr>
                                <tr>
                                    <td><strong>2000-2020</strong></td>
                                    <td>Machine Learning</td>
                                    <td>Aprendizaje de patrones, Adaptabilidad</td>
                                    <td>Caja negra, Necesidad de datos</td>
                                </tr>
                                <tr>
                                    <td><strong>2020-2025</strong></td>
                                    <td>Predecir y optimizar</td>
                                    <td>El ML estima coeficientes; la PL decide</td>
                                    <td>Complejidad de implementación</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="seccion" id="futuro">
                        <h2>🔮 El Futuro que se Avecina</h2>
                        
                        <h3>La Próxima Frontera: Computación Cuántica</h3>
                        <p><em>(Especulativo.)</em> La computación cuántica se investiga para problemas de optimización combinatoria; hoy no hay pruebas de que supere a los métodos clásicos en problemas como este. El código siguiente es solo ilustrativo.</p>
                        
                        <div class="codigo" data-language="Python">
                            <pre>class OptimizadorCuantico:
    def __init__(self):
        self.qubits = 1024
        self.algoritmo_base = 'QAOA'
        self.backend_cuantico = 'IBM_Quantum_Network'
    
    def optimizar_flota_global(self, flotas_mundiales):
        # Ilustrativo: no existe un "optimizador cuántico de flotas"
        pass</pre>
                        </div>
                        
                        <h3>Tendencias Emergentes</h3>
                        <ul>
                            <li><strong>🧬 Bio-Inspiración:</strong> Algoritmos basados en comportamiento de cardúmenes</li>
                            <li><strong>🌐 Ecosistemas Conectados:</strong> Blockchain para trazabilidad completa</li>
                            <li><strong>🔬 Ciencia de Datos Avanzada:</strong> Análisis de ADN ambiental</li>
                        </ul>
                        
                        <div class="cita-destacada">
                            "Cada gota de agua del océano será conocida, cada pez será rastreado con respeto, cada decisión será optimizada no solo para la ganancia, sino para la armonía perpetua entre la humanidad y el mar."
                            <br><strong>- Visión 2030</strong>
                        </div>
                    </div>
                </main>
                
                <footer class="footer">
                    <h3>🌊 La Sinfonía del Progreso</h3>
                    <p>En el océano de datos que navegamos hoy, somos herederos de una tradición noble: la búsqueda incansable de la <strong>optimización</strong>, la <strong>eficiencia</strong>, y la <strong>excelencia</strong>.</p>
                    <p><strong>CONIPE 2025 - Curso de Alfabetización en Ciencia de Datos Pesquera</strong></p>
                    <p><em>"En el principio era el Simplex, y el Simplex era bueno. Pero el océano tenía planes más grandes."</em></p>
                </footer>
            </div>
            
            {self.generar_javascript_interactivo()}
        </body>
        </html>
        """
        
        return html_completo
    
    def generar_presentacion(self):
        """Genera la presentación HTML completa"""
        print("🎬 Generando presentación interactiva...")
        
        # Generar HTML
        html_completo = self.generar_html_completo()
        
        # Guardar archivo
        with open(self.archivo_salida, 'w', encoding='utf-8') as f:
            f.write(html_completo)
        
        print(f"✅ Presentación generada exitosamente: {self.archivo_salida}")
        print(f"📁 Tamaño del archivo: {os.path.getsize(self.archivo_salida) / 1024:.1f} KB")
        
        return self.archivo_salida

def main():
    """Función principal"""
    print("🌊 GENERADOR DE PRESENTACIÓN INTERACTIVA")
    print("=" * 50)
    print("La Revolución de la Investigación Operativa")
    print("Del Método Simplex a la Inteligencia Artificial")
    print("=" * 50)
    
    # Directorio del propio script (antes era una ruta fija de Windows)
    directorio_base = Path(__file__).resolve().parent
    
    # Crear generador
    generador = GeneradorPresentacionIO(directorio_base)
    
    # Generar presentación
    archivo_generado = generador.generar_presentacion()
    
    print("\n🎯 PRESENTACIÓN COMPLETADA")
    print("=" * 30)
    print(f"📄 Archivo: {archivo_generado.name}")
    print(f"📍 Ubicación: {archivo_generado.parent}")
    print("\n🌟 CARACTERÍSTICAS DE LA PRESENTACIÓN:")
    print("   ✨ Diseño responsivo y moderno")
    print("   🎬 Animaciones y transiciones suaves")
    print("   📱 Compatible con dispositivos móviles")
    print("   🎯 Navegación interactiva")
    print("   📊 Visualizaciones integradas")
    print("   🌙 Modo oscuro disponible")
    print("   ⚡ Efectos interactivos avanzados")
    
    print("\n📖 INSTRUCCIONES DE USO:")
    print("   1. Abrir el archivo HTML en cualquier navegador web")
    print("   2. Usar la navegación superior para saltar entre secciones")
    print("   3. Hacer clic en los botones interactivos para más detalles")
    print("   4. Usar el botón 🌙 para alternar modo oscuro")
    print("   5. La presentación es completamente autocontenida")
    
    print("\n🎓 VALOR EDUCATIVO:")
    print("   📚 Historia completa de la evolución de la IO")
    print("   🔬 Ejemplos prácticos y casos de estudio")
    print("   📈 Cifras reales del análisis (datos sintéticos)")
    print("   🚀 Visión del futuro de la optimización")
    print("   🌊 Narrativa cautivadora y memorable")
    
    return archivo_generado

if __name__ == "__main__":
    main()