# 🚀 Proyecto de Investigación Operativa: De Simplex a Machine Learning

## 📋 Descripción del Proyecto

Este proyecto educativo demuestra la evolución de la Investigación Operativa desde los métodos clásicos como el Simplex hasta las técnicas modernas de Machine Learning e Inteligencia Artificial. Utilizando el caso práctico de optimización de flotas pesqueras, exploramos cómo diferentes enfoques pueden resolver problemas complejos de optimización.

## 🎯 Objetivos Educativos

- **Comprender** la evolución histórica de la Investigación Operativa
- **Comparar** métodos clásicos vs modernos de optimización
- **Aplicar** técnicas de Machine Learning a problemas de IO
- **Evaluar** el rendimiento de diferentes enfoques
- **Desarrollar** soluciones híbridas que combinen lo mejor de ambos mundos

## 📁 Estructura del Proyecto

```
Investigacion_Operativa/
├── 📄 README.md                                    # Este archivo
├── 🚀 app_streamlit_io.py                         # APLICACIÓN INTERACTIVA STREAMLIT
├── 🐍 simplex_ml_optimizacion_pesquera.py         # Script principal: Simplex + ML
├── 🎬 animacion_evolucion_simplex_ml.py           # Generador de animación evolutiva
├── 🧠 ml_prediccion_rutas_optimas.py              # Modelos ML avanzados
├── 📊 analisis_comparativo_io.py                  # Análisis comparativo completo
├── 📖 storytelling_evolucion_io.md                # Narrativa educativa
├── 🎨 presentacion_storytelling_io.py             # Generador de presentación
├── 📱 GUIA_STREAMLIT.md                           # Guía de uso de la app Streamlit
├── 📦 requirements_streamlit.txt                  # Dependencias específicas para Streamlit
├── 🌐 presentacion_interactiva_io.html            # Presentación interactiva
├── 📈 analisis_comparativo_io_completo.html       # Visualizaciones comparativas
├── 📋 reporte_ejecutivo_analisis_io.html          # Reporte ejecutivo
└── 🎯 visualizaciones_ml_rutas.html               # Visualizaciones ML
```

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv env

# Activar entorno virtual
# En Windows:
env\Scripts\activate
# En Linux/Mac:
source env/bin/activate

# Instalar dependencias básicas
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly

# Para la aplicación Streamlit (adicional)
pip install streamlit
# O usar el archivo de requisitos específico:
pip install -r requirements_streamlit.txt
```

### Dependencias Específicas

```python
# Análisis de datos y visualización
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Machine Learning
scikit-learn>=1.0.0
scipy>=1.7.0

# Utilidades
warnings
time
datetime
json
```

## 🚀 Guía de Uso

### 🌟 APLICACIÓN INTERACTIVA STREAMLIT (RECOMENDADO)

```bash
streamlit run app_streamlit_io.py
```

**🎯 Características de la App Interactiva:**
- **Interfaz web moderna** con controles deslizantes
- **Comparación en tiempo real** entre Simplex y ML
- **Visualizaciones dinámicas** con Plotly
- **4 pestañas especializadas**: Simplex, ML, Comparación, Análisis
- **Parámetros ajustables** para exploración interactiva
- **Generación de datos sintéticos** configurable
- **Entrenamiento de modelos ML** en vivo
- **Análisis de sensibilidad** automático

📱 **Ver**: `GUIA_STREAMLIT.md` para instrucciones detalladas

### 1. Script Principal: Optimización Híbrida

```bash
python simplex_ml_optimizacion_pesquera.py
```

**Funcionalidades:**
- Genera datos sintéticos de flotas pesqueras
- Aplica optimización Simplex clásica
- Entrena modelos de Machine Learning
- Implementa enfoque híbrido
- Crea visualizaciones comparativas

### 2. Animación Evolutiva

```bash
python animacion_evolucion_simplex_ml.py
```

**Genera:**
- GIF animado mostrando la evolución de IO
- Frames históricos desde 1947 hasta 2025
- Comparación visual de metodologías

### 3. Modelos ML Avanzados

```bash
python ml_prediccion_rutas_optimas.py
```

**Incluye:**
- Random Forest, Gradient Boosting, SVM, Neural Networks
- Análisis exploratorio de datos
- Optimización de hiperparámetros
- Visualizaciones interactivas

### 4. Análisis Comparativo Completo

```bash
python analisis_comparativo_io.py
```

**Proporciona:**
- Benchmarking de 6 métodos diferentes
- Métricas de rendimiento detalladas
- Reporte ejecutivo profesional
- Recomendaciones estratégicas

### 5. Presentación Interactiva

```bash
python presentacion_storytelling_io.py
```

**Crea:**
- Presentación HTML interactiva
- Narrativa educativa integrada
- Timeline interactivo
- Modo oscuro/claro

## 📊 Métodos Evaluados

### 🏛️ Métodos Clásicos

1. **Simplex Clásico**
   - ✅ Solución óptima garantizada
   - ✅ Rápido para problemas lineales
   - ❌ Solo problemas lineales
   - ❌ No maneja incertidumbre

### 🧠 Machine Learning

2. **Random Forest**
   - ✅ Maneja no-linealidad
   - ✅ Robusto al sobreajuste
   - ❌ Menos interpretable
   - ❌ Requiere datos abundantes

3. **Gradient Boosting**
   - ✅ Alta precisión
   - ✅ Captura patrones complejos
   - ❌ Propenso al sobreajuste
   - ❌ Tiempo de entrenamiento

4. **Support Vector Machine (SVM)**
   - ✅ Efectivo en alta dimensionalidad
   - ✅ Memoria eficiente
   - ❌ Lento en datasets grandes
   - ❌ Sensible a escalado

5. **Neural Networks**
   - ✅ Aproximación universal
   - ✅ Aprende representaciones
   - ❌ Caja negra
   - ❌ Requiere muchos datos

### 🚀 Enfoque Híbrido

6. **Simplex + ML**
   - ✅ Combina fortalezas
   - ✅ Adaptable y robusto
   - ❌ Mayor complejidad
   - ❌ Requiere expertise dual

## 📈 Resultados Destacados

### Métricas de Rendimiento

| Método | R² Capturas | MSE | Tiempo (ms) | Clasificación |
|--------|-------------|-----|-------------|---------------|
| **Gradient Boosting** | 0.878 | Bajo | Medio | ⭐ Mejor Precisión |
| **Neural Network** | 0.847 | Bajo | Alto | ⭐ Alta Capacidad |
| **Random Forest** | 0.769 | Medio | Bajo | ⭐ Equilibrado |
| **SVM** | 0.640 | Medio | Muy Bajo | ⭐ Más Rápido |
| **Híbrido** | Variable | Variable | Medio | ⭐ Más Robusto |
| **Simplex** | Variable | Alto | Muy Bajo | ⭐ Garantías |

### Recomendaciones por Escenario

- **🎯 Problemas Lineales Simples:** Simplex Clásico
- **📊 Datos Abundantes:** Random Forest o Gradient Boosting
- **⚡ Tiempo Real:** SVM o Simplex
- **🔬 Exploración:** Neural Networks
- **🚀 Aplicaciones Críticas:** Enfoque Híbrido

## 🎓 Valor Educativo

### Para Estudiantes
- Comprensión práctica de métodos de IO
- Experiencia hands-on con ML
- Análisis comparativo objetivo
- Casos de uso reales

### Para Profesionales
- Benchmarking de metodologías
- Guías de selección de métodos
- Implementaciones listas para usar
- Mejores prácticas

### Para Investigadores
- Framework de evaluación
- Datos sintéticos realistas
- Métricas estandarizadas
- Base para extensiones

## 🔬 Casos de Uso Específicos

### 🐟 Optimización de Flotas Pesqueras
- **Variables:** Tipos de barcos, recursos, condiciones ambientales
- **Objetivos:** Maximizar capturas, minimizar costos
- **Restricciones:** Combustible, tripulación, presupuesto
- **Complejidad:** Factores no-lineales, incertidumbre

### 📊 Análisis de Sensibilidad
- Impacto de condiciones climáticas
- Variaciones en precios de mercado
- Disponibilidad de recursos
- Competencia en zonas de pesca

## 🛡️ Consideraciones Técnicas

### Calidad de Datos
- **Sintéticos:** Generados con patrones realistas
- **Ruido:** Incluido para simular condiciones reales
- **Correlaciones:** Modeladas según conocimiento del dominio
- **Escalabilidad:** 1000+ escenarios para robustez

### Validación de Modelos
- **Cross-validation:** 5-fold para todos los modelos ML
- **Métricas múltiples:** R², MSE, MAE
- **Benchmarking:** Comparación sistemática
- **Interpretabilidad:** Análisis de importancia de características

## 🔮 Extensiones Futuras

### Técnicas Avanzadas
- **Optimización Multiobjetivo:** Pareto-optimal solutions
- **Programación Estocástica:** Manejo de incertidumbre
- **Deep Learning:** Redes neuronales profundas
- **Reinforcement Learning:** Aprendizaje por refuerzo

### Aplicaciones Adicionales
- **Logística:** Optimización de rutas de transporte
- **Manufactura:** Programación de producción
- **Finanzas:** Optimización de portafolios
- **Energía:** Gestión de redes eléctricas

## 📚 Referencias y Recursos

### Literatura Clásica
- Dantzig, G. B. (1963). *Linear Programming and Extensions*
- Hillier, F. S., & Lieberman, G. J. (2015). *Introduction to Operations Research*
- Winston, W. L. (2003). *Operations Research: Applications and Algorithms*

### Machine Learning
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Géron, A. (2019). *Hands-On Machine Learning*

### Recursos Online
- [Scipy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Plotly Documentation](https://plotly.com/python/)

## 🤝 Contribuciones

Este proyecto es parte del **CONIPE 2025 - Curso de Alfabetización en Ciencia de Datos**. Las contribuciones son bienvenidas:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autores

- **Sistema de IA Educativa** - *Desarrollo inicial* - [CONIPE 2025]
- **Comunidad CONIPE** - *Revisión y mejoras*

## 🙏 Agradecimientos

- **CONIPE 2025** por la oportunidad educativa
- **Comunidad de Investigación Operativa** por las bases teóricas
- **Desarrolladores de Python** por las herramientas excepcionales
- **Estudiantes y profesionales** que utilizan este material

---

## 📞 Contacto y Soporte

Para preguntas, sugerencias o soporte:

- **Email:** conipe2025@educacion.gov
- **Foro:** [CONIPE Community Forum]
- **Issues:** [GitHub Issues]

---

**¡Explora, aprende y optimiza! 🚀**

*"La Investigación Operativa no es solo sobre encontrar la solución óptima, sino sobre entender el problema lo suficientemente bien como para saber qué optimizar."*

---

*Última actualización: Enero 2025*