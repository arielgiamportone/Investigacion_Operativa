# 🚀 Guía de Uso - Aplicación Streamlit IO Interactive

## 📋 Descripción General

La aplicación **IO Interactive** es una herramienta educativa interactiva que permite comparar diferentes métodos de Investigación Operativa:
- **Método Simplex** (Programación Lineal Clásica)
- **Machine Learning** (Random Forest, Gradient Boosting, SVM, Neural Networks)
- **Análisis Comparativo** entre todos los métodos

## 🚀 Cómo Ejecutar la Aplicación

### Paso 1: Instalación de Dependencias
```bash
pip install streamlit numpy pandas matplotlib seaborn plotly scikit-learn scipy
```

### Paso 2: Ejecutar la Aplicación
```bash
streamlit run app_streamlit_io.py
```

### Paso 3: Abrir en el Navegador
La aplicación se abrirá automáticamente en: `http://localhost:8501`

## 🎯 Funcionalidades Principales

### 1. **Configuración Inicial (Barra Lateral)**

#### 📊 Generar Datos Sintéticos
- **Número de escenarios**: 100-2000 (recomendado: 1000)
- **Semilla aleatoria**: Para reproducibilidad
- Clic en **"🔄 Generar Datos"**

#### 🧠 Entrenar Modelos ML
- Después de generar datos, clic en **"🎯 Entrenar Modelos ML"**
- Se entrenarán automáticamente 4 modelos diferentes

#### 🎛️ Parámetros del Problema
- **Recursos Disponibles**: Combustible, Tripulación, Presupuesto
- **Condiciones Ambientales**: Temperatura, Salinidad, Viento, Olas
- **Factores de Mercado**: Precio, Demanda, Competencia
- **Factores Temporales**: Mes, Fase Lunar
- **Variables de Decisión**: Número de barcos por tipo

### 2. **Pestaña Simplex 🏛️**

#### Características:
- Optimización determinística garantizada
- Solución óptima matemática
- Visualización 3D del espacio de soluciones
- Tiempo de ejecución ultra-rápido

#### Cómo usar:
1. Ajustar parámetros en la barra lateral
2. Clic en **"🚀 Optimizar con Simplex"**
3. Observar la solución óptima y visualización 3D

### 3. **Pestaña Machine Learning 🧠**

#### Características:
- 4 modelos diferentes: Random Forest, Gradient Boosting, SVM, Neural Network
- Métricas de rendimiento (R², MSE, MAE)
- Predicciones vs valores reales
- Comparación visual de modelos

#### Cómo usar:
1. Asegurar que los modelos estén entrenados
2. Seleccionar un modelo del dropdown
3. Ajustar variables de decisión en la barra lateral
4. Clic en **"🔮 Predecir con ML"**
5. Analizar métricas y gráficos de rendimiento

### 4. **Pestaña Comparación 📊**

#### Características:
- Comparación directa de velocidad de ejecución
- Comparación de precisión entre modelos ML
- Tabla comparativa completa
- Recomendaciones automáticas

#### Cómo usar:
1. Los análisis se ejecutan automáticamente
2. Observar gráficos de tiempo de ejecución
3. Analizar precisión por método
4. Leer recomendaciones de uso

### 5. **Pestaña Análisis 📈**

#### Características:
- Estadísticas descriptivas completas
- Matriz de correlación interactiva
- Distribuciones de variables
- Análisis de sensibilidad
- Insights automáticos

#### Cómo usar:
1. Explorar estadísticas descriptivas
2. Analizar correlaciones entre variables
3. Seleccionar variables para análisis detallado
4. Observar impacto de condiciones ambientales
5. Leer insights automáticos generados

## 🎯 Flujo de Trabajo Recomendado

### Para Estudiantes:
1. **Generar datos** (1000 escenarios)
2. **Entrenar modelos ML**
3. **Explorar Simplex** con diferentes parámetros
4. **Comparar con ML** usando los mismos parámetros
5. **Analizar diferencias** en la pestaña Comparación
6. **Explorar datos** en la pestaña Análisis

### Para Educadores:
1. **Demostrar Simplex** como método clásico
2. **Mostrar limitaciones** de programación lineal
3. **Introducir ML** como alternativa moderna
4. **Comparar métodos** lado a lado
5. **Discutir casos de uso** apropiados

## 🔧 Parámetros Importantes

### Recursos (Restricciones del Simplex):
- **Combustible**: 1000-5000 L
- **Tripulación**: 50-300 personas
- **Presupuesto**: $50,000-$200,000

### Condiciones Ambientales (Factores ML):
- **Temperatura**: 15-25°C (óptimo: 20°C)
- **Viento**: 0-30 km/h (menor es mejor)
- **Salinidad**: 30-40 ppt
- **Olas**: 0.5-4.0 m

### Variables de Decisión:
- **Barcos Pequeños**: 0-20 (costo: $1000, capturas: 50kg)
- **Barcos Medianos**: 0-15 (costo: $2500, capturas: 120kg)
- **Barcos Grandes**: 0-10 (costo: $5000, capturas: 200kg)

## 📊 Interpretación de Resultados

### Métricas de ML:
- **R² Score**: 0-1 (más alto = mejor predicción)
- **MSE**: Error cuadrático medio (más bajo = mejor)
- **MAE**: Error absoluto medio (más bajo = mejor)

### Comparación de Métodos:
- **Simplex**: Rápido, óptimo, pero limitado a problemas lineales
- **Random Forest**: Balanceado, robusto, interpretable
- **Gradient Boosting**: Máxima precisión, pero más lento
- **SVM**: Bueno para alta dimensionalidad
- **Neural Network**: Patrones complejos, requiere más datos

## 🚨 Solución de Problemas

### Error: "Primero genera los datos"
**Solución**: Usar la barra lateral para generar datos sintéticos

### Error: "Primero entrena los modelos ML"
**Solución**: Después de generar datos, entrenar modelos ML

### Aplicación lenta
**Solución**: Reducir número de escenarios a 500-1000

### Gráficos no se muestran
**Solución**: Verificar instalación de plotly: `pip install plotly`

### No se encuentra solución factible (Simplex)
**Solución**: Aumentar recursos disponibles (combustible, tripulación, presupuesto)

## 🎓 Objetivos Educativos

### Conceptos Clave:
1. **Programación Lineal**: Optimización con restricciones lineales
2. **Machine Learning**: Predicción basada en patrones históricos
3. **Trade-offs**: Velocidad vs Precisión vs Flexibilidad
4. **Validación**: Importancia de métricas de evaluación
5. **Aplicación Práctica**: Casos de uso reales en optimización

### Preguntas para Reflexión:
1. ¿Cuándo usar Simplex vs ML?
2. ¿Qué factores ambientales son más importantes?
3. ¿Cómo afecta el tamaño de datos al rendimiento?
4. ¿Qué modelo ML es más apropiado para este problema?
5. ¿Cómo se pueden combinar ambos enfoques?

## 🔗 Recursos Adicionales

- **Documentación Streamlit**: https://docs.streamlit.io/
- **Scikit-learn**: https://scikit-learn.org/
- **SciPy Optimize**: https://docs.scipy.org/doc/scipy/reference/optimize.html
- **Plotly**: https://plotly.com/python/

---

**Desarrollado para CONIPE 2025 - Curso de Alfabetización en Ciencia de Datos**

*Esta aplicación demuestra la evolución de la Investigación Operativa desde métodos clásicos hasta enfoques modernos de IA.*