# Prompt para Video Explicativo: Aplicación Streamlit de Investigación Operativa

## Información General del Video
- **Duración**: 3 minutos
- **Audiencia**: Estudiantes del curso CONIPE 2025
- **Objetivo**: Explicar el funcionamiento de la app Streamlit para optimización pesquera

## Estructura del Video (3 minutos)

### Introducción (30 segundos)
**Texto sugerido:**
"¡Bienvenidos al tutorial de nuestra aplicación interactiva de Investigación Operativa! En los próximos 3 minutos aprenderás a usar esta herramienta que combina métodos clásicos como Simplex con técnicas modernas de Machine Learning para optimizar operaciones pesqueras. Esta aplicación está disponible en http://localhost:8501 y te permitirá experimentar con diferentes enfoques de optimización."

**Elementos visuales:**
- Pantalla de inicio de la aplicación Streamlit
- Logo del curso CONIPE 2025
- Transición suave hacia la interfaz principal

### Sección 1: Navegación y Configuración Inicial (45 segundos)
**Texto sugerido:**
"La aplicación tiene 4 pestañas principales: Simplex, Machine Learning, Comparación y Análisis. Comencemos configurando los parámetros básicos en la barra lateral. Aquí puedes ajustar recursos como combustible, tripulación y tiempo disponible, así como condiciones ambientales como temperatura del agua y corrientes marinas."

**Elementos visuales:**
- Mostrar las 4 pestañas claramente
- Destacar la barra lateral con parámetros
- Ajustar algunos valores en tiempo real
- Mostrar cómo cambian los datos sintéticos generados

### Sección 2: Pestaña Simplex - Método Clásico (45 segundos)
**Texto sugerido:**
"En la pestaña Simplex aplicamos programación lineal clásica. Observa cómo el algoritmo encuentra la solución óptima determinística. Los gráficos muestran la distribución de recursos, las restricciones activas y el beneficio neto por zona de pesca. Este método es rápido y garantiza la solución óptima cuando el problema es lineal."

**Elementos visuales:**
- Ejecutar optimización Simplex
- Mostrar tabla de resultados
- Destacar gráficos de barras y scatter plot
- Señalar métricas clave: tiempo de ejecución y beneficio total

### Sección 3: Pestaña Machine Learning - Métodos Predictivos (45 segundos)
**Texto sugerido:**
"En Machine Learning probamos 4 algoritmos: Random Forest, Gradient Boosting, SVM y Redes Neuronales. Cada modelo aprende patrones de los datos y predice soluciones óptimas. Observa las métricas de evaluación: precisión, recall y F1-score. Gradient Boosting suele ser el más preciso, mientras que Random Forest ofrece el mejor balance."

**Elementos visuales:**
- Entrenar modelos en tiempo real
- Mostrar métricas de evaluación
- Comparar predicciones de diferentes algoritmos
- Destacar matriz de confusión y gráficos de rendimiento

### Sección 4: Comparación y Análisis (30 segundos)
**Texto sugerido:**
"La pestaña Comparación muestra todos los métodos lado a lado. Simplex es el más rápido, pero ML captura mejor la complejidad real. El Análisis incluye estadísticas descriptivas y análisis de sensibilidad para entender cómo los parámetros afectan los resultados."

**Elementos visuales:**
- Tabla comparativa de todos los métodos
- Gráficos de tiempo vs precisión
- Mostrar análisis de sensibilidad
- Destacar estadísticas descriptivas

### Interpretación de Resultados (15 segundos)
**Texto sugerido:**
"Para interpretar correctamente: valores más altos en beneficio neto indican mejores zonas, tiempos de ejecución menores son preferibles para decisiones en tiempo real, y métricas de ML arriba del 80% indican modelos confiables."

**Elementos visuales:**
- Señalar indicadores clave en cada gráfico
- Mostrar rangos de valores aceptables
- Destacar alertas o warnings importantes

## Casos de Uso Específicos a Mencionar

### Caso 1: Planificación de Flota Pesquera
"Usa Simplex cuando necesites decisiones rápidas con recursos limitados conocidos."

### Caso 2: Predicción de Zonas Óptimas
"Aplica ML cuando tengas datos históricos y quieras predecir patrones futuros."

### Caso 3: Análisis Comparativo
"Utiliza la comparación para validar decisiones y entender trade-offs entre velocidad y precisión."

### Caso 4: Investigación y Educación
"Experimenta con diferentes parámetros para entender cómo factores ambientales afectan la optimización."

## Qué Deben Entender los Usuarios de los Resultados

### Métricas Clave
1. **Beneficio Neto**: Ganancia esperada por zona (mayor es mejor)
2. **Tiempo de Ejecución**: Velocidad del algoritmo (menor es mejor para tiempo real)
3. **Precisión ML**: Porcentaje de predicciones correctas (>80% es bueno)
4. **Recursos Utilizados**: Eficiencia en el uso de combustible y tiempo

### Interpretación de Gráficos
1. **Scatter Plot**: Tamaño de puntos = beneficio, colores = diferentes zonas
2. **Gráficos de Barras**: Comparación directa entre métodos
3. **Matriz de Confusión**: Diagonal principal alta = buen modelo
4. **Análisis de Sensibilidad**: Pendientes pronunciadas = parámetros críticos

### Señales de Alerta
- Beneficios negativos: revisar restricciones
- Precisión ML <70%: necesita más datos o ajuste de parámetros
- Tiempos de ejecución >10 segundos: problema de escalabilidad
- Recursos sobreutilizados: solución no factible

### Recomendaciones de Uso
1. **Para decisiones operativas**: Usar Simplex por velocidad
2. **Para planificación estratégica**: Usar ML por precisión
3. **Para investigación**: Usar comparación completa
4. **Para aprendizaje**: Experimentar con todos los parámetros

## Elementos Técnicos a Destacar

### Tecnologías Utilizadas
- Streamlit para interfaz interactiva
- SciPy para optimización Simplex
- Scikit-learn para Machine Learning
- Plotly para visualizaciones dinámicas

### Ventajas Educativas
- Comparación directa entre métodos clásicos y modernos
- Visualización en tiempo real de resultados
- Parámetros configurables para experimentación
- Métricas cuantitativas para evaluación objetiva

## Llamada a la Acción Final
"¡Ahora es tu turno! Accede a la aplicación, experimenta con diferentes configuraciones y descubre cómo la Investigación Operativa puede optimizar las operaciones pesqueras. Recuerda que cada método tiene sus fortalezas: Simplex para velocidad, ML para precisión, y la combinación de ambos para robustez."

---

## Notas para el Productor del Video

### Estilo Visual Recomendado
- Transiciones suaves entre secciones
- Zoom en elementos importantes de la interfaz
- Uso de flechas y destacados para guiar la atención
- Paleta de colores consistente con la aplicación

### Audio y Narración
- Tono educativo pero dinámico
- Pausas estratégicas para asimilar información
- Énfasis en palabras clave (Simplex, Machine Learning, optimización)
- Música de fondo sutil y profesional

### Elementos Interactivos a Mostrar
- Movimiento de sliders en tiempo real
- Actualización automática de gráficos
- Cambio entre pestañas fluido
- Ejecución de algoritmos con indicadores de progreso

### Duración por Sección (Timing Detallado)
- 0:00-0:30: Introducción y bienvenida
- 0:30-1:15: Navegación y configuración
- 1:15-2:00: Demostración Simplex
- 2:00-2:45: Demostración Machine Learning
- 2:45-3:00: Comparación y cierre

Este prompt está diseñado para crear un video educativo completo que maximice el aprendizaje en solo 3 minutos, cubriendo tanto aspectos técnicos como prácticos de la aplicación Streamlit de Investigación Operativa.