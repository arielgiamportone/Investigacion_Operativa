# Prompt para Video Explicativo: Aplicación Streamlit de Investigación Operativa

## Información General del Video
- **Duración**: 3 minutos
- **Audiencia**: Estudiantes del curso CONIPE 2025
- **Objetivo**: Explicar el funcionamiento de la app Streamlit para optimización pesquera

## Estructura del Video (3 minutos)

### Introducción (30 segundos)
**Texto sugerido:**
"¡Bienvenidos al tutorial de nuestra aplicación interactiva de Investigación Operativa! En los próximos 3 minutos aprenderás a usar esta herramienta que usa el método Simplex para decidir y el Machine Learning para predecir, en una flota pesquera con datos sintéticos. Esta aplicación está disponible en http://localhost:8501 y te permitirá experimentar con diferentes enfoques de optimización."

**Elementos visuales:**
- Pantalla de inicio de la aplicación Streamlit
- Logo del curso CONIPE 2025
- Transición suave hacia la interfaz principal

### Sección 1: Navegación y Configuración Inicial (45 segundos)
**Texto sugerido:**
"La aplicación tiene 4 pestañas principales: Simplex, Machine Learning, Decidir vs. predecir y Análisis. Comencemos configurando los parámetros básicos en la barra lateral. Aquí puedes ajustar recursos como combustible, tripulación y presupuesto, así como condiciones ambientales como la temperatura del agua y el viento. Un aviso importante: los datos son sintéticos, generados con una fórmula conocida; no son de una flota real."

**Elementos visuales:**
- Mostrar las 4 pestañas claramente
- Destacar la barra lateral con parámetros
- Ajustar algunos valores en tiempo real
- Mostrar cómo cambian los datos sintéticos generados

### Sección 2: Pestaña Simplex - Método Clásico (45 segundos)
**Texto sugerido:**
"En la pestaña Simplex aplicamos programación lineal. El algoritmo encuentra cuántos barcos de cada tipo enviar para maximizar la captura esperada. La tabla de recursos muestra cuáles limitan y su precio sombra: cuántos kilos más se capturarían con una unidad más de cada recurso. La PL garantiza el óptimo del modelo lineal, siempre que sus coeficientes sean correctos."

**Elementos visuales:**
- Ejecutar optimización Simplex
- Mostrar tabla de resultados
- Destacar gráficos de barras y scatter plot
- Señalar la captura óptima, las holguras y los precios sombra

### Sección 3: Pestaña Machine Learning - Métodos Predictivos (45 segundos)
**Texto sugerido:**
"En Machine Learning probamos 4 algoritmos (Random Forest, Gradient Boosting, SVM y redes neuronales) y los comparamos con dos líneas base: predecir la media y una regresión lineal. Cada modelo predice las capturas; no decide nada. Como es un problema de regresión, se evalúa con R², MSE y MAE. Si un R² sale negativo, el modelo predice peor que la media, y la app lo muestra tal cual."

**Elementos visuales:**
- Entrenar modelos en tiempo real
- Mostrar métricas de evaluación
- Comparar predicciones de diferentes algoritmos
- Destacar el gráfico de predicciones frente a valores reales

### Sección 4: Decidir vs. predecir y Análisis (30 segundos)
**Texto sugerido:**
"Simplex y ML no compiten: responden preguntas distintas. En la pestaña Decidir vs. predecir los combinamos: el ML estima el rendimiento de cada tipo de barco con el clima actual y la PL decide con esas estimaciones. Si subimos el viento, la decisión con los rendimientos de siempre se aleja del óptimo, y la que usa las estimaciones se acerca. El Análisis explora cómo cambian las capturas con las condiciones."

**Elementos visuales:**
- Tabla de modelos frente a líneas base
- Tabla de % del óptimo según el origen de los coeficientes, moviendo el slider de viento
- Destacar estadísticas descriptivas

### Interpretación de Resultados (15 segundos)
**Texto sugerido:**
"Para interpretar correctamente: una decisión se juzga por lo cerca que queda del óptimo; una predicción, por su error frente a una línea base. Y un R² alto en datos sintéticos solo indica que el modelo recupera la fórmula con la que se generaron."

**Elementos visuales:**
- Señalar indicadores clave en cada gráfico
- Mostrar rangos de valores aceptables
- Destacar alertas o warnings importantes

## Casos de Uso Específicos a Mencionar

### Caso 1: Planificación de Flota Pesquera
"Usa la PL para decidir cómo repartir recursos limitados y para saber qué recurso conviene ampliar."

### Caso 2: Predicción de Capturas
"Aplica ML cuando tengas datos históricos y quieras estimar cantidades inciertas; compáralo siempre con la media y una regresión lineal."

### Caso 3: Predecir y luego optimizar
"Usa el ML para estimar los coeficientes inciertos de la PL y evalúa la decisión resultante frente al óptimo."

### Caso 4: Investigación y Educación
"Experimenta con diferentes parámetros para entender cómo factores ambientales afectan la optimización."

## Qué Deben Entender los Usuarios de los Resultados

### Métricas Clave
1. **Captura óptima (PL)**: lo máximo alcanzable con el modelo y los recursos dados
2. **Precio sombra (PL)**: valor de una unidad más de cada recurso; 0 si el recurso sobra
3. **R², MSE, MAE (ML)**: calidad de la predicción frente a la media (R² = 0) y a una regresión lineal
4. **% del óptimo (predecir y optimizar)**: calidad de una decisión tomada con coeficientes estimados

### Interpretación de Gráficos
1. **Predicción vs. real**: puntos cerca de la diagonal = buena predicción
2. **Gráficos de Barras de R²**: comparar con las líneas base, no solo entre modelos
3. **Espacio de soluciones 3D**: puntos factibles coloreados por captura
4. **Capturas según condiciones**: análisis exploratorio de los datos sintéticos

### Señales de Alerta
- R² negativo: el modelo predice peor que la media
- Un modelo complejo que no supera a la regresión lineal: la complejidad no aporta
- "No hay solución factible": las restricciones son incompatibles (es información, no un fallo)

### Recomendaciones de Uso
1. **Para decidir**: PL (y leer sus precios sombra)
2. **Para estimar cantidades inciertas**: ML, frente a líneas base
3. **Cuando los coeficientes de la decisión son inciertos**: predecir y luego optimizar
4. **Para aprendizaje**: Experimentar con todos los parámetros

## Elementos Técnicos a Destacar

### Tecnologías Utilizadas
- Streamlit para interfaz interactiva
- SciPy para optimización Simplex
- Scikit-learn para Machine Learning
- Plotly para visualizaciones dinámicas

### Ventajas Educativas
- Distinción clara entre decidir y predecir, y cómo combinarlos
- Visualización en tiempo real de resultados
- Parámetros configurables para experimentación
- Métricas cuantitativas para evaluación objetiva

## Llamada a la Acción Final
"¡Ahora es tu turno! Accede a la aplicación, experimenta con diferentes configuraciones y descubre cómo la Investigación Operativa puede optimizar las operaciones pesqueras. Recuerda: el Simplex decide, el ML predice, y se combinan cuando el ML estima lo que la decisión necesita saber."

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
- 2:45-3:00: Decidir vs. predecir y cierre

Este prompt está diseñado para crear un video educativo completo que maximice el aprendizaje en solo 3 minutos, cubriendo tanto aspectos técnicos como prácticos de la aplicación Streamlit de Investigación Operativa.