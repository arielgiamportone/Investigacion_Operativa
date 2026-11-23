# 🚀 Guía de Instalación Rápida

## Investigación Operativa: Simplex a Machine Learning

### ⚡ Instalación Express (5 minutos)

#### 1. Verificar Python
```bash
python --version
# Debe ser Python 3.10 o superior
```

#### 2. Crear Entorno Virtual
```bash
# Navegar al directorio del proyecto (la carpeta donde clonaste el repositorio)
cd Investigacion_Operativa

# Crear entorno virtual
python -m venv env

# Activar entorno (Windows)
env\Scripts\activate

# Activar entorno (Linux/Mac)
source env/bin/activate
```

#### 3. Instalar Dependencias
```bash
# Instala todo lo necesario (scripts, app Streamlit y tests)
pip install -r requirements.txt

# Verificar instalación
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, scipy, plotly, streamlit; print('✅ Todas las dependencias instaladas correctamente')"

# Comprobar que todo funciona
pytest
```

### 🎯 Ejecución Rápida

#### Ejecutar Análisis Completo
```bash
# Análisis comparativo (genera el informe y el JSON con las cifras)
python analisis_comparativo_io.py

# Presentación interactiva (usa las cifras del paso anterior)
python presentacion_storytelling_io.py

# Problema de "Don Carlos": PL con precios sombra + ML con líneas base
python simplex_ml_optimizacion_pesquera.py

# Aplicación interactiva
streamlit run app_streamlit_io.py
```

### 📁 Archivos Generados

Después de la ejecución, encontrarás:

- `reporte_ejecutivo_analisis_io.html` - Informe: decidir (PL), predecir (ML) y predecir y optimizar
- `analisis_comparativo_io_completo.html` - Gráficos interactivos del informe
- `resultados_analisis_io.json` - Todas las cifras del informe
- `presentacion_interactiva_io.html` - Presentación principal
- `comparacion_metodos_io.png` - Gráficos de `simplex_ml_optimizacion_pesquera.py`
- `visualizaciones_ml_rutas.html` - Gráficos de `ml_prediccion_rutas_optimas.py`

⚠️ Todos los datos son **sintéticos** (ver `nucleo_io.py` y el README).

### 🔧 Solución de Problemas

#### Error: "No module named 'plotly'"
```bash
pip install -r requirements.txt
```

#### Error: "No module named 'sklearn'"
```bash
pip install -r requirements.txt
```

#### Error al instalar Streamlit con Python 3.8 o 3.9
La app necesita Streamlit 1.51 o superior, que requiere Python 3.10. Actualiza Python.

#### Error de permisos en Windows
```bash
# Ejecutar como administrador o usar:
pip install --user [paquete]
```

### ✅ Verificación de Instalación

```python
# Ejecutar este código para verificar
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    import scipy
    import plotly
    import streamlit
    print("✅ Todas las dependencias están disponibles")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Plotly: {plotly.__version__}")
    print(f"Streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")
```

### 🎓 Primeros Pasos

1. **Leer** el `README.md` completo
2. **Ejecutar** `analisis_comparativo_io.py`
3. **Abrir** `reporte_ejecutivo_analisis_io.html` en el navegador
4. **Explorar** las visualizaciones interactivas
5. **Experimentar** con los parámetros en los scripts

### 📞 Soporte

Si encuentras problemas:
1. Verifica la versión de Python
2. Asegúrate de que el entorno virtual esté activado
3. Reinstala las dependencias si es necesario
4. Consulta el README.md para más detalles

---

**¡Listo para explorar la evolución de la Investigación Operativa! 🚀**