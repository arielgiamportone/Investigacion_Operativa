# 🚀 Guía de Instalación Rápida

## Investigación Operativa: Simplex a Machine Learning

### ⚡ Instalación Express (5 minutos)

#### 1. Verificar Python
```bash
python --version
# Debe ser Python 3.8 o superior
```

#### 2. Crear Entorno Virtual
```bash
# Navegar al directorio del proyecto
cd "C:\Users\Ariel Giamporte\Desktop\Conipe2025\curso_alfabetizacion_CONIPE25-main\extras\Investigacion_Operativa"

# Crear entorno virtual
python -m venv env

# Activar entorno (Windows)
env\Scripts\activate

# Activar entorno (Linux/Mac)
source env/bin/activate
```

#### 3. Instalar Dependencias
```bash
# Instalación básica
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly

# Verificar instalación
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, scipy, plotly; print('✅ Todas las dependencias instaladas correctamente')"
```

### 🎯 Ejecución Rápida

#### Ejecutar Análisis Completo
```bash
# Script principal
python simplex_ml_optimizacion_pesquera.py

# Análisis comparativo
python analisis_comparativo_io.py

# Presentación interactiva
python presentacion_storytelling_io.py
```

### 📁 Archivos Generados

Después de la ejecución, encontrarás:

- `presentacion_interactiva_io.html` - Presentación principal
- `analisis_comparativo_io_completo.html` - Visualizaciones comparativas
- `reporte_ejecutivo_analisis_io.html` - Reporte ejecutivo
- `visualizaciones_ml_rutas.html` - Análisis ML detallado

### 🔧 Solución de Problemas

#### Error: "No module named 'plotly'"
```bash
pip install plotly
```

#### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

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
    print("✅ Todas las dependencias están disponibles")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Plotly: {plotly.__version__}")
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