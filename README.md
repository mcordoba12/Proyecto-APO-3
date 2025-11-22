# Sistema de Reconocimiento de Actividades Humanas

**Proyecto Final - Algoritmos y Programación III**  
Universidad ICESI, Semestre 2025-1

## Autores

- Angela Maria Gonzalez Cordoba - A00399435
- Juan Manuel Casanova Marin - A00400090  
- Juliana Filigrana Valencia - A00153988

---

## Descripción del Proyecto

Sistema de reconocimiento de actividades humanas en tiempo real utilizando MediaPipe Pose y aprendizaje supervisado. El sistema clasifica 5 actividades fundamentales:

1. **Caminar hacia la cámara** (caminar_frente)
2. **Caminar de regreso** (caminar_espalda)
3. **Girar**
4. **Sentarse** (sentado)
5. **Ponerse de pie** (ponerse_de_pie)

### Características Principales

- **Detección sin marcadores:** Utiliza únicamente cámara RGB estándar (webcam)
- **Tiempo real:** 26 FPS en CPU sin GPU
- **Alta precisión:** 80.4% accuracy validado con GroupKFold
- **Eficiente:** Reducción de 93.9% en dimensionalidad (1,629 → 100 features)
- **Interpretable:** Random Forest con análisis de feature importance

---

## Datos del Proyecto

**Dataset y archivos generados:**  
[📁 Google Drive - Datos del Proyecto](https://drive.google.com/drive/folders/1R1xmOhAPnwtCdt_xnbxkV7s3HXar5VZk?usp=drive_link)
Video link - (https://youtu.be/viMRYyA53uE)

Contenido:
- Videos originales (18 videos, 15,656 frames)
- Archivos CSV con landmarks extraídos
- Modelo entrenado (`modelo_final.pkl`)
- Gráficos y resultados

---

## Estructura del Repositorio

```
Proyecto-APO-3/
│
├── __pycache__/                          # Archivos de caché de Python
│
├── Entrega1/
│   └── Entrega_1.pdf                     # Reporte primera entrega (EDA)
│
├── Entrega2/
│   └── Entrega_2.pdf                     # Reporte segunda entrega (modelos baseline)
│
├── Entrega3/
│   ├── graficos/                         # Gráficos generados en análisis final
│   │   ├── confusion_matrix_final.png
│   │   ├── diagrama_mediapipe.svg
│   │   ├── diagrama_pipeline.svg
│   │   ├── feature_importance_top20.png
│   │   └── pca_variance.png
│   ├── app_streamlit.py                  # Aplicación web en tiempo real
│   ├── Entrega_3.pdf                     # Reporte tercera entrega
│   ├── Final_report.pdf                  # Reporte final técnico (6 páginas)
│   ├── modelo_final.pkl                  # Modelo Random Forest entrenado
│   └── pipeline_RandomForest.joblib      # Pipeline completo (alternativo)
│
├── input/                                # Datos de entrada (vacío en repo)
├── output/                               # Resultados generados (vacío en repo)
│
├── .gitignore                            # Archivos ignorados por Git
├── INFORME_LINK_NOTEBOOK.docx            # Links a notebooks editables en Colab
├── label_one.py                          # Script auxiliar de etiquetado
├── labeled_coords.py                     # Script procesamiento coordenadas
├── mediapipe_full_export.py              # Script extracción MediaPipe
├── Proyecto_25-2_apo3.pdf                # Documento inicial del proyecto
├── README.md                             # Este archivo (documentación principal)
└── requirements.txt                      # Dependencias del proyecto
```

**Nota importante sobre notebooks editables:**

Los notebooks de Jupyter (.ipynb) **NO están en este repositorio** para mantenerlo ligero. Los notebooks editables están alojados en Google Colab. Encuentra los links en:

📄 **INFORME_LINK_NOTEBOOK.docx** (en la raíz del repositorio)

---

## Entorno Recomendado

### Versión de Python

**Python 3.10 (recomendado)** o Python 3.11

⚠️ **IMPORTANTE:** MediaPipe 0.10.x **NO es compatible** con Python 3.13 y suele dar problemas con Python 3.12

### Paquetes Principales (Obligatorios)

```txt
mediapipe==0.10.14
protobuf==4.25.3          # ⚠️ Obligatorio; versiones 5.x rompen MediaPipe
opencv-python==4.12.0.0   # o 4.10.x si prefieres
numpy>=1.23,<3
pandas>=1.5
scikit-learn>=1.3.0       # Para Random Forest
joblib>=1.3.0             # Para cargar modelo_final.pkl
streamlit>=1.28.0         # Para la aplicación web
```

### Paquetes Opcionales (Recomendados)

```txt
matplotlib>=3.7           # Para gráficos de EDA
seaborn>=0.12            # Para visualizaciones
pyarrow>=14              # Si decides exportar a Parquet
Pillow>=10.0             # Para procesamiento de imágenes
```

---

## Instalación

### Opción 1: Con Conda (Recomendado)

```bash
# Crear entorno virtual
conda create -n pose_env python=3.10 -y
conda activate pose_env

# Actualizar pip
python -m pip install --upgrade pip setuptools wheel

# Instalar dependencias principales
python -m pip install "mediapipe==0.10.14" "protobuf==4.25.3" "opencv-python==4.12.0.0"
python -m pip install numpy pandas scikit-learn joblib streamlit

# Instalar opcionales (para gráficos)
python -m pip install matplotlib seaborn pillow
```

### Opción 2: Con pip y venv

```bash
# Crear entorno virtual
python3.10 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar desde requirements.txt
pip install -r requirements.txt
```

### Opción 3: Instalación Rápida desde requirements.txt

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/proyecto-apo3-har.git
cd proyecto-apo3-har

# Crear entorno e instalar
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso del Sistema

### 1. Descargar Modelo Entrenado (si no está en el repositorio)

⚠️ **Importante:** El archivo `modelo_final.pkl` (~14.6 MB) puede no estar incluido en el repositorio de GitHub debido a limitaciones de tamaño.

**Si no encuentras `modelo_final.pkl` en la carpeta `Entrega3/`:**

1. Descárgalo desde Google Drive: [📁 Carpeta Drive](https://drive.google.com/drive/folders/1R1xmOhAPnwtCdt_xnbxkV7s3HXar5VZk?usp=drive_link)
2. Busca el archivo `modelo_final.pkl`
3. Descárgalo y colócalo en la carpeta `Entrega3/` de tu repositorio local

**Verificar que el modelo existe:**

```bash
# Navegar a la carpeta
cd Entrega3

# Verificar que el archivo existe
ls -lh modelo_final.pkl  # Linux/Mac
dir modelo_final.pkl     # Windows
```

Si el comando anterior muestra el archivo (~14.6 MB), estás listo para continuar.

### 2. Ejecutar Aplicación Web (Streamlit)

```bash
# Navegar a la carpeta
cd Entrega3

# Ejecutar aplicación
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

**Instrucciones de uso:**

1. Haz clic en **"▶️ Iniciar Cámara"**
2. Permite el acceso a la cámara cuando el navegador lo solicite
3. Posiciónate frente a la cámara con **cuerpo completo visible**
4. Realiza las actividades y observa la predicción en tiempo real
5. Ajusta configuración en el panel lateral:
   - Umbral de confianza
   - Mostrar/ocultar esqueleto
   - Configuración de MediaPipe

### 3. Explorar Notebooks

**Notebook de Entrega 3 (completo):**

```bash
# Abrir en Jupyter
jupyter notebook Entrega3/notebook_entrega3_final.ipynb

# O abrir en Google Colab
# Subir el archivo a tu Drive y abrirlo con Colab
```

Contenido del notebook:
- Validación cruzada con GroupKFold
- Reducción de características (1,629 → 100)
- Entrenamiento del modelo final
- Evaluación y métricas
- Análisis de resultados

---

## Resultados Principales

### Métricas Globales

| Métrica | Valor |
|---------|-------|
| **Test Accuracy** | 80.4% |
| **F1-Score (weighted)** | 0.792 |
| **Cross-validation (GroupKFold)** | 80.3% ± 3.9% |
| **FPS en tiempo real** | 26 |
| **Tiempo de inferencia** | 3.2 ms/frame |
| **Features originales** | 1,629 |
| **Features seleccionadas** | 100 (reducción 93.9%) |

### Desempeño por Clase

| Actividad | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| caminar_frente | 0.895 | 0.888 | **0.891** | 998 |
| caminar_espalda | 0.829 | 0.959 | **0.889** | 726 |
| girar | 0.859 | 0.794 | **0.825** | 970 |
| sentado | 0.716 | 0.912 | **0.802** | 768 |
| ponerse_de_pie | 0.578 | 0.335 | **0.424** | 574 |

**Mejor clase:** caminar_frente (F1=0.891)  
**Peor clase:** ponerse_de_pie (F1=0.424)

### Confusiones Principales

- **47.7%** de "ponerse_de_pie" → clasificado como "sentado" (actividades de transición similares)
- **14.8%** de "girar" → clasificado como "caminar_espalda" (giro incluye pasos hacia atrás)

---

## Metodología

### Pipeline Completo

```
Video (30 FPS) 
    ↓
MediaPipe Pose (33 landmarks × 3D)
    ↓
Feature Engineering (ángulos, velocidades → 1,629 features)
    ↓
Feature Selection (RF Gini Importance → top 100)
    ↓
Random Forest (300 trees, max_depth=30)
    ↓
Predicción (5 clases)
```

### Técnicas Utilizadas

1. **Extracción de características:** MediaPipe Pose para 33 puntos corporales 3D
2. **Ingeniería de características:** 
   - Ángulos articulares (rodillas, codos, caderas)
   - Velocidades y aceleraciones
   - Proporciones corporales (ancho hombros, altura torso)
3. **Selección de features:** Random Forest Gini Importance (top 100)
4. **Validación rigurosa:** GroupKFold para prevenir data leakage temporal
5. **Modelo final:** Random Forest optimizado con GridSearchCV

### Dataset

- **Total:** 15,656 frames etiquetados
- **Videos:** 18 grabaciones de 3 sujetos
- **Duración:** 10-23 segundos por video
- **Condiciones:** Ambiente controlado, iluminación uniforme, fondo plano
- **Split:** 74% train (14 videos), 26% test (4 videos completamente no vistos)
- **Clases:** 5 actividades (balanceadas, rango 1,960-4,260 frames)

---

## Verificar Instalación

Ejecuta este script para verificar que todo funciona:

```python
# test_installation.py
import sys
print(f"Python version: {sys.version}")

try:
    import mediapipe as mp
    print(f"✓ MediaPipe: {mp.__version__}")
except ImportError as e:
    print(f"✗ MediaPipe: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")

try:
    import streamlit
    print(f"✓ Streamlit: {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Streamlit: {e}")

try:
    import joblib
    print(f"✓ joblib: {joblib.__version__}")
except ImportError as e:
    print(f"✗ joblib: {e}")

print("\n Todas las dependencias instaladas correctamente")
```

Ejecutar:
```bash
python test_installation.py
```

---

## Solución de Problemas Comunes

### Error: "No module named 'mediapipe'"

```bash
pip install "mediapipe==0.10.14" "protobuf==4.25.3"
```

### Error: MediaPipe no detecta landmarks

**Causas:**
- Cuerpo no completamente visible
- Iluminación insuficiente
- Cámara de baja calidad

**Soluciones:**
- Aléjate de la cámara (mínimo 2 metros)
- Asegura que cabeza, hombros, caderas, rodillas y pies estén visibles
- Mejora la iluminación del ambiente
- Ajusta `min_detection_confidence` en el panel lateral de Streamlit

### Error: "La cámara no se activa"

**Soluciones:**
- Verifica que no haya otra app usando la cámara (Zoom, Teams)
- Dale permisos de cámara al navegador
- Reinicia el navegador
- Prueba con otro navegador (Chrome recomendado)

### Streamlit muy lento

**Causas:** Overhead del navegador + actualización de widgets

**Soluciones:**
- Reduce FPS de captura (ajusta `time.sleep` en el código)
- Desactiva visualizaciones innecesarias (ángulos, probabilities)
- Usa la versión OpenCV standalone (más rápida, sin interfaz web)

### Protobuf version conflict

Si ves error: `TypeError: Descriptors cannot not be created directly`

```bash
pip uninstall protobuf
pip install "protobuf==4.25.3"
```

---

## Requisitos del Sistema

### Hardware Mínimo

- **CPU:** Intel i5 o equivalente (2+ cores)
- **RAM:** 4 GB mínimo (8 GB recomendado)
- **Webcam:** Cualquier cámara USB o integrada (720p mínimo)
- **Almacenamiento:** 500 MB libres (para dependencias + modelo)


---

## Trabajo Futuro

### Mejoras a Corto Plazo

- Aumentar dataset a 50+ sujetos en ambientes diversos
- Implementar suavizado temporal (LSTM/GRU) para transiciones
- Balancear clases (más ejemplos de "ponerse_de_pie")
- Agregar data augmentation (flip horizontal, time warping)

### Extensiones a Largo Plazo

- Multi-persona tracking simultáneo
- Detección de caídas y anomalías (para adultos mayores)
- Deployment en dispositivos móviles (TensorFlow Lite)
- Más actividades (correr, saltar, agacharse, caída)
- Sistema de feedback para corrección postural en tiempo real

---

## Tecnologías Utilizadas

- **Python 3.10:** Lenguaje principal
- **MediaPipe 0.10.14:** Detección de pose en tiempo real
- **scikit-learn 1.3:** Machine learning (Random Forest)
- **OpenCV 4.12:** Procesamiento de video
- **Streamlit 1.28:** Interfaz web interactiva
- **Pandas/NumPy:** Manipulación de datos
- **Matplotlib/Seaborn:** Visualización

---

## Referencias

1. Bazarevsky, V., et al. "BlazePose: On-device real-time body pose tracking." *arXiv:2006.10204*, 2020.
2. Breiman, L. "Random forests." *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.
3. Chapman, P., et al. "CRISP-DM 1.0: Step-by-step data mining guide." *SPSS Inc.*, 2000.
4. Lugaresi, C., et al. "MediaPipe: A framework for building perception pipelines." *arXiv:1906.08172*, 2019.

---

## Licencia

Este proyecto fue desarrollado con fines académicos para el curso Algoritmos y Programación III de la Universidad ICESI.


---

## Agradecimientos

- Profesor Milton Sarria por la guía durante el proyecto
- Universidad ICESI por los recursos computacionales
- Google por MediaPipe open-source
- Compañeros de clase por feedback y pruebas del sistema

---

