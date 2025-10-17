# Entorno recomendado para ejecutar el proyecto


**Python**

Python 3.10 (recomendado) o Python 3.11.

    ⚠️ MediaPipe 0.10.x no es compatible con Python 3.13 y suele dar problemas con 3.12


**Paquetes principales**

- mediapipe==0.10.14

- protobuf==4.25.3 (obligatorio; versiones 5.x rompen MediaPipe)

- opencv-python==4.12.0.0 (o 4.10.x si prefieres)

- numpy>=1.23,<3

- pandas>=1.5

**(Opcionales pero útiles)**

- matplotlib>=3.7 (para gráficos de EDA si los haces en Python)

- pyarrow>=14 (si decides exportar también a Parquet)


**Instalación (ejemplo reproducible)**

Con Conda (recomendado):

 ```
conda create -n ia310 python=3.10 -y
conda activate ia310
python -m pip install --upgrade pip setuptools wheel
python -m pip install "mediapipe==0.10.14" "protobuf==4.25.3" "opencv-python==4.12.0.0" numpy pandas
# opcionales:
# python -m pip install matplotlib pyarrow

 ```


 