# Clasificación de categorías de deuda de sueño con MLflow

## 1. Descripción del proyecto

Este proyecto implementa un modelo de clasificación basado en regresión logística multinomial para predecir categorías de deuda de sueño a partir de variables relacionadas con los hábitos de uso del teléfono, el descanso y el estilo de vida.

El proyecto utiliza Python y Scikit-learn para el preprocesamiento, entrenamiento y evaluación del modelo, y MLflow para registrar los hiperparámetros, las métricas de evaluación, los artefactos y el modelo entrenado.

## 2. Requisitos previos

Antes de ejecutar el proyecto, es necesario contar con los siguientes componentes:

- Python instalado.
- pip para administrar las dependencias.
- Una cuenta de Kaggle y acceso al conjunto de datos utilizado por el proyecto.
- MLflow para el seguimiento de experimentos.
- Una terminal de comandos, como PowerShell o la terminal integrada de VS Code.

El conjunto de datos utilizado es [Sleep Debt and Screen Time: Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits).

## 3. Instalación

### 3.1. Clonar o descargar el proyecto

Descarga el repositorio o coloca los archivos del proyecto en un directorio local.

La estructura esperada es:

```text
proyecto/
├── train.py
├── requirements.txt
└── README.md
```

### 3.2. Crear un entorno virtual

Desde la terminal, accede al directorio del proyecto y ejecuta:

En Windows:

```powershell
python -m venv .venv
```

Activa el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si utilizas el símbolo del sistema (CMD), ejecuta:

```cmd
.venv\Scripts\activate.bat
```

### 3.3. Instalar las dependencias

Con el entorno virtual activado, ejecuta:

```powershell
python -m pip install --upgrade pip
```

Después, instala las dependencias registradas en el archivo `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Se recomienda utilizar las versiones registradas en dicho archivo para mantener la consistencia del entorno de ejecución.

## 4. Configuración de Kaggle

El script utiliza KaggleHub para descargar y cargar el conjunto de datos. Es posible que sea necesario configurar las credenciales de Kaggle para acceder al dataset.

Consulta la documentación oficial de [KaggleHub](https://github.com/Kaggle/kagglehub) para configurar la autenticación según tu entorno.

El script utiliza el siguiente dataset y archivo:

- Dataset: `samartalwar/sleep-debt-and-screen-time-late-night-phone-habits`
- Archivo: `bedtime_screentime_sleep_debt.csv`

## 5. Ejecutar el entrenamiento

### 5.1. Iniciar el servidor de MLflow

Abre una terminal con el entorno virtual activado y ejecuta:

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

Mantén esta terminal abierta mientras ejecutas el entrenamiento.

### 5.2. Ejecutar el script

Abre una segunda terminal, activa el mismo entorno virtual y accede al directorio del proyecto.

Ejecuta el script con los valores por defecto:

```powershell
python train.py
```

Los parámetros predeterminados son:

| Parámetro | Valor por defecto | Descripción |
|---|---:|---|
| `--max-iter` | 1000 | Número máximo de iteraciones del modelo |
| `--C` | 1.0 | Inverso de la fuerza de regularización |
| `--test-size` | 0.30 | Proporción de datos para pruebas |
| `--seed` | 7 | Semilla aleatoria para la partición de los datos |

### 5.3. Ejecutar con parámetros personalizados

Es posible modificar los parámetros del modelo desde la línea de comandos sin editar el código.

Por ejemplo:

```powershell
python train.py --max-iter 2000 --C 0.5 --test-size 0.25 --seed 42
```

Este comando configura el entrenamiento con 2000 iteraciones máximas, un valor de regularización `C=0.5`, un 25 % de los datos para pruebas y la semilla 42.

Para consultar todos los argumentos disponibles:

```powershell
python train.py --help
```

## 6. Seguimiento de experimentos con MLflow

El script registra las ejecuciones en el experimento `Clasificacion_Estado_Sueno`.

Durante cada ejecución se registran:

- **Parámetros:** número máximo de iteraciones, valor de regularización, proporción de prueba, semilla y configuración del modelo.
- **Métricas:** exactitud (`Accuracy`) y F1 ponderado (`f1_weighted`).
- **Artefactos:** matriz de confusión generada durante la evaluación.
- **Modelo:** pipeline entrenado con escalado y regresión logística.

### 6.1. Consultar los resultados

Con el servidor de MLflow en ejecución, abre el siguiente enlace en el navegador:

http://localhost:5000

Selecciona el experimento `Clasificacion_Estado_Sueno` para consultar las corridas registradas, comparar sus parámetros y métricas y revisar los artefactos de cada ejecución.

### 6.2. Comparar distintas corridas

Para comparar configuraciones, ejecuta el script varias veces con diferentes hiperparámetros. Por ejemplo:

```powershell
python train.py --max-iter 1000 --C 1.0 --test-size 0.30 --seed 7
python train.py --max-iter 2000 --C 1.0 --test-size 0.30 --seed 7
python train.py --max-iter 1000 --C 0.1 --test-size 0.30 --seed 7
python train.py --max-iter 1000 --C 10.0 --test-size 0.30 --seed 7
python train.py --max-iter 2000 --C 0.5 --test-size 0.30 --seed 7
```

En la interfaz de MLflow, selecciona las corridas y utiliza la opción de comparación para contrastar los resultados.

## 7. Reproducibilidad

Para facilitar la reproducción de los experimentos:

1. Utiliza el archivo `requirements.txt` para instalar las versiones registradas de las dependencias.
2. Mantén los mismos argumentos y la misma semilla aleatoria.
3. Utiliza la misma versión del conjunto de datos.
4. Conserva el mismo código de preprocesamiento y entrenamiento.
5. Compara las métricas y predicciones de las corridas en MLflow.

La misma semilla no garantiza por sí sola la reproducibilidad si cambian los datos, las versiones de las bibliotecas o las condiciones de ejecución.

## 8. Solución de problemas

**Python no encuentra `train.py`.**

Verifica que la terminal esté situada en el directorio donde se encuentra el archivo:

```powershell
dir
```

**No se puede conectar con MLflow.**

Comprueba que el servidor esté activo en `http://localhost:5000` y que el script utilice esa misma dirección como URI de seguimiento.

**Error al descargar el conjunto de datos.**

Verifica la conexión a Internet, el identificador del dataset y la configuración de las credenciales de Kaggle.

**Error al instalar las dependencias.**

Comprueba que el entorno virtual esté activado y que la versión de Python sea compatible con las dependencias registradas.

## 9. Tecnologías utilizadas

- Python
- Pandas y NumPy
- Matplotlib
- Scikit-learn
- KaggleHub
- MLflow