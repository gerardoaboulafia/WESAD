# WESAD
Este repositorio contiene toda la información relacionada al proyecto de detección de estrés, que se llevó a cabo con los datos de WESAD.

Este proyecto tiene como objetivo analizar datos fisiológicos multicanal provenientes del dataset WESAD (Wearable Stress and Affect Detection) con el fin de desarrollar modelos de detección de estrés utilizando señales de sensores portables. 
El objetivo principal es distinguir entre personas en estado de estrés y no estrés, mediante el entrenamiento de modelos que puedan clasificar estos estados de manera confiable. Este desarrollo se proyecta como una solución potencialmente aplicable en el contexto de dispositivos inteligentes, orientada a empresas que diseñan sensores portátiles enfocados en el monitoreo del bienestar y la salud mental.

## Dataset: WESAD
El conjunto de datos contiene datos fisiológicos obtenidos de 15 sujetos (17 originales menos 2 descartados) durante diferentes condiciones emocionales (baseline, estrés, diversión, meditación). Las señales fueron registradas mediante dos dispositivos:

**RespiBAN** (sensor del pecho), este incluye: el electrocardiograma (ECG), la actividad electrodermal (EDA), la electromiografía (EMG), temperatura (°C), respiración y acelerómetro (para los tres ejes X, Y, Z). En este, todas las señales están sampleadas a 700 Hz.

**Empatica E4** (sensor de la muñeca), este incluye: EDA sampleado a 4 Hz, temperatura sampleado a 4 Hz, BVP  sampleado a 64 Hz y acelerómetro para los tres ejes sampleado a 32 Hz. Este sensor fue colocado en la mano no dominante del paciente. 

Donde X es el número de paciente, cada paciente posee los siguientes archivos:
SX_readme.txt: Contiene información del paciente X, información sobre la recolección de los datos y su calidad. Por ejemplo, en un paciente puede aclararse que el sensor Empatica E4 estuvo mal colocado durante el estudio por lo que sus señales no serían válidas (como fue el caso de los pacientes 1 y 12). 
**SX_respiban.txt:** Datos crudos del RespiBAN. 

**SX_E4_Data.zip:** Datos crudos del Empatica E4.


**SX.pkl:** Datos sincronizados con etiquetas de condición.


**SX_quest.csv:** Datos de protocolos y cuestionarios, estos son un autoinforme completado por cada paciente. Estas incluyen 3 encuestas: SSSQ, STAI y PANAS, cada una con sus respectivas variables. 


*SSSQ:* 'Participant', 'I was committed to attaining my performance goals',  'I wanted to succeed on the task', 'I was motivated to do the task', 'I reflected about myself', 'I was worried about what other people think of me', 'I felt concerned about the impression I was making'.


*STAI:* 'Participant', 'Condition', 'Ease', 'Nervous', 'Jittery', 'Relaxed', 'Worried', 'Pleasant'.


*PANAS:* 'Participant', 'Condition', 'Active', 'Distressed', 'Interested',  'Inspired', 'Annoyed', 'Strong', 'Guilty', 'Scared', 'Hostile', 'Excited', 'Proud', 'Irritable', 'Enthusiastic', 'Ashamed', 'Alert',  'Nervous', 'Nervous.1', 'Attentive', 'Jittery', 'Afraid', 'Stressed', 'Frustrated', 'Happy', 'Sad', 'Angry', 'Irritated'. 

Es importante destacar que, para este estudio, se decidió trabajar exclusivamente con los datos registrados por el dispositivo RespiBAN, ya que estos suelen presentar mayor precisión para la detección de emociones como el estrés, en comparación con los datos obtenidos desde la muñeca. 
Esta elección también se fundamenta en que, al revisar los archivos README individuales de los participantes del dataset WESAD, se identificaron fallas recurrentes en los sensores de muñeca (como colocación incorrecta o mal funcionamiento), lo que podría comprometer la calidad y consistencia de los datos.



## Estructura de datos
Como ya fue establecido, los datos de sensores brutos fueron recolectados mediante dos dispositivos: un sensor de pecho (RespiBAN) y uno de muñeca (Empatica E4). Las etiquetas del protocolo experimental están sincronizadas directamente con los datos del RespiBAN, mientras que los datos del Empatica E4 requieren una sincronización manual. 
Se incluyen, además, archivos preprocesados por sujeto en formato SX.pkl, que contienen los datos de sensores ya sincronizados junto con sus etiquetas. Cada uno de estos archivos es un diccionario con tres claves principales:


**subject:** ID del sujeto participante.


**signal:** datos crudos organizados por ubicación del sensor (chest y wrist).


**label:** etiquetas que indican la condición experimental en cada momento (0: indefinido, 1: baseline, 2: estrés, 3: diversión, 4: meditación).



## Procesamiento de los datos
El procesamiento fue realizado en Python. Se utilizó el “pkl” y se extrajeron las señales fisiológicas desde el diccionario, organizadas por sensor y ubicación. Luego para poder utilizar los datos y tener una línea temporal se segmentan con una ventana deslizante. Se aplicó una técnica de Sliding window para dividir las señales fisiológicas en segmentos de tiempo más cortos, lo cual facilita la extracción de características estadísticas locales y mejora el entrenamiento de modelos supervisados. Los parámetros utilizados fueron: 

**Tamaño de ventana:** 60 segundos (ventanas de 42000 muestras).


**Desplazamiento (window shift):** 0.25 de segundos (ventanas de 175 muestras). 



En este proyecto se busca poder distinguir particularmente el estado de estrés, por lo que se  utilizó únicamente el label 1 y  label 2, es decir baseline y stress respectivamente. 

## Modelo Aplicado
### Validación Leave-One-Subject-Out
Para evaluar la capacidad de generalización del modelo a nuevos individuos, se utilizó el método de validación cruzada Leave-One-Subject-Out (LOSO). Este enfoque consiste en:

Entrenar el modelo dejando completamente afuera un sujeto (paciente) en cada iteración.  
Evaluar el rendimiento del modelo sobre el sujeto excluido.  
Repetir el procedimiento para todos los sujetos del conjunto de datos.
Para la etapa de **modelado** se evaluaron dos clasificadores supervisados bajo el mismo esquema de validación **Leave-One-Subject-Out (LOSO)**: primero un **Random Forest** como línea base y luego un **XGBoost** para contrastar desempeño.

### 1. Random Forest Classifier

|          | pred_0 | pred_1 |
|----------|--------|--------|
| **real_0** | 49 634 | 16 119 |
| **real_1** | 12 266 | 24 918 |

- **Accuracy LOSO media:** **0.725 ± 0.221**

El Random Forest confirmó la **viabilidad del proyecto**, alcanzando un 72 % de acierto global. Sin embargo, la desviación estándar relativamente alta (± 0.221) indica **gran variabilidad entre sujetos**: en algunos pacientes el modelo rinde muy bien, en otros no tanto.

---

### 2. XGBoost Classifier

|          | pred_0 | pred_1 |
|----------|--------|--------|
| **real_0** | 50 200 | 15 553 |
| **real_1** | 12 721 | 24 463 |

- **Accuracy LOSO media:** **0.725 ± 0.207**

Se entrenó un modelo **XGBoost** (200 árboles, `learning_rate=0.1`, `max_depth=6`) con los mismos pliegues LOSO.  
Aunque la media de *accuracy* **es prácticamente idéntica** a la del Random Forest, la **dispersión entre pacientes es menor** (± 0.207), lo que sugiere un rendimiento más estable cuando el modelo se aplica a individuos no vistos.

---

### Selección del modelo

En un escenario de negocio donde el producto debe **funcionar con pacientes que no formaron parte del entrenamiento**, la **consistencia** es tan crítica como la precisión media.  
Por ello se **elige el modelo de XGBoost**:

- Ofrece la **misma tasa de acierto global** que el Random Forest.  
- Presenta **menor variabilidad entre sujetos**, reduciendo el riesgo de fallos graves en casos individuales.  
- Permite ajustes finos de regularización y *shrinkage* que facilitan la **interpretación de importancia de variables** para futuras mejoras.

En síntesis, XGBoost proporciona un **compromiso equilibrado** entre desempeño y robustez, adecuado para la implementación en dispositivos de monitoreo del estrés en usuarios finales.
  

## Créditos y referencias
Dataset creado por Philip Schmidt et al. (2018)  
Referencia: [WESAD: A Multimodal Dataset for Wearable Stress and Affect]

https://machinelearningmastery.com/loocv-for-evaluating-machine-learning-algorithms/#:~:text=fold%20Cross%2DValidation-,Leave%2Done%2Dout%20cross%2Dvalidation%2C%20or%20LOOCV%2C,or%20costly%20models%20to%20fit. 

