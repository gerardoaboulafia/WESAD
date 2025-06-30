# Selección de los datos a utilizar
Los datos seleccionados para el entrenamiento y armado del modelo fueron únicamente los obtenidos con el RespiBAN (datos del pecho). Esto se debe a que estos suelen presentar mayor precisión para la detección de emociones como el estrés, en comparación con los datos obtenidos desde la muñeca.
Esta elección también se fundamenta en que, al revisar los archivos README individuales de los participantes del dataset WESAD, se identificaron fallas recurrentes en los sensores de muñeca (como colocación incorrecta o mal funcionamiento), lo que podría comprometer la calidad y consistencia de los datos.

# Definición de sliding window para la segmentación de los datos.
Se decidió trabajar con un sliding window de 60 segundos, la cual contiene un total de 42000 muestras por ventana (ya que la frecuencia de muestreo es de 700 HZ)  con un window shift de ¼ de segundo (la ventana se va desplazando cada 175 observaciones). Esta decisión se debe a que al utilizar ventanas que abarquen una gran cantidad de observaciones de la señal, es posible obtener features estables y representativas de variables tanto cardiovasculares como respiratorias.
Kreibig (2010) describe que las respuestas a las diferentes emociones pueden ser mejor capturadas al analizar amplios segmentos de la señal, precisamente la medida estándar utilizada es de 60 segundos. 

# Decisión de trabajar con dos estados de ánimo
La selección de únicamente dos clases (baseline y estrés) responde a una necesidad de negocio: el cliente requiere saber con precisión cuándo un paciente se encuentra estresado. En ese contexto, incorporar otras emociones (por ejemplo, felicidad o diversión) habría añadido complejidad sin aportar valor directo al objetivo del proyecto. 
Por ello se descartó la idea inicial de entrenar un modelo multiclase y se optó por una formulación binaria que emula el caso de uso real: indicar, si la persona está o no bajo estrés.


# Elección de modelo XG Boost
Se eligió **XGBoost** porque, aun ofreciendo la misma accuracy promedio que el Random Forest (≈ 72,5 %), mostró una variabilidad menor entre sujetos: la desviación estándar en la validación LOSO bajó de ± 0,221 a ± 0,207. Esa reducción de dispersión implica que el modelo se comporta de forma más homogénea paciente a paciente, un requisito clave para un producto que debe funcionar con usuarios nunca vistos.

Además, XGBoost aporta ventajas técnicas alineadas con el caso de uso: maneja bien datos heterogéneos, y ofrece métricas de importancia de variables que facilitan la interpretación clínica. En definitiva, combina robustez predictiva y consistencia con la flexibilidad necesaria para futuras iteraciones, lo que lo convierte en la opción más fiable para detectar episodios de estrés en tiempo real.
