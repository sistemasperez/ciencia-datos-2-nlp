# INF-8239 · Unidad 02 · Proyecto de Procesamiento de Lenguaje Natural (PLN)

> **Autor académico del diseño base:** Edwin Ramón José Nolasco  
> **Adaptación del laboratorio:** Clasificación de Sentimientos y Análisis Semántico sobre *Twitter US Airline Sentiment*

---

## 1. Descripción del Proyecto y Dataset

Este proyecto implementa un flujo reproducible de Procesamiento de Lenguaje Natural (PLN) orientado a clasificar opiniones de pasajeros sobre aerolíneas comerciales en tres categorías de sentimiento: `negative`, `neutral` y `positive`.

* **Dataset adoptado:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (CrowdFlower / Kaggle).
* **Ficha Técnica Completa:** Consulte [docs/DATASET_CARD.md](docs/DATASET_CARD.md) para el diccionario de variables, licencias, calidad de datos y consideraciones éticas.
* **Dictamen del Estudiante:** Consulte [docs/DICTAMEN_DATASET.md](docs/DICTAMEN_DATASET.md) para la evaluación formal, justificación de selección frente a candidatos y riesgos identificados.
* **Resumen de Datos Auditados (`scripts/audit_data.py`):**
  * **Registros totales:** 14,640 filas y 15 columnas.
  * **Integridad (SHA-256):** `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`.
  * **Nulos en variables clave:** 0 en `text`, 0 en `airline_sentiment`.
  * **Duplicados:** 213 textos eliminados en preprocesamiento (14,427 tweets únicos).
  * **Distribución de clases (Desbalance):** Negativo: 62.7%, Neutral: 21.2%, Positivo: 16.1%.

---

## 2. Configuración del Entorno y Ejecución Rápida

El proyecto utiliza [`uv`](https://docs.astral.sh/uv/) para gestionar dependencias y entornos virtuales con Python 3.12.

### A. Preparación del archivo `.env`
El proyecto sigue el estándar de configuración desacoplada. Asegúrese de que el archivo `.env` en la raíz contenga:

```env
DATA_SOURCE=local
DATASET_PATH=data/raw/Tweets.csv
DATASET_URL=https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
TEXT_COLUMN=text
TARGET_COLUMN=airline_sentiment
RANDOM_STATE=42
```

### B. Comandos de Verificación Inicial

```bash
# 1. Instalar Python 3.12 y sincronizar dependencias
uv python install 3.12
uv sync

# 2. Ejecutar batería de pruebas unitarias (6 tests)
uv run pytest -q

# 3. Auditar contrato de datos e integridad
uv run python scripts/audit_data.py
```

---

## 3. Entrenamiento y Resultados Experimentales

Para entrenar los modelos de clasificación, evaluar métricas y generar artefactos visuales:

```bash
uv run python scripts/train_text.py
```

### A. Comparativa de Modelos (Evaluación en Test Set 25% Estratificado)

| Modelo | F1-Macro | Accuracy | Precisión Ponderada | Recall Ponderado | Rol Experimental |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DummyClassifier** | **0.258** | 0.63 | 0.40 | 0.63 | **Control / Línea Base Ingenua:** Asigna siempre la clase mayoritaria (`negative`). Demuestra por qué el *Accuracy* es engañoso en clases desbalanceadas. |
| **Naive Bayes (ComplementNB)** | **0.550** | 0.72 | 0.76 | 0.72 | **Modelo Probabilístico:** Optimizado para texto desbalanceado. Excelente precisión en positivos y neutrales, pero bajo recall en neutros. |
| **Logistic Regression (balanced)** | **0.741** | **0.79** | **0.80** | **0.79** | **Modelo Seleccionado:** Vectorización TF-IDF (unigramas y bigramas) con ponderación balanceada de clases. Logra el mejor equilibrio general. |

### B. Matriz de Confusión (Modelo Seleccionado: Regresión Logística)

La siguiente matriz evidencia la capacidad del modelo para distinguir las tres clases y revela que la mayor fricción se presenta entre opiniones neutrales y negativas:

![Matriz de Confusión](reports/confusion_text.png)

### C. Análisis Cualitativo de 20 Errores del Modelo

El archivo [reports/error_analysis.csv](reports/error_analysis.csv) contiene los 764 casos de prueba donde el clasificador de Regresión Logística discrepó de la etiqueta asignada por humanos. Para dar cumplimiento a la rúbrica analítica de evaluación de errores, se auditó cualitativamente una muestra representativa de **20 fallos sistemáticos**:

| # | Texto del Tweet de Prueba | Real | Predicho | Taxonomía del Error | Justificación Técnica / Lingüística del Fallo |
|---|---|:---:|:---:|---|---|
| **1** | `@united I forgot that Intl flights out of LAX don't go from Intl Terminal! Easiest re-check in ever! woo!` | `positive` | `neutral` | **Exclamación Informal / Jerga** | El sentimiento positivo se apoya en *"woo!"* y *"Easiest re-check in"*. El modelo ponderó el vocabulario aeroportuario estándar (*"LAX"*, *"flights"*, *"terminal"*) y predijo neutral. |
| **2** | `@SouthwestAir Customer Centricity is knowing people #ANAMarketers` | `positive` | `negative` | **Jerga Corporativa Abstracta** | Lenguaje de mercadeo institucional sin léxico afectivo explícito. Debido al fuerte desbalance hacia quejas, el modelo lo categorizó como negativo por defecto. |
| **3** | `"@SouthwestAir That was Chantilly, Paris, France."` | `neutral` | `positive` | **Dato Geográfico sin Polaridad** | Respuesta puramente informativa. Términos turísticos como *"Paris"* y *"France"* poseen pesos positivos residuales por coocurrencia en el corpus. |
| **4** | `@united Thanks! Good to know.` | `neutral` | `positive` | **Cortesía Protocolar** | Simple acuse de recibo conversacional. Las palabras *"Thanks"* y *"Good"* tienen coeficientes altamente positivos en TF-IDF, superando el contexto neutro. |
| **5** | `@JetBlue what are the chances of one of your flights leaving at 6am out of Newark nj being Cancelled Flightled tomorrow?` | `neutral` | `negative` | **Pregunta Operativa con Término Crítico** | Consulta logística preventiva de un usuario. El n-grama *"Cancelled Flightled"* tiene un peso fuertemente negativo que arrastra toda la predicción hacia queja. |
| **6** | `@SouthwestAir thanks for the response - flight time has passed - rebooked now` | `negative` | `positive` | **Cortesía en Reclamo** | El usuario agradece la respuesta tras perder su vuelo original. El clasificador prioriza los términos *"thanks"* y *"rebooked"* sobre la incidencia negativa de fondo. |
| **7** | `@united Did somebody say flight upgrade?` | `negative` | `neutral` | **Ironía / Expectativa** | Pregunta retórica quejosa sobre no recibir una mejora de asiento. Al no incluir insultos ni adjetivos negativos clásicos, se clasifica como neutra. |
| **8** | `@JetBlue I shouldn't be made to feel like a criminal because I want to sit near my 5 year old on a flight.` | `negative` | `neutral` | **Metáfora Emocional Compleja** | Queja profunda estructurada mediante una analogía (*"feel like a criminal"*). TF-IDF carece de contexto semántico para vincular la frase con una falla de servicio al cliente. |
| **9** | `"@united i understand, thanks"` | `neutral` | `positive` | **Acuse de Recibo Ambiguo** | Aceptación resignada de un pasajero. El token *"thanks"* sesga la decisión hacia clase positiva. |
| **10** | `@united You delayed a connection for customer service today now I'll be able to pay my final respects to a dear family member. Thank you :)` | `positive` | `negative` | **Término Negativo en Elogio** | Pasajero conmovido porque demoraron una conexión para permitirle asistir a un funeral. Los tokens *"delayed"* y *"connection"* forzaron la predicción hacia negativo. |
| **11** | `@USAirways and @AmericanAir aren't as merged as the marketing departments would have us believe. Maybe @Delta or @united can get me home?` | `neutral` | `negative` | **Crítica Sutil / Pregunta** | Comentario escéptico etiquetado como neutral por humanos, pero clasificado como reclamo por la mención a aerolíneas competidoras y rescate de viaje (*"get me home"*). |
| **12** | `@SouthwestAir can i get tiks for #DestinationDragons omg id love to! Plz 😱❤️im dying` | `positive` | `neutral` | **Jerga Informal, Emojis y Exageración** | Pasajera entusiasmada pidiendo boletos de concierto. TF-IDF no captura emojis (😱❤️) ni modismos de entusiasmo (*"omg id love to"*, *"im dying"*). |
| **13** | `"@SouthwestAir yes, thank you. Just sent DM."` | `neutral` | `positive` | **Cortesía de Trámite** | Mensaje operativo de seguimiento de soporte. La presencia de *"thank you"* activa la clase positiva en lugar de neutral. |
| **14** | `@SouthwestAir it's ok...I forgive you guys.` | `negative` | `positive` | **Perdón tras Incidencia** | El pasajero acepta una disculpa (*"I forgive you"*). Aunque implica un problema previo (reclamo), el modelo interpreta *"forgive"* y *"ok"* como polaridad positiva. |
| **15** | `@united the os isn't controlled by me but rather @VerizonWireless . App is new.` | `neutral` | `negative` | **Aclaración Técnica Operativa** | Explicación sobre una falla técnica móvil. El modelo lineal asocia problemas de la app a reclamos de usuarios. |
| **16** | `@SouthwestAir Just picked up my bag at SEATAC all the clothes are soaking wet... #bagawim` | `negative` | `positive` | **Sentimiento Implícito (Sin Adjetivos)** | Reclamo severo (ropa mojada en el equipaje) redactado de forma factual sin insultos ni adjetivos como *"awful"* o *"terrible"*. TF-IDF falla al no encontrar palabras de queja directa. |
| **17** | `"@united we had a wonderful flight attendant named Leah that was with us from lga to den, then den to anchorage!"` | `positive` | `negative` | **Coocurrencia de Aeropuertos en Quejas** | Elogio explícito (*"wonderful"*), pero la acumulación de escalas (*"lga"*, *"den"*, *"anchorage"*) suele coocurrir en vuelos con demoras en el corpus de entrenamiento. |
| **18** | `"😢😢 RT“@JetBlue: @Peaches275 Oh no! Unfortunately, the code must be entered at the time of purchase in order to qualify for the points.”"` | `neutral` | `negative` | **Retweet Informativo con Emojis** | Retweet de una aclaración oficial. Palabras como *"Unfortunately"* y emojis de tristeza sesgan la clasificación hacia queja. |
| **19** | `@JetBlue weather where? It's 60 in SFO and flight 834 is still on time to boston.` | `neutral` | `negative` | **Pregunta Retórica Desafiante** | Duda sobre reportes de clima. El clasificador asocia preguntas sobre clima a justificaciones de demoras y quejas. |
| **20** | `@USAirways Thanks again for ruiningmy vacation. You all are the best for this. Could not have done it without you.` | `negative` | `positive` | **Sarcasmo Puro / Ironía** | Caso paradigmático: uso de léxico hiperbólico positivo (*"Thanks"*, *"the best"*, *"could not have done it without you"*) para expresar furia. El modelo lineal es ciego a la inversión semántica del sarcasmo. |

#### Distribución Taxonómica y Lecciones Aprendidas:
1. **Sarcasmo y Cortesía en Reclamos (40% de los fallos):** Palabras como *"Thanks"*, *"Good"* o *"Best"* enmascaran quejas severas cuando el modelo opera bajo una suma ponderada de términos individuales o bigramas simples.
2. **Consultas Operativas Neutrales con Léxico Crítico (35% de los fallos):** Preguntar si un vuelo está cancelado o demorado no constituye una queja *per se*, pero activa los n-gramas más pesados de la clase negativa.
3. **Sentimiento Implícito y Metáforas (25% de los fallos):** Las fallas reales narradas objetivamente (ropa mojada, asientos separados de niños) no utilizan adjetivos afectivos tradicionales y requieren comprensión del sentido común.
4. **Propuesta de Mitigación:** Para producción, se recomienda incorporar un modelo preentrenado con atención bidireccional contextualizada (ej. `RoBERTa-base` o `Twitter-XLM-RoBERTa`), complementado con calibración de umbrales específicos para la clase neutral mediante curvas Precision-Recall.


---

## 4. Análisis Semántico (Word2Vec) y Redes de Grafos

El script `scripts/embeddings_network.py` entrena un modelo de representación densa (*Word Embeddings*) sobre los tweets y genera métricas de centralidad y comunidades:

```bash
uv run python scripts/embeddings_network.py --word flight --network-demo
```

### A. Semántica Aprendida (Vecinos más cercanos a `"flight"`):
* **Vocabulario:** 15,051 términos únicos.
* **Términos más similares:** `flt` (similitud 0.640), `connection` (0.634), `flights` (0.602), `departure` (0.541), `ticket` (0.540), `trip` (0.519), `rebooked` (0.495).  
* **Conclusión semántica:** El espacio vectorial capturó con alta fidelidad las relaciones contextuales y jerga aeroportuaria sin intervención manual de sinónimos.

### B. Grafo de Comunidades Estructurales

El análisis estructural de modularidad identificó 3 comunidades de alta densidad (modularidad = 0.411):

![Red de Comunidades](reports/network.png)

Las métricas de centralidad (Grado, Betweenness, PageRank) se encuentran documentadas en [reports/centralities.csv](reports/centralities.csv).

---

## 5. Demostración Interactiva con Streamlit

El modelo seleccionado y serializado en [models/text_model.joblib](models/text_model.joblib) puede ejecutarse interactivamente mediante la interfaz web:

```bash
uv run streamlit run app/streamlit_app.py
```

### Casos de Prueba Sugeridos para la Demostración

Copie y pegue estos ejemplos en el área de texto de la aplicación para validar el comportamiento del clasificador:

| # | Texto del Tweet de Prueba (Inglés) | Traducción al Español | Clase Esperada | Justificación Lingüística |
|---|---|---|:---:|---|
| **1** | `"Our flight was cancelled with zero notice and customer service was extremely rude! Never flying with you again."` | *Nuestro vuelo fue cancelado sin aviso y el servicio al cliente fue sumamente grosero. Nunca volveré a volar con ustedes.* | **`negative`** | Fuertes n-gramas negativos: *"cancelled"*, *"zero notice"*, *"extremely rude"*. |
| **2** | `"Incredible flight! The flight attendants were super friendly and we landed 25 minutes ahead of schedule. Thank you!"` | *¡Vuelo increíble! Los asistentes de vuelo fueron súper amables y aterrizamos 25 minutos antes de lo previsto. ¡Gracias!* | **`positive`** | Polaridad positiva explícita: *"incredible"*, *"super friendly"*, *"thank you"*. |
| **3** | `"Flight 402 from Boston to Dallas is scheduled to depart at gate B12 tomorrow at 8:30 AM."` | *El vuelo 402 de Boston a Dallas está programado para salir en la puerta B12 mañana a las 8:30 AM.* | **`neutral`** | Lenguaje factual y operativo sin carga emocional subjetiva. |

---

## 6. Marco de Interpretación de Resultados

En conformidad con las directrices académicas de la asignatura:

1. **Observación:** La Regresión Logística con TF-IDF alcanzó un F1-macro de **0.741** y un Accuracy global de **79%**, superando holgadamente a la línea base ingenua (F1 = 0.258) y al clasificador Naive Bayes (F1 = 0.550).
2. **Evidencia:** La matriz de confusión demuestra una sensibilidad del 86% para la clase negativa (2,270 muestras de prueba) y del 74% para la positiva, con un F1 de 0.62 en la clase neutral.
3. **Interpretación:** El desbalance del 62.7% en contra de quejas no impidió un rendimiento balanceado gracias a la ponderación por clases inversas (`class_weight="balanced"`). No obstante, los errores en tweets neutrales reflejan la dificultad inherente de TF-IDF para captar sutilezas contextuales y preguntas operativas sin marcadores afectivos explícitos.
4. **Decisión:** Para un despliegue operativo en monitoreo de atención al cliente, se adopta el modelo de Regresión Logística como filtro de triaje automático para enrutar quejas urgentes a agentes humanos, manteniendo supervisión manual sobre casos neutrales o limítrofes.
