# Dictamen de Evaluación y Selección del Dataset

**Asignatura:** INF-8239 · Ciencia de Datos II  
**Unidad:** Unidad 02 · Procesamiento de Lenguaje Natural (PLN)  
**Autor (Estudiante):** Aníbal Pérez  
**Fecha:** Octubre de 2026  
**Dataset Evaluado:** Twitter US Airline Sentiment  
**Archivo Auditado:** `data/raw/Tweets.csv`  

---

## Redacción del Estudiante

### 1. Resultado principal
La auditoría técnica automatizada ejecutada mediante `scripts/audit_data.py` confirmó que el dataset cumple satisfactoriamente con el contrato de datos del proyecto (`validate_dataframe`), sin requerir transformaciones estructurales forzadas. 
* **Volumen:** 14,640 registros brutos y 15 atributos.
* **Integridad referencial:** 0 valores nulos en el campo de texto (`text`) y 0 en la variable objetivo (`airline_sentiment`).
* **Deduplicación:** Se identificaron 213 textos repetidos (1.45% del total), correspondientes a retweets o mensajes de difusión masiva, los cuales fueron purgados exitosamente por el pipeline, conservando 14,427 instancias únicas.
* **Longitud del texto:** Media de 103.8 caracteres (mediana de 114 caracteres, máximo de 186 caracteres), compatible con modelos de n-gramas TF-IDF y Word2Vec sin truncamiento excesivo.

---

### 2. Evidencia de calidad y procedencia
* **Fuente original:** Plataforma Kaggle, publicado a través de la biblioteca de datos abiertos de CrowdFlower (posteriormente Figure Eight / Appen).
* **Fecha de captura:** Febrero de 2015.
* **Licencia de uso:** *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International* (CC BY-NC-SA 4.0), habilitada para investigación académica y docencia reproducible.
* **Consenso humano:** Las anotaciones de sentimiento fueron producidas mediante *crowdsourcing* supervisado, incluyendo una métrica cuantitativa de confianza (`airline_sentiment_confidence`) con una media superior a 0.90 en la gran mayoría del corpus.
* **Integridad criptográfica:** El archivo local genera de forma determinística el hash **SHA-256**:
  ```text
  ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
  ```
* **Verificación de arquitectura:** Se verificó el paso exitoso de la suite de pruebas unitarias (`pytest -q`, 6/6 tests aprobados).

---

### 3. Riesgo o sesgo identificado
1. **Sesgo de severidad y negatividad (Selection Bias):** El 62.69% de los tweets son de polaridad negativa (`negative`), frente a 21.17% neutrales y 16.14% positivos. En canales públicos de atención al cliente, los usuarios acuden predominantemente a expresar quejas e insatisfacción, lo que genera un marcado desbalance de clases que penaliza severamente el uso de métricas simples de exactitud (*accuracy*).
2. **Sesgo contextual y temporal:** La recolección en febrero de 2015 coincidió con eventos meteorológicos invernales extremos en el noreste de Estados Unidos, inflando la frecuencia de motivos de queja como *Cancelled Flight* y *Late Flight*.
3. **Ambigüedad por sarcasmo e ironía:** Presencia recurrente de expresiones sarcásticas (ej. *"Thanks @United for delaying us another 3 hours, excellent job!"*), donde palabras típicamente positivas inducen a error a modelos basados en bolsa de palabras o n-gramas.
4. **Sesgo de población:** Exclusión de usuarios no angloparlantes y de clientes que gestionan sus reclamos por canales privados (mostrador, teléfono o correo electrónico).

---

### 4. Decisión de aprobación o rechazo
**Decisión:** **APROBADO**

**Justificación de la decisión:**
Frente a la alternativa evaluada (*Mental Health Corpus* de Reddit), este dataset presenta ventajas técnicas y metodológicas determinantes:
1. **Alineación arquitectónica directa:** Encaja 1:1 con la estructura multiclase (`negative`, `neutral`, `positive`) del laboratorio base, permitiendo evaluar clasificación con ponderación de clases (`class_weight="balanced"`) y métrica F1-macro sin alterar los contratos de código.
2. **Riesgo ético controlado:** Se trata de interacciones comerciales de acceso público, a diferencia del *Mental Health Corpus*, que involucra publicaciones sobre ideación suicida y depresión severa, con elevadas restricciones bioéticas y de privacidad.
3. **Rendimiento validado:** Permitió alcanzar un F1-macro de **0.741** y un Accuracy de **79%** con Regresión Logística, superando con creces la línea base ingenua (F1 = 0.258).

---

### 5. Limitación que debe comunicarse
* **No aplicable para decisiones operativas automáticas:** La predicción del modelo debe considerarse exclusivamente como una herramienta de apoyo al triaje y priorización de mensajes para operadores humanos; en ningún caso debe usarse para sancionar empleados o descartar reclamos de pasajeros.
* **Menor sensibilidad en la clase neutral:** El modelo presenta su menor rendimiento en la clase `neutral` (F1 = 0.62), confundiéndola frecuentemente con quejas leves debido a la falta de términos afectivos marcados en consultas operativas.
* **Restricción de dominio e idioma:** El modelo entrenado no generaliza a textos en español ni a dominios distintos a la aviación comercial angloparlante sin un reentrenamiento y adaptación de dominio previos.

---

### 6. Siguiente verificación
1. **Auditoría de falsos positivos en sarcasmo:** Evaluar el impacto de incorporar léxicos de polaridad o modelos preentrenados basados en Transformers (ej. RoBERTa entrenado en Twitter) para capturar dependencias contextuales y de negación compleja.
2. **Calibración de umbrales en clase neutral:** Ajustar las probabilidades de salida de la regresión logística mediante análisis de curvas de precisión-exhaustividad para optimizar el punto de corte entre consultas informativas (`neutral`) y quejas latentes (`negative`).
3. **Monitoreo de deriva de datos (*Data Drift*):** En caso de despliegue continuo, programar una rutina de verificación periódica para detectar si la proporción de quejas o el vocabulario de quejas varía frente a nuevas temporadas o aerolíneas.
