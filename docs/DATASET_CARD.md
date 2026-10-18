# Ficha Técnica del Dataset (Dataset Card)

## Identificación
- **Nombre:** Twitter US Airline Sentiment (Análisis de Sentimiento de Aerolíneas de EE. UU.)
- **Fuente original:** Kaggle / CrowdFlower Data for Everyone Library ([Enlace Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment))
- **Responsable:** CrowdFlower (actualmente Figure Eight / Appen)
- **Versión o fecha:** Febrero de 2015
- **Licencia:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Idioma:** Inglés (`en`)

---

## Propósito y variable objetivo
- **Propósito:** Clasificar de manera automatizada las opiniones, retroalimentaciones y quejas de los pasajeros dirigidas a las principales aerolíneas comerciales de Estados Unidos a través de Twitter, permitiendo priorizar la atención al cliente y detectar incidentes operativos en tiempo real.
- **Variable objetivo (`TARGET_COLUMN`):** `airline_sentiment`. Representa la valencia emocional de cada tweet y adopta tres clases discretas:
  - `negative` (negativo): Expresiones de descontento, demoras, cancelaciones, extravío de equipaje o mala atención.
  - `neutral` (neutral): Consultas factuales, estados de vuelo o información operativa sin carga emocional explícita.
  - `positive` (positivo): Felicitaciones, agradecimientos y menciones de buen servicio.

---

## Diccionario de datos

| Columna | Tipo | Descripción | Valores o unidad |
|---|---|---|---|
| `tweet_id` | `int64` | Identificador único del tweet en la plataforma Twitter | Identificador numérico entero |
| `airline_sentiment` | `string` | **Variable objetivo**: Clasificación de sentimiento asignada | `negative`, `neutral`, `positive` |
| `airline_sentiment_confidence` | `float64` | Nivel de confianza o consenso entre anotadores humanos | Rango decimal de `0.0` a `1.0` |
| `negativereason` | `string` | Categoría principal atribuida a la queja (si el sentimiento es negativo) | `Bad Flight`, `Cancelled Flight`, `Late Flight`, `Customer Service Issue`, `Lost Luggage`, etc. |
| `negativereason_confidence` | `float64` | Nivel de confianza para la razón negativa asignada | Rango decimal de `0.0` a `1.0` |
| `airline` | `string` | Aerolínea estadounidense mencionada o destinataria | `Virgin America`, `United`, `Southwest`, `Delta`, `US Airways`, `American` |
| `airline_sentiment_gold` | `string` | Etiqueta de control de calidad auditada por expertos (*gold standard*) | `negative`, `neutral`, `positive` (escaso) |
| `name` | `string` | Nombre de usuario de la cuenta de Twitter del autor | Texto alfanumérico |
| `negativereason_gold` | `string` | Motivo de queja auditado por control de calidad | Texto descriptivo |
| `retweet_count` | `int64` | Cantidad de veces que el tweet fue retuiteado | Entero $\ge 0$ |
| `text` | `string` | **Variable de entrada**: Contenido textual original del mensaje | Cadena de texto libre (longitud histórica $\le 140$ caracteres) |
| `tweet_coord` | `string` | Coordenadas geográficas de emisión del tweet (si estaban activas) | `[latitud, longitud]` (mayormente nulo) |
| `tweet_created` | `string` | Marca temporal de publicación del tweet | Formato ISO/Timestamp (Febrero 2015) |
| `tweet_location` | `string` | Ubicación geográfica declarada en el perfil del usuario | Texto libre |
| `user_timezone` | `string` | Zona horaria configurada en la cuenta del usuario | Texto descriptivo (ej. `Eastern Time (US & Canada)`) |

---

## Procedimiento de obtención
Los datos fueron recolectados mediante la API pública de Twitter durante el mes de febrero de 2015, filtrando mensajes dirigidos a las cuentas oficiales de 6 grandes aerolíneas estadounidenses (`@VirginAmerica`, `@United`, `@SouthwestAir`, `@Delta`, `@USAirways`, `@AmericanAir`). Posteriormente, CrowdFlower sometió los mensajes a anotación colaborativa (*crowdsourcing*), donde cada tweet fue evaluado por múltiples revisores independientes para asignar el sentimiento y la confianza asociada.

---

## Calidad observada
La auditoría ejecutada con `scripts/audit_data.py` arrojó los siguientes resultados verificables:
- **Total de registros:** 14,640 filas y 15 columnas.
- **Hash criptográfico (SHA-256):** `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`.
- **Valores nulos en variables clave:** 0 en `text`, 0 en `airline_sentiment` (100% de integridad en el contrato de datos).
- **Textos duplicados:** 213 registros (mensajes automáticos, bots o citas idénticas). Fueron removidos automáticamente en el pipeline de preprocesamiento, quedando 14,427 tweets únicos.
- **Distribución de clases (Desbalance pronunciado):**
  - `negative`: 62.69% (9,178 tweets)
  - `neutral`: 21.17% (3,099 tweets)
  - `positive`: 16.14% (2,363 tweets)
- **Longitud de los textos (en caracteres):**
  - Media: 103.82 caracteres
  - Desviación estándar: 36.28 caracteres
  - Mediana (P50): 114 caracteres | P90: 140 caracteres | Máximo: 186 caracteres (incluye entidades HTML como `&amp;`).

---

## Población cubierta y excluida
- **Población cubierta:** Pasajeros y usuarios de habla inglesa con cuenta activa en Twitter que interactuaron públicamente con aerolíneas estadounidenses durante febrero de 2015.
- **Población excluida:**
  - Pasajeros que no utilizan redes sociales o no poseen cuenta de Twitter.
  - Clientes que manifestaron quejas o felicitaciones por canales privados (centro de llamadas, correo electrónico, mostradores de aeropuerto, encuestas post-vuelo).
  - Usuarios no angloparlantes o vuelos de aerolíneas no incluidas en el muestreo.

---

## Riesgos, sesgos y usos prohibidos
- **Sesgo de negatividad (Selection Bias):** Los usuarios recurren a Twitter con mayor probabilidad cuando sufren una experiencia frustrante (cancelaciones, equipaje perdido) que cuando el servicio opera con normalidad. Esto explica la concentración del 62.7% en sentimiento negativo.
- **Sesgo temporal y contextual:** Febrero de 2015 coincidió con tormentas invernales severas en la costa este de EE. UU., provocando picos anómalos de quejas por vuelos cancelados que no necesariamente reflejan el estándar operativo general.
- **Sarcasmo e ironía:** Twitter presenta un alto volumen de lenguaje sarcástico (ej. *"Thanks for leaving my bag in Chicago, great job!"*), lo cual introduce ruido en los modelos basados únicamente en bolsa de palabras o n-gramas.
- **Usos prohibidos:**
  - **No usar para evaluación individual ni sanciones automáticas** sobre la tripulación o el personal de servicio al cliente.
  - **No usar para discriminación o denegación de servicios** a pasajeros en base a su historial o tono en redes sociales.
  - La predicción generada es una herramienta analítica orientada a monitoreo agregado y no constituye una decisión operativa automatizada.
