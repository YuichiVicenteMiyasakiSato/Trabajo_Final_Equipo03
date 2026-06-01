<h1 align="center"> Efecto del poder de las redes sociales (Reddit) en movimientos bursátiles</h1>

<p align="center">
  <i>Análisis de la correlación entre el sentimiento de r/wallstreetbets y el precio de GME y AMC</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-2.0-150458?logo=pandas&logoColor=white">
</p>

<p align="center">
  Universidad Nacional Autónoma de México - Facultad de Ingeniería<br>
  <b>Análisis y Procesamiento Inteligente de Textos</b> - Equipo 03 - Los Analistas
</p>

---

## Hipótesis

> Un incremento estadísticamente significativo en el volumen de publicaciones
> con sentimiento positivo hacia una acción en r/wallstreetbets está
> correlacionado con un aumento de precio en las siguientes **24–48 horas**.

---

## Experimentos y resultados

> Los números son los que **calcula** `src/analisis.py` sobre los datos
> simulados; no están escritos a mano.

### 1. Detección de tickers · `n = 1,000`

| Método | Precisión | Recall | F1 |
|:--|:--:|:--:|:--:|
| Regex simple | 0.58 | 0.85 | 0.69 |
| **Detector por contexto (NER)** | **0.88** | **0.88** | **0.88** |

El detector con reglas de contexto supera al regex ingenuo, que sobre-detecta
jerga en mayúsculas (`YOLO`, `HODL`, `FOMO`…) y confunde `BB` usado como
"big buy".

### 2. Clasificación de sentimiento · 5-fold CV · `n = 2,400`

| Modelo | F1 (bull) | F1 macro | Notas |
|:--|:--:|:--:|:--|
| VADER-léxico | 0.46 | 0.49 | cálculo real; sufre con ironía |
| SVM + TF-IDF | 0.73 | 0.74 | cálculo real |
| **FinBERT** | **0.78** | **0.79** |  **SIMULADO** (modelo no ejecutado aquí) |

> FinBERT requiere el modelo preentrenado `ProsusAI/finbert` y GPU; en este
> entorno sus predicciones se generaron a una calidad objetivo y están
> marcadas como simuladas. VADER y SVM **sí** se calculan de verdad.

### 3. Correlación ISN vs retorno · días con >= 30 publicaciones

| Acción | Lag | r de Pearson | p-value | Significativo (α = 0.05) |
|:--:|:--:|:--:|:--:|:--:|
| GME | 1 día | **+0.45** | < 0.001 |  Sí |
| GME | 2 días | +0.23 | 0.003 |  Sí |
| GME | 3 días | +0.12 | 0.108 |  No |
| AMC | 1 día | **+0.33** | < 0.001 |  Sí |
| AMC | 2 días | +0.14 | 0.078 |  No |
| AMC | 3 días | +0.07 | 0.351 |  No |

**Robustez** (GME, excluyendo el squeeze de enero): `r = +0.38`, `p < 0.001`.

---

## Conclusión

Sobre los datos, la hipótesis principal **se cumple**: hay
correlación positiva y estadísticamente significativa entre el Índice de
Sentimiento Neto y el retorno a 1–2 días, que decae al tercer día. Esto
demuestra el **funcionamiento del pipeline** sobre datos.

---

## Autores — Equipo 03

- Miyasaki Sato Yuichi Vicente
- Miranda González José Francisco
- Ríos Valdés Oscar

**Profesor:** M.P. Octavio Augusto Sánchez Velázquez

---

<p align="center"><sub>Proyecto académico · datos simulados con fines de demostración · 2026</sub></p>
