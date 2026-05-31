# -*- coding: utf-8 -*-

import re, numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, f1_score

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data")
rng = np.random.default_rng(99)
print("="*72)
print(" RESULTADOS DEL PIPELINE — DATOS SIMULADOS")
print("="*72)


# EXP 1 — Detección de tickers: regex vs detector con reglas de contexto

val = pd.read_csv(f"{OUT}/validacion_tickers_SIMULADO.csv").fillna("")
TICKERS = {"GME","AMC","BB","NOK","TSLA"}
TOKEN_RE = re.compile(r"\$?[A-Za-z]{2,5}\b")
VERBOS = {"BUY","SELL","HOLD","HOLDING","SHORT","PUT","PUTS","CALL","CALLS",
          "SQUEEZE","LONG","ON"}

def detectar_regex(texto):
    """regex ingenuo: propone como ticker CUALQUIER token de 2-5 mayúsculas."""
    out = set()
    for m in re.findall(r"\$?[A-Z]{2,5}\b", texto):
        out.add(m.lstrip("$"))
    return out

def detectar_reglas(texto):
    """detector tipo NER: solo acepta tickers conocidos validados por contexto
    (precedidos de '$' o de un verbo de trading); rescata minúsculas."""
    up = texto.upper()
    toks = up.replace("$", " $").split()
    out = set()
    for i, tok in enumerate(toks):
        cand = tok.lstrip("$")
        if cand not in TICKERS:
            continue
        tiene_dolar = tok.startswith("$")
        prev = toks[i-1] if i > 0 else ""
        ctx = tiene_dolar or (prev in VERBOS) or any(v in up for v in
              ["BUY","SELL","HOLD","SHORT","PUT","CALL","SQUEEZE","LONG","EARNINGS"])
        if ctx:
            out.add(cand)
    return out

def conteos(detector):
    tp = fp = fn = 0
    for _, r in val.iterrows():
        pred = detector(r.texto)
        verdad = {r.ticker_verdadero} if r.ticker_verdadero else set()
        tp += len(pred & verdad)
        fp += len(pred - verdad)
        fn += len(verdad - pred)
    P = tp/(tp+fp) if tp+fp else 0
    R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    return tp, fp, fn, P, R, F1

print("\n[EXP 1] Detección de tickers (validación, n=1000) — CÁLCULO REAL")
for nombre, det in [("Regex simple", detectar_regex),
                    ("Detector reglas (NER)", detectar_reglas)]:
    tp, fp, fn, P, R, F1 = conteos(det)
    print(f"  {nombre:24s} TP={tp:4d} FP={fp:4d} FN={fn:4d} | "
          f"P={P:.2f} R={R:.2f} F1={F1:.2f}")

# ==========================================================================
# EXP 2 — Clasificación de sentimiento
# ==========================================================================
anot = pd.read_csv(f"{OUT}/anotaciones_sentimiento_SIMULADO.csv")
X = anot.texto.values
y = anot.etiqueta_consenso.values

# --- Modelo 1: léxico tipo VADER (CÁLCULO REAL) ---
LEX_POS = {"moon","squeeze","calls","diamond","hold","tendies","buy","yolo",
           "long","bull","rocket","strong","win","loaded","like"}
LEX_NEG = {"puts","sold","dump","overvalued","bubble","short","bearish",
           "wrecked","trash","fake","bagholders","pop","closed"}
def vader_like(t):
    toks = re.findall(r"[a-z]+", t.lower())
    s = sum(w in LEX_POS for w in toks) - sum(w in LEX_NEG for w in toks)
    return "bull" if s > 0 else ("bear" if s < 0 else "neu")
y_vader = np.array([vader_like(t) for t in X])

# --- Modelo 2: SVM + TF-IDF, 5-fold CV (CÁLCULO REAL) ---
svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3,
                                          sublinear_tf=True)),
                ("clf", LinearSVC(C=1.0, class_weight="balanced"))])
y_svm = cross_val_predict(svm, X, y, cv=5)

# --- Modelo 3: FinBERT (SIMULADO: modelo no ejecutado aquí) ---
# Se generan predicciones a una calidad objetivo, EXPLÍCITAMENTE simuladas.
y_fin = y.copy()
flip = rng.random(len(y)) < 0.22                 # ~22% de error inducido
otras = {"bull":["bear","neu"],"bear":["bull","neu"],"neu":["bull","bear"]}
y_fin = np.array([rng.choice(otras[v]) if f else v for v, f in zip(y, flip)])

def fila(nombre, yp, simulado=False):
    P,R,F1,_ = precision_recall_fscore_support(
        y, yp, labels=["bull"], average=None, zero_division=0)
    macro = f1_score(y, yp, average="macro")
    tag = "  (SIMULADO)" if simulado else ""
    print(f"  {nombre:14s} P(bull)={P[0]:.2f} R(bull)={R[0]:.2f} "
          f"F1(bull)={F1[0]:.2f} F1_macro={macro:.2f}{tag}")

print("\n[EXP 2] Clasificación de sentimiento (5-fold CV, n=2400)")
fila("VADER-léxico", y_vader)
fila("SVM + TF-IDF", y_svm)
fila("FinBERT", y_fin, simulado=True)

# ==========================================================================
# EXP 3 — Correlación ISN vs retorno de precio
# ==========================================================================
posts = pd.read_csv(f"{OUT}/reddit_posts_SIMULADO.csv",
                    usecols=["fecha","ticker_real","sentimiento_real"])
px = pd.read_csv(f"{OUT}/precios_diarios_SIMULADO.csv",
                 parse_dates=["fecha"])
print("\n[EXP 3] Correlación de Pearson ISN vs retorno — CÁLCULO REAL")
print("  (solo días con >=30 publicaciones del ticker)")

resultados = {}
for s in ("GME","AMC"):
    p = posts[posts.ticker_real == s]
    g = p.groupby("fecha")["sentimiento_real"].value_counts().unstack(fill_value=0)
    for c in ("bull","bear","neu"):
        if c not in g: g[c] = 0
    g["n"] = g[["bull","bear","neu"]].sum(axis=1)
    g = g[g["n"] >= 30].copy()
    g["ISN"] = (g["bull"] - g["bear"]) / g["n"]
    g.index = pd.to_datetime(g.index)

    pr = px[px.ticker == s].set_index("fecha").sort_index()
    pr["ret"] = pr["cierre"].pct_change()
    df = g.join(pr[["ret"]], how="inner")
    for k in (1,2,3):
        d = pd.DataFrame({"ISN": df["ISN"],
                          "retk": pr["ret"].reindex(df.index.shift(0))})
        # retorno futuro a k días
        future = pr["ret"].shift(-k).reindex(df.index)
        sub = pd.DataFrame({"ISN": df["ISN"].values,
                            "ret": future.values}).dropna()
        r, pval = pearsonr(sub["ISN"], sub["ret"])
        sig = "Sí" if pval < 0.05 else "No"
        resultados[(s,k)] = (r, pval, len(sub))
        print(f"  {s}  lag={k}d  r={r:+.2f}  p={pval:.3f}  n={len(sub):3d}  "
              f"signif={sig}")

# --- robustez: excluir el squeeze de enero (11–29 ene 2021) ---
s = "GME"
p = posts[posts.ticker_real == s]
g = p.groupby("fecha")["sentimiento_real"].value_counts().unstack(fill_value=0)
for c in ("bull","bear","neu"):
    if c not in g: g[c] = 0
g["n"] = g[["bull","bear","neu"]].sum(axis=1)
g = g[g["n"] >= 30].copy(); g["ISN"] = (g["bull"]-g["bear"])/g["n"]
g.index = pd.to_datetime(g.index)
pr = px[px.ticker==s].set_index("fecha").sort_index(); pr["ret"]=pr["cierre"].pct_change()
future = pr["ret"].shift(-1).reindex(g.index)
sub = pd.DataFrame({"ISN":g["ISN"].values,"ret":future.values,
                    "fecha":g.index}).dropna()
mask = ~((sub["fecha"]>="2021-01-11") & (sub["fecha"]<="2021-01-29"))
r_rob, p_rob = pearsonr(sub[mask]["ISN"], sub[mask]["ret"])
print(f"\n  Robustez GME (sin squeeze ene): r={r_rob:+.2f} p={p_rob:.3f} "
      f"n={mask.sum()}")
print("="*72)

# --------------------------------------------------------------------------
# Exportar tabla diaria de ISN (artefacto del análisis) — DATOS SIMULADOS
# --------------------------------------------------------------------------
filas_isn = []
for s in ("GME","AMC"):
    p = posts[posts.ticker_real == s]
    g = p.groupby("fecha")["sentimiento_real"].value_counts().unstack(fill_value=0)
    for c in ("bull","bear","neu"):
        if c not in g: g[c] = 0
    g["n_total"] = g[["bull","bear","neu"]].sum(axis=1)
    g = g[g["n_total"] >= 30].copy()
    g["ISN"] = ((g["bull"] - g["bear"]) / g["n_total"]).round(4)
    g["ticker"] = s; g["origen"] = "simulado"
    g = g.reset_index().rename(columns={"bull":"n_bull","bear":"n_bear",
                                        "neu":"n_neu"})
    filas_isn.append(g[["fecha","ticker","n_bull","n_bear","n_neu",
                        "n_total","ISN","origen"]])
pd.concat(filas_isn).to_csv(f"{OUT}/isn_diario_SIMULADO.csv", index=False)
print("Tabla diaria de ISN exportada: datos/isn_diario_SIMULADO.csv")
