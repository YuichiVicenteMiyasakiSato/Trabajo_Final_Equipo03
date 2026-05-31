# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

rng = np.random.default_rng(2021)
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# 1) Calendario bursátil ene–ago 2021 (días hábiles)
# --------------------------------------------------------------------------
dias = pd.bdate_range("2021-01-04", "2021-08-31")
T = len(dias)

# --------------------------------------------------------------------------
# 2) Factor latente de sentimiento z_t por acción (AR(1) + eventos)
# --------------------------------------------------------------------------
def factor_latente(spikes):
    z = np.zeros(T); phi = 0.55
    for t in range(1, T):
        z[t] = phi * z[t-1] + rng.normal(0, 0.8)
    for ini, fin, amp in spikes:
        m = (dias >= ini) & (dias <= fin)
        z[m] += amp
    return (z - z.mean()) / z.std()

z = {
    # GME: gran pico del short squeeze a finales de enero 2021
    "GME": factor_latente([("2021-01-22", "2021-01-29", 3.2),
                           ("2021-02-24", "2021-03-10", 1.4),
                           ("2021-06-01", "2021-06-09", 1.1)]),
    # AMC: pico menor en enero y un fuerte repunte en junio 2021
    "AMC": factor_latente([("2021-01-25", "2021-01-29", 1.6),
                           ("2021-05-28", "2021-06-11", 2.4)]),
}

# --------------------------------------------------------------------------
# 3) Retornos de precio: R(t+1) depende de z_t  (b*z_t + ruido)
# --------------------------------------------------------------------------
def precios(zt, b, ruido, p0):
    ret = np.zeros(T)
    for t in range(T-1):
        ret[t+1] = b * zt[t] + rng.normal(0, ruido)
    # amplificar el squeeze de GME para realismo visual
    ret = np.clip(ret, -0.35, 0.6)
    precio = p0 * np.cumprod(1 + ret)
    return precio, ret

precio = {}; ret = {}
precio["GME"], ret["GME"] = precios(z["GME"], b=0.060, ruido=0.11, p0=17.0)
precio["AMC"], ret["AMC"] = precios(z["AMC"], b=0.050, ruido=0.12, p0=2.0)

filas_px = []
for s in ("GME", "AMC"):
    base = precio[s]
    for t, d in enumerate(dias):
        cierre = round(float(base[t]), 2)
        apertura = round(cierre * (1 + rng.normal(0, 0.015)), 2)
        alto = round(max(apertura, cierre) * (1 + abs(rng.normal(0, 0.02))), 2)
        bajo = round(min(apertura, cierre) * (1 - abs(rng.normal(0, 0.02))), 2)
        vol = int(abs(rng.normal(0, 1)) * 4e7 + 1e7 + abs(z[s][t]) * 9e7)
        filas_px.append([d.date(), s, apertura, alto, bajo, cierre, vol, "simulado"])

px = pd.DataFrame(filas_px, columns=["fecha","ticker","apertura","alto",
                                     "bajo","cierre","volumen","origen"])
px.to_csv(f"{OUT}/precios_diarios_SIMULADO.csv", index=False)

# --------------------------------------------------------------------------
# 4) Publicaciones de Reddit (texto plantillado con jerga WSB)
# --------------------------------------------------------------------------
PLANT = {
 "bull": ["{tk} to the moon 🚀🚀 diamond hands hold the line",
          "buying more ${tk} calls, this short squeeze is just starting",
          "${tk} gamma squeeze incoming, hedgies are done. YOLO 100k",
          "we like the stock 💎🙌 ${tk} price target 500 easy",
          "loaded up on {tk} today, apes together strong 🦍",
          "${tk} breaking resistance, tendies on the way 🤑",
          "hold ${tk}, never selling, this is the way",
          "all in on {tk} leaps, retail is winning this one"],
 "bear": ["${tk} is overvalued, this bubble will pop hard",
          "sold all my ${tk}, taking profits before the dump",
          "puts on {tk}, the squeeze is over, dead cat bounce",
          "{tk} bagholders about to get wrecked, short it",
          "this {tk} pump is fake, insiders dumping on retail",
          "${tk} going back to single digits, bearish af",
          "closed my {tk} position, momentum is gone",
          "shorting ${tk} here, fundamentals are trash"],
 "neu": ["what do you guys think about {tk} earnings next week",
         "any DD on ${tk}? new to this stock",
         "{tk} volume looks normal today, sideways action",
         "is ${tk} still being talked about here? asking",
         "holding a small {tk} position, watching for now",
         "${tk} options chain looks thin, low IV today",
         "anyone have the {tk} float numbers handy",
         "neutral on {tk}, waiting for a clearer signal"],
}
# frases ambiguas compartidas por todas las clases (dificultan al clasificador)
AMBIGUAS = ["{tk} is wild today, no idea where this goes",
            "watching ${tk} closely, could go either way 👀",
            "{tk} moving a lot, volume is insane rn",
            "${tk} again huh, this sub never sleeps",
            "idk man {tk} is doing {tk} things",
            "${tk} chart looks crazy, anyone else seeing this"]
# ironía: el texto contradice la etiqueta real (típico de WSB)
IRONICAS = {
 "bull": ["${tk} going to zero... TO THE MOON 🚀 (im not selling)",
          "{tk} is trash but im holding anyway lmao 💎🙌",
          "everyone says sell {tk}, so obviously i bought more"],
 "bear": ["${tk} to the moon 🚀 said the bagholder before the dump",
          "diamond hands on {tk}? more like paper hands incoming",
          "buy the dip they said... {tk} keeps dipping 📉"],
 "neu": ["{tk} 🚀🚀 puts and calls and whatever idk",
         "${tk} moon dump moon dump, pick one already"],
}
# confundidores: 'GME' no-financiero, 'BB' = 'big buy', URLs, emojis-ruido
CONFUSORES = ["lol GME just game me some tendies bro 😂",
              "BB big buy on everything, no ticker here",
              "check this https://t.co/abc123 not financial advice",
              "🚀🚀🚀🚀🚀🚀 wen lambo 🦍🦍 to the mooooooon",
              "deleted [removed] [removed] spam spam spam"]

def gen_texto(clase, tk):
    """genera texto con mezcla de claro / ambiguo / irónico para realismo."""
    u = rng.random()
    if u < 0.34:
        return rng.choice(PLANT[clase]).format(tk=tk)        # claro
    elif u < 0.80:
        return rng.choice(AMBIGUAS).format(tk=tk)            # ambiguo
    else:
        return rng.choice(IRONICAS[clase]).format(tk=tk)     # irónico

N_TOTAL = 148320
peso_dia = {}
for s in ("GME", "AMC"):
    base = np.exp(1.3 * np.abs(z[s])) + 0.4
    peso_dia[s] = base / base.sum()
# GME concentra ~62% de las publicaciones
share = {"GME": 0.62, "AMC": 0.38}

autores = [f"u/ape_{i:05d}" for i in range(8000)]
filas = []
pid = 0
for s in ("GME", "AMC"):
    n_s = int(N_TOTAL * share[s])
    conteo = rng.multinomial(n_s, peso_dia[s])
    for t, d in enumerate(dias):
        n = conteo[t]
        if n == 0:
            continue
        zt = z[s][t]
        # probabilidades de clase dependientes del factor latente z_t
        logits = np.array([1.0*zt, -1.0*zt, 0.0])      # bull, bear, neu
        p = np.exp(logits) / np.exp(logits).sum()
        clases = rng.choice(["bull","bear","neu"], size=n, p=p)
        for c in clases:
            # 6% de ruido/confundidores que ensucian el texto
            if rng.random() < 0.06:
                txt = rng.choice(CONFUSORES)
                lab = "neu"
            else:
                txt = gen_texto(c, s)
                lab = c
            ts = int(pd.Timestamp(d).timestamp()) + int(rng.integers(0, 86400))
            filas.append([f"t3_{pid:07d}", ts, str(d.date()), s, txt, lab,
                          int(abs(rng.normal(0,1))*300), rng.choice(autores),
                          "simulado"])
            pid += 1

posts = pd.DataFrame(filas, columns=["id","timestamp","fecha","ticker_real",
        "texto","sentimiento_real","score","autor","origen"])
posts = posts.sample(frac=1, random_state=7).reset_index(drop=True)
posts.to_csv(f"{OUT}/reddit_posts_SIMULADO.csv", index=False)

# --------------------------------------------------------------------------
# 5) Subconjunto etiquetado para clasificación (2,400 balanceado)
# --------------------------------------------------------------------------
limpio = posts[posts.texto.str.contains(r"\{|removed", regex=True) == False]
muestra = []
for c in ("bull","bear","neu"):
    sub = limpio[limpio.sentimiento_real == c].sample(800, random_state=11)
    muestra.append(sub)
anot = pd.concat(muestra).sample(frac=1, random_state=13).reset_index(drop=True)
anot = anot[["id","fecha","ticker_real","texto","sentimiento_real","origen"]]
anot.columns = ["id","fecha","ticker","texto","etiqueta_consenso","origen"]
anot.to_csv(f"{OUT}/anotaciones_sentimiento_SIMULADO.csv", index=False)

# --------------------------------------------------------------------------
# 6) Conjunto de validación de detección de tickers (1,000)
#    Diseñado para que un regex ingenuo (cualquier token en MAYÚSCULAS) genere
#    falsos positivos con jerga (YOLO, HODL, CEO...) y pierda tickers escritos
#    en minúscula; el detector por contexto debe resolver ambos casos.
# --------------------------------------------------------------------------
TK = ["GME","AMC","BB","NOK","TSLA"]
JERGA = ["YOLO","HODL","FOMO","ATH","DD","WSB","CEO","SEC","FUD"]
val = []
for i in range(1000):
    u = rng.random()
    jg = f" {rng.choice(JERGA)}" if rng.random() < 0.10 else ""   # jerga ocasional
    if u < 0.50:                                  # ticker real CON contexto
        tk = rng.choice(TK)
        forma = tk if rng.random() < 0.83 else tk.lower()         # 83% mayúscula
        verbo = rng.choice(["buy","holding","squeeze on","puts on","long"])
        texto = f"{verbo} ${forma}{jg} lets go" if rng.random()<0.5 \
                else f"{verbo} {forma}{jg} lets go"
        verdad = tk
    elif u < 0.56:                                # ticker real SIN contexto
        tk = rng.choice(TK)
        forma = tk if rng.random() < 0.83 else tk.lower()
        texto = f"{forma}{jg} thoughts anyone"                    # NER lo pierde
        verdad = tk
    elif u < 0.63:                                # 'BB' = big buy (no financiero)
        texto = f"buy BB big buy energy today{jg}"                # NER falso +
        verdad = ""
    else:                                         # sin ticker (mayormente minúsc.)
        if rng.random() < 0.30:                   # minoría con jerga MAYÚS -> FP regex
            texto = f"{rng.choice(JERGA)} {rng.choice(JERGA)} this is the way"
        else:                                     # minúsculas: regex acierta vacío
            texto = rng.choice(["game me some tendies lol no ticker",
                                "to the moon diamond hands not advice",
                                "wen lambo apes together strong"])
        verdad = ""
    val.append([f"v_{i:04d}", texto, verdad, "simulado"])
pd.DataFrame(val, columns=["id","texto","ticker_verdadero","origen"]
            ).to_csv(f"{OUT}/validacion_tickers_SIMULADO.csv", index=False)

print("Generación completa (DATOS SIMULADOS):")
print(f"  reddit_posts_SIMULADO.csv .......... {len(posts):,} filas")
print(f"  precios_diarios_SIMULADO.csv ....... {len(px):,} filas ({T} días x 2)")
print(f"  anotaciones_sentimiento_SIMULADO.csv {len(anot):,} filas")
print(f"  validacion_tickers_SIMULADO.csv .... 1,000 filas")
