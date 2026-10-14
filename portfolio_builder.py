"""
Formación de cartera con restricciones (MTaurus – Chart Lab).

Idea: elegís los instrumentos que sí o sí querés tener (con su peso) y la herramienta busca qué
agregar y en qué proporción para reducir el riesgo, usando la estructura de correlaciones del universo.

Qué hace
  1. Ranking de diversificadores: para cada candidato, correlación con el "núcleo" (tus imprescindibles),
     cuánto baja la volatilidad si le sumás un X% y qué le pasa en los peores días del núcleo.
  2. Optimización con restricciones: mínima varianza, paridad de riesgo entre los complementos o máxima
     diversificación, con máximo por activo, topes para inversos y apalancados, posición mínima y
     cantidad máxima de activos.
  3. Validación fuera de muestra (walk-forward): estima con una ventana, evalúa en el período siguiente
     y compara contra el núcleo solo y contra igual ponderación.

Qué NO hace: no predice retornos ni es una recomendación de inversión. Minimizar varianza no es cubrir un
riesgo concreto; y ignora CCL, ratios de CEDEAR, liquidez y comisiones (trabaja con retornos en dólares
del subyacente).

Uso en app.py:  render_portfolio_ui(load_correlation_groups())
"""
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

from correlation_plot import INVERSE_TICKERS, _order_by_clustering

# ================= CONFIG =================
# Productos con decaimiento / apalancamiento: la correlación diaria los hace ver mejor de lo que son
# como cobertura a plazos más largos. Por defecto quedan afuera de la optimización (tope 0%).
INVERSE_GROUP = set(INVERSE_TICKERS) | {'SQQQ', 'SPXU', 'SDS', 'SDOW', 'UVXY', 'SOXS'}
LEVERAGED_GROUP = {'TQQQ', 'SPXL', 'UPRO', 'SOXL', 'QLD', 'SSO', 'UDOW', 'TNA'}

OBJECTIVES = {
    "Mínima varianza": "minvar",
    "Paridad de riesgo entre los complementos": "erc",
    "Máxima diversificación": "maxdiv",
}

COLOR_CORE = '#ffaa60'
COLOR_COMP = '#8f96d8'
TAIL_Q = 0.05          # "peores días" = 5% inferior de los retornos del núcleo


class PortfolioError(Exception):
    """Error de uso (restricciones imposibles, datos insuficientes, etc.) con mensaje para el usuario."""


def ticker_type(t):
    if t in INVERSE_GROUP:
        return "Inverso / volatilidad"
    if t in LEVERAGED_GROUP:
        return "Apalancado"
    return "Normal"


# ================= DATOS =================
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, years):
    """Precios de cierre ajustados (yfinance) de los tickers, últimos `years` años."""
    tickers = tuple(tickers)
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 7)
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    close = data['Close'] if 'Close' in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how='all').astype(float)


def to_returns(close, freq='D'):
    """Retornos simples diarios ('D') o semanales ('W': cierre del viernes)."""
    if freq == 'W':
        close = close.resample('W-FRI').last()
    return close.pct_change(fill_method=None).iloc[1:]


# ================= COVARIANZA Y ESTADÍSTICAS =================
def ledoit_wolf_cov(X):
    """Covarianza con shrinkage de Ledoit-Wolf hacia la identidad escalada. Devuelve (S, intensidad)."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    T, N = X.shape
    S = X.T @ X / T
    mu = np.trace(S) / N
    d2 = float(((S - mu * np.eye(N)) ** 2).sum())
    if d2 <= 0:
        return S, 0.0
    b2 = (np.sum(np.sum(X ** 2, axis=1) ** 2) - 2 * np.sum((X @ S) * X) + T * np.sum(S ** 2)) / T ** 2
    k = float(min(max(b2, 0.0), d2) / d2)
    return k * mu * np.eye(N) + (1 - k) * S, k


def estimate_cov(R, shrink, ann):
    """Covarianza anualizada de los retornos R (DataFrame) y la intensidad de shrinkage usada."""
    if shrink:
        S, k = ledoit_wolf_cov(R.values)
    else:
        S, k = np.cov(R.values, rowvar=False), 0.0
    return np.atleast_2d(S) * ann, k


def portfolio_stats(w, Sigma, tickers):
    """Volatilidad anual, contribución de cada activo al riesgo y ratio de diversificación."""
    w = np.asarray(w, dtype=float)
    m = Sigma @ w
    var = float(w @ m)
    vol = float(np.sqrt(max(var, 1e-18)))
    sig = np.sqrt(np.diag(Sigma))
    return {
        'vol': vol,
        'rc': pd.Series(w * m / max(var, 1e-18), index=tickers),
        'dr': float((w * sig).sum() / vol),
    }


def max_drawdown(values):
    v = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(v)
    return float((v / peak - 1).min())


# ================= RANKING DE DIVERSIFICADORES =================
def normalized_core(mandatory):
    w = pd.Series(mandatory, dtype=float)
    if w.sum() <= 0:
        raise PortfolioError("Los pesos de los imprescindibles suman 0%: poné al menos uno mayor a cero.")
    return w / w.sum()


def diversifier_ranking(R, mandatory, ann, blend=0.10, tail_q=TAIL_Q):
    """
    Para cada candidato (no imprescindible): correlación con el núcleo, peso de mínima varianza en la
    mezcla núcleo+candidato, reducción de volatilidad con ese peso y con `blend`, y retorno medio del
    candidato en los peores días del núcleo.
    """
    wn = normalized_core(mandatory)
    r_core = R[list(wn.index)] @ wn
    vc = float(r_core.var(ddof=1))
    tail = r_core <= r_core.quantile(tail_q)
    vol_core = float(np.sqrt(vc * ann))

    def mix_vol(a, vj, cov):
        return float(np.sqrt(max((1 - a) ** 2 * vc + a ** 2 * vj + 2 * a * (1 - a) * cov, 0) * ann))

    rows = []
    for t in R.columns:
        if t in mandatory:
            continue
        r = R[t]
        vj = float(r.var(ddof=1))
        cov = float(np.cov(r_core.values, r.values, ddof=1)[0, 1])
        rho = cov / np.sqrt(vc * vj) if vc > 0 and vj > 0 else np.nan
        den = vc + vj - 2 * cov
        a_star = float(np.clip((vc - cov) / den, 0, 1)) if den > 0 else 0.0
        rows.append({
            'Ticker': t,
            'Tipo': ticker_type(t),
            'Correlación con el núcleo': rho,
            'Vol. anual del activo (%)': np.sqrt(vj * ann) * 100,
            f'Vol. con {blend:.0%} (%)': mix_vol(blend, vj, cov) * 100,
            f'Reducción de vol. con {blend:.0%} (%)': (1 - mix_vol(blend, vj, cov) / vol_core) * 100,
            'Peso de mínima varianza (%)': a_star * 100,
            'Reducción de vol. con ese peso (%)': (1 - mix_vol(a_star, vj, cov) / vol_core) * 100,
            'Retorno medio en los peores días del núcleo (%)': float(r[tail].mean()) * 100,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(f'Reducción de vol. con {blend:.0%} (%)', ascending=False).reset_index(drop=True)


# ================= OPTIMIZACIÓN =================
def _solve(Sigma, lb, ub, ineq, objective, free_mask):
    """Un solve de SLSQP con cotas lb/ub, suma 1 y topes de grupo (lista de (índices, tope))."""
    n = len(lb)
    span = ub - lb
    if span.sum() < 1 - lb.sum() - 1e-9:
        raise PortfolioError("Con ese máximo por activo no alcanza para llegar a 100%: subí el máximo "
                             "por activo, ampliá el universo o permití más inversos/apalancados.")
    x0 = lb + (1 - lb.sum()) * span / max(span.sum(), 1e-12)
    sig = np.sqrt(np.diag(Sigma))

    jac = None
    if objective == 'minvar':
        def fun(w):
            return float(w @ Sigma @ w) * 100.0

        def jac(w):
            return 2.0 * (Sigma @ w) * 100.0
    elif objective == 'erc':
        def fun(w):
            m = Sigma @ w
            var = max(float(w @ m), 1e-18)
            f = (w * m / var)[free_mask]
            return float(((f - f.mean()) ** 2).sum()) * 1e3
    elif objective == 'maxdiv':
        def fun(w):
            return -float(w @ sig) / np.sqrt(max(float(w @ Sigma @ w), 1e-18))
    else:
        raise PortfolioError(f"Objetivo desconocido: {objective}")

    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0, 'jac': lambda w: np.ones_like(w)}]
    for mem, cap in ineq:
        g = np.zeros(n)
        g[mem] = -1.0
        cons.append({'type': 'ineq', 'fun': (lambda w, mem=mem, cap=cap: cap - w[mem].sum()),
                     'jac': (lambda w, g=g: g)})

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(fun, x0, jac=jac, bounds=list(zip(lb, ub)), constraints=cons,
                       method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-12})
    w = np.clip(res.x, lb, ub)
    ok_sum = abs(w.sum() - 1) < 1e-4
    ok_grp = all(w[mem].sum() <= cap + 1e-4 for mem, cap in ineq)
    if not (ok_sum and ok_grp):
        raise PortfolioError(f"El optimizador no encontró una solución que cumpla las restricciones ({res.message}).")
    return w


def optimize_portfolio(Sigma, tickers, mandatory, objective='minvar', max_w=0.25, exact=True,
                       caps=None, min_pos=0.0, max_assets=0):
    """
    Pesos (Series indexada por ticker) que optimizan `objective` cumpliendo:
      - imprescindibles: peso mínimo (exact=False) o exacto (exact=True)
      - máximo `max_w` por complemento; suma 100%; sin posiciones cortas
      - caps: dict nombre -> (conjunto de tickers, tope total); tope 0 = excluidos (salvo imprescindibles)
      - min_pos: se descartan complementos con menos de ese peso (y se reoptimiza)
      - max_assets: cantidad máxima de activos en total, incluyendo imprescindibles (0 = sin límite)
    """
    tickers = list(tickers)
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    for t in mandatory:
        if t not in idx:
            raise PortfolioError(f"{t} no tiene datos suficientes en la ventana elegida.")
    lb = np.zeros(n)
    ub = np.full(n, float(max_w))
    mand_idx = [idx[t] for t in mandatory]
    for t, m in mandatory.items():
        lb[idx[t]] = m
        ub[idx[t]] = m if exact else max(m, max_w)
    ineq = []
    for _, (members, cap) in (caps or {}).items():
        mem = [idx[t] for t in members if t in idx and t not in mandatory]
        if not mem:
            continue
        if cap <= 0:
            ub[mem] = 0.0
        else:
            ineq.append((mem, float(cap)))
    if lb.sum() > 1 + 1e-9:
        raise PortfolioError("Los pesos mínimos de los imprescindibles suman más de 100%.")
    if ub.sum() < 1 - 1e-9:
        raise PortfolioError("Con ese máximo por activo no alcanza para llegar a 100%: subí el máximo "
                             "por activo o ampliá el universo.")
    n_mand = len(mand_idx)
    if max_assets and max_assets <= n_mand and abs(lb.sum() - 1) > 1e-9:
        raise PortfolioError("El máximo de activos tiene que ser mayor que la cantidad de imprescindibles.")

    mand_mask = np.zeros(n, dtype=bool)
    mand_mask[mand_idx] = True
    active_ub = ub.copy()
    w = None
    for _ in range(15):
        free_mask = (~mand_mask) & (active_ub > 0)
        if objective == 'erc' and not free_mask.any() and abs(lb.sum() - 1) > 1e-9:
            raise PortfolioError("No quedan complementos disponibles para repartir el riesgo.")
        w = _solve(Sigma, lb, active_ub, ineq, objective, free_mask)
        w[(w < 1e-6) & (lb <= 0)] = 0.0
        diff = 1.0 - w.sum()
        fp = (w > 0) & (~mand_mask)
        if abs(diff) > 1e-9 and fp.any():
            w[fp] += diff * w[fp] / w[fp].sum()
        free_pos = [i for i in range(n) if (not mand_mask[i]) and w[i] > 1e-9]
        drop = [i for i in free_pos if w[i] < min_pos]
        if max_assets:
            allowed = max_assets - n_mand
            need = len(free_pos) - len(drop) - allowed
            if need > 0:
                rest = sorted((i for i in free_pos if i not in drop), key=lambda i: w[i])
                drop += rest[: need if need <= 2 else int(np.ceil(need / 2))]
        if not drop:
            break
        active_ub[drop] = 0.0
    return pd.Series(w, index=tickers)


# ================= VALIDACIÓN FUERA DE MUESTRA =================
def walk_forward(R, mandatory, opt_kwargs, est_periods, reb_periods, ann, shrink=True):
    """
    Cada `reb_periods` re-estima la covarianza con los últimos `est_periods`, optimiza y mantiene esos
    pesos (sin rebalanceo intermedio) durante el período siguiente.
    Devuelve (DataFrame con el valor acumulado de 'Cartera optimizada', 'Solo imprescindibles' e
    'Igual ponderación' (sin inversos ni apalancados), cantidad de rebalanceos donde falló el optimizador).
    """
    tickers = list(R.columns)
    n = len(R)
    core = pd.Series(0.0, index=tickers)
    core[list(mandatory)] = normalized_core(mandatory)
    elig = [t for t in tickers if t not in INVERSE_GROUP and t not in LEVERAGED_GROUP]
    eq = pd.Series(0.0, index=tickers)
    eq[elig] = 1.0 / max(len(elig), 1)
    names = ['Cartera optimizada', 'Solo imprescindibles', 'Igual ponderación']
    cur = dict.fromkeys(names, 1.0)
    out = {k: [] for k in names}
    idx = []
    fails = 0
    last = core.copy()
    for s in range(est_periods, n, reb_periods):
        window = R.iloc[s - est_periods:s]
        Sigma, _ = estimate_cov(window, shrink, ann)
        try:
            w = optimize_portfolio(Sigma, tickers, mandatory, **opt_kwargs)
            last = w
        except PortfolioError:
            fails += 1
            w = last
        period = R.iloc[s:s + reb_periods]
        growth = (1 + period).cumprod().values
        for name, wv in zip(names, (w, core, eq)):
            vals = cur[name] * (growth @ wv.values)
            out[name].extend(vals.tolist())
            cur[name] = out[name][-1]
        idx.extend(period.index)
    return pd.DataFrame(out, index=idx), fails


def oos_summary(df, ann):
    rows = []
    for col in df.columns:
        v = df[col].values
        ret = pd.Series(v).pct_change().dropna()
        years = max(len(v) / ann, 1e-9)
        rows.append({
            'Estrategia': col,
            'Retorno acumulado (%)': (v[-1] - 1) * 100,
            'Retorno anualizado (%)': (v[-1] ** (1 / years) - 1) * 100,
            'Volatilidad anual (%)': float(ret.std(ddof=1) * np.sqrt(ann)) * 100,
            'Máxima caída (%)': max_drawdown(np.r_[1.0, v]) * 100,
        })
    return pd.DataFrame(rows)


# ================= ANÁLISIS COMPLETO (sin Streamlit) =================
def analyze(R_raw, mandatory, *, objective='minvar', max_w=0.25, exact=True, cap_inverse=0.0,
            cap_leveraged=0.0, min_pos=0.02, max_assets=0, shrink=True, freq='D', blend=0.10,
            oos=True, est_years=2, reb_months=3, min_hist=0.8):
    """
    R_raw: retornos (DataFrame, columnas = tickers del universo; puede tener NaN por historia corta).
    mandatory: dict ticker -> peso (fracción) de los imprescindibles.
    Devuelve un dict con pesos, estadísticas, ranking, correlaciones, validación fuera de muestra y avisos.
    """
    ann = 252 if freq == 'D' else 52
    notes = []
    mandatory = {t: float(w) for t, w in mandatory.items()}
    if not mandatory:
        raise PortfolioError("Elegí al menos un imprescindible.")
    keep = R_raw.notna().mean() >= min_hist
    dropped = [c for c in R_raw.columns if not keep[c]]
    bad = [t for t in mandatory if t in dropped or t not in R_raw.columns]
    if bad:
        raise PortfolioError(f"{', '.join(bad)}: historia insuficiente en la ventana elegida "
                             f"(se exige al menos {min_hist:.0%} de la ventana). Acortá la ventana o sacalos.")
    if dropped:
        notes.append(f"Excluidos por historia corta: {', '.join(dropped)}.")
    R = R_raw.loc[:, keep].dropna()
    tickers = list(R.columns)
    if len(R) < 30 or len(tickers) < 3:
        raise PortfolioError("No hay datos suficientes para estimar la covarianza (probá con una ventana mayor).")
    if len(R) < 2 * len(tickers):
        notes.append("Hay pocas observaciones respecto de la cantidad de activos: la covarianza es ruidosa. "
                     "Conviene ventana más larga, frecuencia diaria o un universo más chico.")

    Sigma, k = estimate_cov(R, shrink, ann)
    caps = {'inversos': (INVERSE_GROUP, cap_inverse), 'apalancados': (LEVERAGED_GROUP, cap_leveraged)}
    opt_kwargs = dict(objective=objective, max_w=max_w, exact=exact, caps=caps,
                      min_pos=min_pos, max_assets=max_assets)
    w = optimize_portfolio(Sigma, tickers, mandatory, **opt_kwargs)

    core_full = pd.Series(0.0, index=tickers)
    core_full[list(mandatory)] = normalized_core(mandatory)
    st_core = portfolio_stats(core_full.values, Sigma, tickers)
    st_port = portfolio_stats(w.values, Sigma, tickers)

    r_core = R @ core_full
    r_port = R @ w
    tail = r_core <= r_core.quantile(TAIL_Q)
    tail_stats = {'core': float(r_core[tail].mean()), 'port': float(r_port[tail].mean()),
                  'n': int(tail.sum())}

    block_of = {}
    try:
        res = _order_by_clustering(R.corr(), residual=True)
        if res['blocks'] is not None:
            ids = {v: i + 1 for i, v in enumerate(dict.fromkeys(res['blocks']))}
            block_of = {t: ids[b] for t, b in zip(res['corr'].index, res['blocks'])}
    except Exception:
        pass

    sel = [t for t in tickers if w[t] > 1e-9]
    table = pd.DataFrame({
        'Ticker': sel,
        'Rol': ['Imprescindible' if t in mandatory else 'Complemento' for t in sel],
        'Peso (%)': [w[t] * 100 for t in sel],
        'Contribución al riesgo (%)': [st_port['rc'][t] * 100 for t in sel],
        'Tipo': [ticker_type(t) for t in sel],
        'Bloque': [block_of.get(t) for t in sel],
    }).sort_values(['Rol', 'Peso (%)'], ascending=[True, False]).reset_index(drop=True)
    # imprescindibles primero
    table = pd.concat([table[table['Rol'] == 'Imprescindible'], table[table['Rol'] != 'Imprescindible']],
                      ignore_index=True)

    result = {
        'ann': ann, 'freq': freq, 'n_obs': len(R), 'first': R.index[0], 'last': R.index[-1],
        'tickers': tickers, 'dropped': dropped, 'shrink_k': k,
        'weights': w, 'table': table, 'mandatory': mandatory, 'objective': objective,
        'vol_core': st_core['vol'], 'vol_port': st_port['vol'],
        'dr_core': st_core['dr'], 'dr_port': st_port['dr'],
        'tail': tail_stats,
        'ranking': diversifier_ranking(R, mandatory, ann, blend=blend),
        'corr_sel': R[sel].corr(),
        'oos': None, 'oos_summary': None, 'oos_fails': 0, 'notes': notes, 'blend': blend,
    }
    if oos:
        est_p = int(round(est_years * ann))
        reb_p = int(round(reb_months * (21 if freq == 'D' else 4.345)))
        if len(R) > est_p + reb_p:
            df, fails = walk_forward(R, mandatory, opt_kwargs, est_p, reb_p, ann, shrink)
            result['oos'], result['oos_fails'] = df, fails
            result['oos_summary'] = oos_summary(df, ann)
        else:
            notes.append("La ventana no alcanza para la validación fuera de muestra: "
                         "alargá la ventana o acortá la ventana de estimación.")
    return result


# ================= GRÁFICOS =================
def _fig_weights(table):
    colors = [COLOR_CORE if r == 'Imprescindible' else COLOR_COMP for r in table['Rol']]
    t = table.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=t['Ticker'], x=t['Peso (%)'], orientation='h', name='Peso',
                marker_color=[c for c in colors[::-1]], text=[f"{v:.1f}%" for v in t['Peso (%)']],
                textposition='outside')
    fig.add_bar(y=t['Ticker'], x=t['Contribución al riesgo (%)'], orientation='h', name='Contribución al riesgo',
                marker_color='rgba(224,224,255,0.35)')
    fig.update_layout(barmode='group', title="Peso y contribución al riesgo por activo",
                      xaxis_title="%", height=max(320, 34 * len(t) + 120), legend_orientation='h',
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


def _fig_corr(corr):
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, '#ff2040'], [0.5, '#12121f'], [1, '#00aa30']],
        texttemplate="%{z:.2f}", textfont=dict(size=10), colorbar=dict(title='ρ')))
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title="Correlaciones entre los activos de la cartera",
                      height=max(360, 40 * len(corr) + 140), margin=dict(l=10, r=10, t=60, b=10))
    return fig


def _fig_oos(df):
    fig = go.Figure()
    styles = {'Cartera optimizada': dict(color=COLOR_COMP, width=3),
              'Solo imprescindibles': dict(color=COLOR_CORE, width=2),
              'Igual ponderación': dict(color='rgba(224,224,255,0.5)', width=1.5, dash='dot')}
    for c in df.columns:
        fig.add_scatter(x=df.index, y=(df[c] - 1) * 100, mode='lines', name=c, line=styles.get(c))
    fig.update_layout(title="Validación fuera de muestra: retorno acumulado (%)", yaxis_title="%",
                      height=420, legend_orientation='h', margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ================= INTERFAZ STREAMLIT =================
def _stretch(fn, obj, **kw):
    """Llama a st.dataframe / st.plotly_chart a ancho completo con el parámetro que acepte la versión."""
    try:
        return fn(obj, width="stretch", **kw)
    except Exception:
        return fn(obj, use_container_width=True, **kw)


def _parse_tickers(text):
    return [t.strip().upper() for t in text.replace(';', ',').split(',') if t.strip()]


def render_portfolio_ui(groups):
    """Sección 'Formación de cartera' de la app. `groups` = lista de {'title':..., 'tickers':[...]}."""
    st.subheader("Formación de cartera con restricciones")
    st.caption("Elegí los instrumentos que sí o sí querés tener y la herramienta busca qué sumar, y en qué "
               "proporción, para reducir el riesgo según las correlaciones históricas. "
               "Es una herramienta de análisis, no una recomendación de inversión.")

    titles = ["(ninguno)"] + [g["title"] for g in groups]
    default = titles.index("ETFs que tienen CEDEAR") if "ETFs que tienen CEDEAR" in titles else 0
    group_sel = st.selectbox("Universo (grupo predefinido)", titles, index=default, key="pf_group")
    manual = st.text_input("Tickers adicionales (coma)", "", key="pf_manual")
    universe = []
    if group_sel != "(ninguno)":
        universe += next(g for g in groups if g["title"] == group_sel)["tickers"]
    universe += _parse_tickers(manual)
    universe = list(dict.fromkeys(t.upper() for t in universe))
    if len(universe) < 4:
        st.info("Elegí un universo de al menos 4 tickers.")
        return

    must = st.multiselect("Imprescindibles (los que sí o sí querés tener)", universe, key="pf_must")
    mandatory = {}
    if must:
        cols = st.columns(min(len(must), 4))
        default_w = float(round(50 / len(must)))
        for i, t in enumerate(must):
            with cols[i % len(cols)]:
                mandatory[t] = st.number_input(f"Peso de {t} (%)", min_value=0.0, max_value=100.0,
                                               value=default_w, step=1.0, key=f"pf_w_{t}") / 100.0

    with st.expander("Ajustes", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = st.slider("Ventana de datos (años)", 1, 10, 5, key="pf_years")
            freq_label = st.radio("Frecuencia de retornos", ["Diaria", "Semanal"], horizontal=True, key="pf_freq",
                                  help="Con ETFs internacionales (EWJ, EWZ, FXI, EEM...) la frecuencia diaria "
                                       "subestima la correlación por diferencias de horario de cierre; la "
                                       "semanal lo corrige pero tiene menos observaciones.")
            shrink = st.checkbox("Suavizar la covarianza (Ledoit-Wolf)", value=True, key="pf_shrink")
        with c2:
            objective_label = st.selectbox("Objetivo", list(OBJECTIVES), key="pf_obj")
            max_w = st.slider("Máximo por complemento (%)", 5, 100, 25, key="pf_maxw") / 100.0
            exact = st.checkbox("Fijar el peso de los imprescindibles exacto (si no, es un mínimo)",
                                value=True, key="pf_exact")
        with c3:
            cap_inv = st.slider("Tope total de inversos/volatilidad (%)", 0, 30, 0, key="pf_capinv",
                                help="VXX, SH, PSQ y similares. 0 = excluidos como complemento.") / 100.0
            cap_lev = st.slider("Tope total de apalancados (%)", 0, 30, 0, key="pf_caplev",
                                help="TQQQ, SPXL y similares. 0 = excluidos como complemento.") / 100.0
            min_pos = st.slider("Posición mínima de un complemento (%)", 0, 10, 2, key="pf_minpos") / 100.0
            max_assets = st.slider("Máximo de activos en total (0 = sin límite)", 0, 30, 0, key="pf_maxassets")
        c4, c5, c6 = st.columns(3)
        with c4:
            blend = st.slider("Peso de prueba para el ranking (%)", 5, 30, 10, key="pf_blend") / 100.0
        with c5:
            do_oos = st.checkbox("Validar fuera de muestra", value=True, key="pf_oos")
        with c6:
            est_years = st.slider("Ventana de estimación (años)", 1, 8, 2, key="pf_est", disabled=not do_oos)
            reb_months = st.selectbox("Rebalanceo (meses)", [1, 3, 6], index=1, key="pf_reb", disabled=not do_oos)

    if st.button("Armar cartera", key="pf_go"):
        if not mandatory:
            st.warning("Elegí al menos un imprescindible.")
        else:
            freq = 'D' if freq_label == "Diaria" else 'W'
            try:
                with st.spinner("Descargando precios y optimizando..."):
                    close = download_prices(tuple(universe), years)
                    if close is None or close.empty:
                        raise PortfolioError("No se pudieron descargar precios.")
                    R_raw = to_returns(close, freq)
                    res = analyze(R_raw, mandatory, objective=OBJECTIVES[objective_label], max_w=max_w,
                                  exact=exact, cap_inverse=cap_inv, cap_leveraged=cap_lev, min_pos=min_pos,
                                  max_assets=max_assets, shrink=shrink, freq=freq, blend=blend, oos=do_oos,
                                  est_years=est_years, reb_months=reb_months)
                res['objective_label'] = objective_label
                st.session_state['pf_result'] = res
            except PortfolioError as e:
                st.session_state.pop('pf_result', None)
                st.error(str(e))

    res = st.session_state.get('pf_result')
    if res:
        _show_results(res)


def _show_results(res):
    st.markdown(f"**Resultado** · objetivo: {res.get('objective_label', res['objective'])} · "
                f"{res['n_obs']} observaciones ({res['first']:%d/%m/%Y} – {res['last']:%d/%m/%Y}) · "
                f"suavizado: {res['shrink_k']:.1%}")
    for n in res['notes']:
        st.warning(n)

    red = (1 - res['vol_port'] / res['vol_core']) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volatilidad anual del núcleo solo", f"{res['vol_core'] * 100:.1f}%")
    m2.metric("Volatilidad anual de la cartera", f"{res['vol_port'] * 100:.1f}%", f"{-red:.1f}%",
              delta_color="inverse")
    m3.metric("Ratio de diversificación", f"{res['dr_port']:.2f}", f"{res['dr_port'] - res['dr_core']:+.2f}")
    m4.metric("Activos en la cartera", f"{len(res['table'])}")

    t = res['tail']
    st.caption(f"En el 5% de los períodos más malos del núcleo ({t['n']} observaciones) el retorno medio fue "
               f"{t['core'] * 100:.2f}% para el núcleo solo y {t['port'] * 100:.2f}% para la cartera "
               f"(en la muestra con la que se optimizó).")

    _stretch(
        st.dataframe, res['table'], hide_index=True,
        column_config={
            'Peso (%)': st.column_config.NumberColumn(format="%.1f"),
            'Contribución al riesgo (%)': st.column_config.NumberColumn(format="%.1f"),
            'Bloque': st.column_config.NumberColumn(format="%d", help="Bloque del agrupamiento jerárquico "
                                                    "(sin factor de mercado): complementos de otro bloque "
                                                    "que el núcleo aportan más diversificación."),
        })
    st.download_button("Descargar pesos (CSV)", res['table'].to_csv(index=False).encode('utf-8'),
                       file_name="cartera_pesos.csv", mime="text/csv", key="pf_dl")

    _stretch(st.plotly_chart, _fig_weights(res['table']))

    st.markdown("#### Ranking de diversificadores")
    st.caption(f"Candidatos ordenados por cuánto bajaría la volatilidad del núcleo si les sumás {res['blend']:.0%}. "
               "Los inversos/volatilidad y apalancados aparecen marcados: su correlación diaria sobreestima "
               "su utilidad como cobertura a plazos más largos.")
    rk = res['ranking']
    if rk.empty:
        st.info("No hay candidatos en el universo.")
    else:
        _stretch(st.dataframe, rk.round(2), hide_index=True)

    if len(res['corr_sel']) >= 2:
        _stretch(st.plotly_chart, _fig_corr(res['corr_sel']))

    if res['oos'] is not None:
        st.markdown("#### Validación fuera de muestra")
        _stretch(st.plotly_chart, _fig_oos(res['oos']))
        _stretch(st.dataframe, res['oos_summary'].round(2), hide_index=True)
        if res['oos_fails']:
            st.caption(f"En {res['oos_fails']} rebalanceo(s) el optimizador no encontró solución y se mantuvo "
                       "la cartera anterior.")

    with st.expander("Limitaciones a tener en cuenta"):
        st.markdown(
            "- Las correlaciones pasadas **no son estables**: en las crisis casi todo sube junto. "
            "Mirá la validación fuera de muestra y los peores días del núcleo, no solo la volatilidad estimada.\n"
            "- **Minimizar varianza no es cubrir un riesgo concreto.** Definí primero contra qué querés cubrirte.\n"
            "- Se trabaja con retornos en **dólares del subyacente**: no se ve el CCL, los ratios de los CEDEARs, "
            "la liquidez ni las comisiones.\n"
            "- Los inversos, de volatilidad y apalancados tienen decaimiento: por eso quedan afuera por defecto.\n"
            "- La herramienta ignora retornos esperados: una cartera de mínimo riesgo puede tener poco retorno."
        )
