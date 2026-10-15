"""
Formación de cartera con restricciones (MTaurus – Chart Lab).

Idea: elegís (opcional) los instrumentos que sí o sí querés tener, con su peso, y la herramienta busca qué
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


def reference_weights(tickers, mandatory):
    """
    Cartera de referencia contra la que se mide la mejora: el núcleo (imprescindibles, normalizados a 100%)
    o, si no hay imprescindibles, igual ponderación del universo sin inversos ni apalancados.
    """
    tickers = list(tickers)
    if mandatory:
        return normalized_core(mandatory).reindex(tickers).fillna(0.0)
    elig = [t for t in tickers if t not in INVERSE_GROUP and t not in LEVERAGED_GROUP] or tickers
    w = pd.Series(0.0, index=tickers)
    w[elig] = 1.0 / len(elig)
    return w


def diversifier_ranking(R, ref_w, ann, blend=0.10, tail_q=TAIL_Q, exclude=()):
    """
    Para cada candidato (salvo los de `exclude`): correlación con la referencia `ref_w` (Series de pesos),
    peso de mínima varianza en la mezcla referencia+candidato, reducción de volatilidad con ese peso y con
    `blend`, y retorno medio del candidato en los peores períodos de la referencia.
    """
    ref = ref_w[ref_w > 0]
    ref = ref / ref.sum()
    r_core = R[list(ref.index)] @ ref
    vc = float(r_core.var(ddof=1))
    tail = r_core <= r_core.quantile(tail_q)
    vol_core = float(np.sqrt(vc * ann))

    def mix_vol(a, vj, cov):
        return float(np.sqrt(max((1 - a) ** 2 * vc + a ** 2 * vj + 2 * a * (1 - a) * cov, 0) * ann))

    rows = []
    for t in R.columns:
        if t in exclude:
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
            'Correlación con la referencia': rho,
            'Vol. anual del activo (%)': np.sqrt(vj * ann) * 100,
            f'Vol. con {blend:.0%} (%)': mix_vol(blend, vj, cov) * 100,
            f'Reducción de vol. con {blend:.0%} (%)': (1 - mix_vol(blend, vj, cov) / vol_core) * 100,
            'Peso de mínima varianza (%)': a_star * 100,
            'Reducción de vol. con ese peso (%)': (1 - mix_vol(a_star, vj, cov) / vol_core) * 100,
            'Retorno medio en los peores días de la referencia (%)': float(r[tail].mean()) * 100,
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
def walk_forward(R, mandatory, opt_kwargs, est_periods, reb_periods, ann, shrink=True,
                 bench=None, bench_name='Benchmark'):
    """
    Cada `reb_periods` re-estima la covarianza con los últimos `est_periods`, optimiza y mantiene esos
    pesos (sin rebalanceo intermedio) durante el período siguiente.
    Devuelve (DataFrame con el valor acumulado de 'Cartera optimizada', 'Solo imprescindibles' (solo si hay
    imprescindibles), 'Igual ponderación' (sin inversos ni apalancados) y, si se pasa `bench` (Series de
    retornos alineada a R), 'Benchmark (<nombre>)'; más la cantidad de rebalanceos donde falló el optimizador).
    """
    tickers = list(R.columns)
    n = len(R)
    elig = [t for t in tickers if t not in INVERSE_GROUP and t not in LEVERAGED_GROUP]
    eq = pd.Series(0.0, index=tickers)
    eq[elig] = 1.0 / max(len(elig), 1)
    core = reference_weights(tickers, mandatory) if mandatory else None
    bname = f"Benchmark ({bench_name})"
    names = (['Cartera optimizada'] + (['Solo imprescindibles'] if mandatory else []) + ['Igual ponderación']
             + ([bname] if bench is not None else []))
    cur = dict.fromkeys(names, 1.0)
    out = {k: [] for k in names}
    idx = []
    fails = 0
    last = core.copy() if mandatory else eq.copy()
    for s in range(est_periods, n, reb_periods):
        window = R.iloc[s - est_periods:s]
        Sigma, _ = estimate_cov(window, shrink, ann)
        try:
            w = optimize_portfolio(Sigma, tickers, mandatory, **opt_kwargs)
            last = w
        except PortfolioError:
            fails += 1
            w = last
        strategies = {'Cartera optimizada': w, 'Igual ponderación': eq}
        if mandatory:
            strategies['Solo imprescindibles'] = core
        period = R.iloc[s:s + reb_periods]
        growth = (1 + period).cumprod().values
        growth_b = (1 + bench.iloc[s:s + reb_periods]).cumprod().values if bench is not None else None
        for name in names:
            if name == bname:
                vals = cur[name] * growth_b
            else:
                vals = cur[name] * (growth @ strategies[name].values)
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
            oos=True, est_years=2, reb_months=3, min_hist=0.8, bench=None, bench_name='Benchmark'):
    """
    R_raw: retornos (DataFrame, columnas = tickers del universo; puede tener NaN por historia corta).
    mandatory: dict ticker -> peso (fracción) de los imprescindibles (puede estar vacío: en ese caso la
               referencia es la igual ponderación del universo y todas las posiciones las elige la herramienta).
    bench: Series de retornos de un benchmark opcional (p. ej. SPY; no tiene que estar en el universo). Se usa solo
    como comparador adicional (volatilidad, correlación, beta y validación fuera de muestra).
    Devuelve un dict con pesos, estadísticas, ranking, correlaciones, validación fuera de muestra y avisos.
    """
    ann = 252 if freq == 'D' else 52
    notes = []
    mandatory = {t: float(w) for t, w in mandatory.items()}
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

    core_full = reference_weights(tickers, mandatory)   # núcleo, o igual ponderación si no hay imprescindibles
    st_core = portfolio_stats(core_full.values, Sigma, tickers)
    st_port = portfolio_stats(w.values, Sigma, tickers)

    r_core = R @ core_full
    r_port = R @ w
    tail = r_core <= r_core.quantile(TAIL_Q)
    tail_stats = {'core': float(r_core[tail].mean()), 'port': float(r_port[tail].mean()),
                  'n': int(tail.sum())}

    bench_info, bench_aligned = None, None
    if bench is not None:
        b = bench.reindex(R.index)
        if b.notna().mean() < 0.95:
            notes.append(f"El benchmark {bench_name} no tiene datos para al menos el 95% de las fechas de la "
                         "ventana: se omite la comparación.")
        else:
            bench_aligned = b.fillna(0.0)
            var_b = float(bench_aligned.var(ddof=1))

            def _rel(r):
                cov = float(np.cov(r.values, bench_aligned.values, ddof=1)[0, 1])
                var = float(r.var(ddof=1))
                return (cov / np.sqrt(var_b * var) if var_b > 0 and var > 0 else np.nan,
                        cov / var_b if var_b > 0 else np.nan)
            c_p, b_p = _rel(r_port)
            c_c, b_c = _rel(r_core)
            bench_info = {'name': bench_name, 'vol': float(np.sqrt(var_b * ann)),
                          'corr': c_p, 'beta': b_p, 'corr_core': c_c, 'beta_core': b_c}

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
        'Rol': ['Imprescindible' if t in mandatory else ('Complemento' if mandatory else 'Posición') for t in sel],
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
        'has_core': bool(mandatory), 'bench': bench_info,
        'vol_core': st_core['vol'], 'vol_port': st_port['vol'],
        'dr_core': st_core['dr'], 'dr_port': st_port['dr'],
        'tail': tail_stats,
        'ranking': diversifier_ranking(R, core_full, ann, blend=blend, exclude=set(mandatory)),
        'corr_sel': R[sel].corr(),
        'oos': None, 'oos_summary': None, 'oos_fails': 0, 'notes': notes, 'blend': blend,
    }
    if oos:
        est_p = int(round(est_years * ann))
        reb_p = int(round(reb_months * (21 if freq == 'D' else 4.345)))
        if len(R) > est_p + reb_p:
            df, fails = walk_forward(R, mandatory, opt_kwargs, est_p, reb_p, ann, shrink,
                                     bench=bench_aligned, bench_name=bench_name)
            result['oos'], result['oos_fails'] = df, fails
            result['oos_params'] = (est_years, reb_months)
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
        fig.add_scatter(x=df.index, y=(df[c] - 1) * 100, mode='lines', name=c,
                        line=styles.get(c) or dict(color='#ff7aa8', width=2, dash='dash'))
    fig.update_layout(title="Validación fuera de muestra: retorno acumulado (%)", yaxis_title="%",
                      height=420, legend_orientation='h', margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ================= INTERFAZ STREAMLIT =================
OBJECTIVE_HELP = {
    "Mínima varianza": (
        "Busca la combinación que **menos oscila** en conjunto. Es la opción más directa si tu meta es "
        "reducir el riesgo. Suele cargar mucho en los activos más tranquilos y menos parecidos entre sí."),
    "Paridad de riesgo entre los complementos": (
        "Reparte el **riesgo** (no el dinero) en partes iguales entre los activos que la herramienta agrega: "
        "los más movidos reciben menos peso. Da una cartera más equilibrada y menos concentrada."),
    "Máxima diversificación": (
        "Busca que los activos se **compensen entre sí** lo más posible (maximiza el ratio de diversificación). "
        "Se parece a mínima varianza, pero premia a los activos que se mueven distinto del resto."),
}

HOW_IT_WORKS = """
**La pregunta que responde esta herramienta:** dado un grupo de ETFs, ¿cómo combinarlos para que la cartera
suba y baje lo menos posible? Acá "riesgo" significa **volatilidad**: cuánto oscila el valor.

1. **Elegís el universo:** el grupo de ETFs entre los que puede elegir (por defecto, los que tienen CEDEAR).
2. **Opcional, marcás imprescindibles:** instrumentos que querés tener sí o sí, con el peso que les quieras dar
   (por ejemplo EWJ y XLE). Si no marcás ninguno, la herramienta arma la cartera de menor riesgo con todo el universo.
3. **La herramienta mira el pasado:** mide cuánto oscila cada ETF y cuánto se mueven juntos (la *correlación*).
   Dos activos que suben y bajan a la vez no se compensan; dos que se mueven distinto, sí.
4. **Busca qué sumar y en qué proporción**, respetando los límites que fijes (máximo por activo, cantidad de
   activos, etc.).
5. **Te muestra el resultado** y, además, cómo le habría ido a este método en el pasado con datos que no usó
   para decidir (*validación fuera de muestra*).

**Lo que NO hace:** no predice qué va a subir, no es una recomendación de inversión y no ve el dólar CCL, los
ratios de los CEDEARs, la liquidez ni las comisiones (trabaja con precios en dólares del ETF original).
"""

GLOSSARY = """
- **Volatilidad:** cuánto suele subir y bajar el valor en un año, medido en %. Una volatilidad de 10% significa
  que en un año "normal" el valor se mueve del orden de ±10%. En una crisis puede ser mucho más.
- **Correlación:** de -1 a +1. Con +1 dos activos se mueven siempre juntos; con 0 no tienen relación; con -1 se
  mueven en sentido contrario. Para diversificar conviene combinar activos con correlación baja o negativa.
- **Diversificar:** combinar activos que no se mueven igual, para que las caídas de unos se compensen con otros.
- **Ratio de diversificación:** compara el riesgo de los activos por separado con el de la cartera. 1 = no hay
  beneficio; 2 = la cartera es la mitad de riesgosa que el promedio de sus partes.
- **Contribución al riesgo:** qué parte del riesgo total aporta cada activo. Un activo que pesa 25% pero aporta
  35% del riesgo es más riesgoso de lo que su tamaño sugiere.
- **Máxima caída (drawdown):** la mayor baja desde un máximo hasta el mínimo siguiente. Mide cuánto "dolió" lo peor.
- **Suavizado de la covarianza (Ledoit-Wolf):** las correlaciones medidas con pocos datos tienen ruido y el
  optimizador tiende a creérselas demasiado. Este ajuste las "afloja" hacia un valor neutro.
- **Rebalanceo:** volver a armar la cartera cada cierto tiempo con datos nuevos.
- **Bloque:** grupo de ETFs que se comportan parecido (agrupamiento jerárquico, descontando el factor común de
  mercado). Complementos de un bloque distinto al de tu núcleo suelen diversificar más.
- **Inversos / volatilidad (SH, PSQ, VXX):** suben cuando el mercado baja. En la correlación diaria parecen
  coberturas perfectas, pero pierden valor con el tiempo y no sirven para mantener meses.
- **Apalancados (TQQQ, SPXL):** multiplican los movimientos diarios y se desgastan en mercados laterales.
- **Benchmark:** una referencia contra la cual medirte, por ejemplo el S&P 500 (SPY). Sirve para saber si tu
  cartera se mueve parecido al mercado o distinto, y cómo le fue en comparación.
- **Beta:** cuánto tiende a moverse la cartera cuando se mueve el benchmark. Beta 0,6 = si el benchmark cae 10%,
  la cartera tiende a caer cerca de 6%. Beta baja = depende menos del benchmark (no es una garantía en crisis).
- **Fuera de muestra:** datos que el método no vio al decidir. Es la prueba más honesta de si funciona.
"""


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
    st.markdown("Armá una cartera de ETFs que **oscile lo menos posible**, con o sin instrumentos que quieras "
                "tener sí o sí. Es una herramienta de análisis, no una recomendación de inversión.")
    with st.expander("¿Cómo funciona? Leer primero", expanded=False):
        st.markdown(HOW_IT_WORKS)

    titles = ["(ninguno)"] + [g["title"] for g in groups]
    default = titles.index("ETFs que tienen CEDEAR") if "ETFs que tienen CEDEAR" in titles else 0
    group_sel = st.selectbox("Universo (grupo predefinido)", titles, index=default, key="pf_group",
                             help="Conjunto de ETFs entre los que la herramienta puede elegir.")
    manual = st.text_input("Tickers adicionales (coma)", "", key="pf_manual",
                           help="Para sumar al universo otros tickers de Yahoo Finance, separados por coma. "
                                "Ejemplo: AGG, TLT.")
    universe = []
    if group_sel != "(ninguno)":
        universe += next(g for g in groups if g["title"] == group_sel)["tickers"]
    universe += _parse_tickers(manual)
    universe = list(dict.fromkeys(t.upper() for t in universe))
    if len(universe) < 4:
        st.info("Elegí un universo de al menos 4 tickers.")
        return

    must = st.multiselect(
        "Imprescindibles (opcional)", universe, key="pf_must",
        help="Instrumentos que querés tener sí o sí. Podés dejarlo vacío: en ese caso la herramienta arma la "
             "cartera de menor riesgo con todo el universo.")
    mandatory = {}
    if must:
        cols = st.columns(min(len(must), 4))
        default_w = float(round(50 / len(must)))
        for i, t in enumerate(must):
            with cols[i % len(cols)]:
                mandatory[t] = st.number_input(
                    f"Peso de {t} (%)", min_value=0.0, max_value=100.0, value=default_w, step=1.0,
                    key=f"pf_w_{t}", help="Porcentaje de la cartera que querés destinar a este instrumento. "
                                          "El resto lo completa la herramienta.") / 100.0
    else:
        st.caption("Sin imprescindibles: la herramienta busca la combinación de menor riesgo entre todos los "
                   "ETFs del universo, respetando los límites de abajo.")

    with st.expander("Ajustes (opcionales)", expanded=False):
        st.markdown("**Datos**")
        c1, c2, c3 = st.columns(3)
        with c1:
            years = st.slider("Ventana de datos (años)", 1, 10, 5, key="pf_years",
                              help="Cuántos años de historia se usan para medir riesgo y correlaciones. Más años "
                                   "dan estimaciones más estables, pero mezclan épocas distintas del mercado.")
        with c2:
            freq_label = st.radio(
                "Frecuencia de retornos", ["Diaria", "Semanal"], horizontal=True, key="pf_freq",
                help="Con qué frecuencia se miden las subas y bajas. Con ETFs de otros países (EWJ, EWZ, FXI, "
                     "EEM...) la diaria subestima lo parecidos que son, porque cierran a otra hora. La semanal "
                     "lo corrige, pero tiene menos observaciones.")
        with c3:
            shrink = st.checkbox(
                "Suavizar la covarianza (Ledoit-Wolf)", value=True, key="pf_shrink",
                help="Las correlaciones medidas con pocos datos tienen ruido y el optimizador tiende a creérselas "
                     "demasiado. El suavizado corrige ese exceso de confianza. Recomendado dejarlo tildado.")

        st.markdown("**Cómo se arma la cartera**")
        objective_label = st.selectbox("Objetivo", list(OBJECTIVES), key="pf_obj")
        st.caption(OBJECTIVE_HELP[objective_label])
        c1, c2, c3 = st.columns(3)
        with c1:
            max_w = st.slider(
                "Máximo por complemento (%)" if mandatory else "Máximo por activo (%)", 5, 100, 25, key="pf_maxw",
                help="Tope de peso para cada activo que agrega la herramienta (no aplica a los imprescindibles). "
                     "Evita que toda la cartera termine en uno o dos ETFs.") / 100.0
        with c2:
            min_pos = st.slider(
                "Posición mínima (%)", 0, 10, 2, key="pf_minpos",
                help="Evita posiciones insignificantes: si un activo recibiría menos que esto, se descarta y la "
                     "cartera se recalcula.") / 100.0
        with c3:
            max_assets = st.slider(
                "Máximo de activos en total (0 = sin límite)", 0, 30, 0, key="pf_maxassets",
                help="Cantidad máxima de ETFs distintos en la cartera, incluyendo los imprescindibles. "
                     "Útil si no querés manejar muchas posiciones.")
        exact = st.checkbox(
            "Fijar el peso de los imprescindibles exacto (si no, es un mínimo)", value=True, key="pf_exact",
            disabled=not mandatory,
            help="Tildado: los imprescindibles tienen exactamente el peso que pusiste. Destildado: ese peso es "
                 "un mínimo y la herramienta puede aumentarlo si conviene, hasta el máximo por activo.")

        st.markdown("**Instrumentos especiales**")
        c1, c2 = st.columns(2)
        with c1:
            cap_inv = st.slider(
                "Tope total de inversos/volatilidad (%)", 0, 30, 0, key="pf_capinv",
                help="SH, PSQ y VXX suben cuando el mercado baja, y en la correlación diaria parecen coberturas "
                     "perfectas. Pero pierden valor con el tiempo y no sirven para mantener meses. Por eso "
                     "quedan en 0% (excluidos). Si querés permitirlos, fijá un tope para el total.") / 100.0
        with c2:
            cap_lev = st.slider(
                "Tope total de apalancados (%)", 0, 30, 0, key="pf_caplev",
                help="TQQQ y SPXL multiplican los movimientos diarios y se desgastan en mercados laterales. "
                     "Quedan en 0% (excluidos) salvo que fijes un tope.") / 100.0

        st.markdown("**Análisis adicional**")
        bench_txt = st.text_input(
            "Benchmark de comparación (opcional)", "SPY", key="pf_bench",
            help="Un ticker de Yahoo Finance contra el cual comparar la cartera (no hace falta que esté en el "
                 "universo). Ejemplos: SPY (S&P 500), QQQ (Nasdaq 100), ACWI (acciones globales). Se muestra su "
                 "volatilidad, la correlación y la beta de tu cartera respecto de él, y su línea en la validación "
                 "fuera de muestra. Vacío = sin benchmark. No modifica la cartera: es solo para comparar.")
        c1, c2, c3 = st.columns(3)
        with c1:
            blend = st.slider(
                "Peso de prueba para el ranking (%)", 5, 30, 10, key="pf_blend",
                help="Para el ranking de diversificadores: qué porción de la cartera se le destinaría a cada "
                     "candidato en la prueba (por ejemplo, el 10%).") / 100.0
        with c2:
            do_oos = st.checkbox(
                "Validar fuera de muestra", value=True, key="pf_oos",
                help="Simula cómo habría funcionado este método en el pasado, con datos que no usó para decidir. "
                     "Es la prueba más honesta. Recomendado.")
        with c3:
            est_years = st.slider(
                "Ventana de estimación (años)", 1, 8, 2, key="pf_est", disabled=not do_oos,
                help="En la simulación, cuántos años hacia atrás mira el método cada vez que arma la cartera.")
            reb_months = st.selectbox(
                "Rebalanceo (meses)", [1, 3, 6], index=1, key="pf_reb", disabled=not do_oos,
                help="En la simulación, cada cuántos meses se vuelve a armar la cartera.")

    if st.button("Armar cartera", key="pf_go"):
        freq = 'D' if freq_label == "Diaria" else 'W'
        try:
            with st.spinner("Descargando precios y optimizando..."):
                close = download_prices(tuple(universe), years)
                if close is None or close.empty:
                    raise PortfolioError("No se pudieron descargar precios.")
                R_raw = to_returns(close, freq)
                bench, bench_name, bench_msg = None, None, None
                b_tk = _parse_tickers(bench_txt)[:1]
                if b_tk:
                    bench_name = b_tk[0]
                    bclose = download_prices((bench_name,), years)
                    if bclose is None or bclose.empty or bench_name not in bclose.columns:
                        bench_msg = f"No se pudo descargar el benchmark {bench_name}: se omite la comparación."
                    else:
                        bench = to_returns(bclose[[bench_name]], freq)[bench_name]
                res = analyze(R_raw, mandatory, objective=OBJECTIVES[objective_label], max_w=max_w,
                              exact=exact, cap_inverse=cap_inv, cap_leveraged=cap_lev, min_pos=min_pos,
                              max_assets=max_assets, shrink=shrink, freq=freq, blend=blend, oos=do_oos,
                              est_years=est_years, reb_months=reb_months,
                              bench=bench, bench_name=bench_name or 'Benchmark')
                if bench_msg:
                    res['notes'].append(bench_msg)
            res['objective_label'] = objective_label
            st.session_state['pf_result'] = res
        except PortfolioError as e:
            st.session_state.pop('pf_result', None)
            st.error(str(e))

    res = st.session_state.get('pf_result')
    if res:
        _show_results(res)


def _ranking_config(blend):
    """Ayuda por columna del ranking de diversificadores."""
    b = f"{blend:.0%}"
    num = st.column_config.NumberColumn
    return {
        'Ticker': st.column_config.TextColumn(help="ETF candidato."),
        'Tipo': st.column_config.TextColumn(
            help="Normal, Inverso / volatilidad o Apalancado. Estos dos últimos pierden valor con el tiempo: "
                 "su utilidad como cobertura de largo plazo es menor de lo que muestra la correlación."),
        'Correlación con la referencia': num(
            format="%.2f", help="Qué tan parecido se mueve respecto de la cartera de referencia. Cerca de 0 o "
                                "negativo = se compensa bien; cerca de 1 = se mueve igual, no diversifica."),
        'Vol. anual del activo (%)': num(format="%.1f", help="Cuánto oscila el activo por sí solo en un año."),
        f'Vol. con {b} (%)': num(format="%.1f", help=f"Volatilidad anual de la referencia si le sumás {b} de este activo."),
        f'Reducción de vol. con {b} (%)': num(
            format="%.1f", help=f"Cuánto baja el riesgo de la referencia al destinarle {b} a este activo. "
                                "Cuanto más alto, mejor diversificador (la tabla está ordenada por esta columna)."),
        'Peso de mínima varianza (%)': num(
            format="%.1f", help="Mezclando solo la referencia con este activo, el porcentaje de este activo que "
                                "deja el riesgo más bajo posible."),
        'Reducción de vol. con ese peso (%)': num(
            format="%.1f", help="Cuánto baja el riesgo con ese peso óptimo (el mejor caso para este activo solo)."),
        'Retorno medio en los peores días de la referencia (%)': num(
            format="%.2f", help="Qué hizo este activo, en promedio, en el 5% de los períodos en que a la "
                                "referencia peor le fue. Positivo = amortiguó las caídas."),
    }


def _show_results(res):
    has_core = res.get('has_core', bool(res['mandatory']))
    ref_name = ("tus imprescindibles solos (el núcleo)" if has_core else
                "una cartera con todos los ETFs del universo en partes iguales (sin inversos ni apalancados)")
    ref_short = "del núcleo solo" if has_core else "de igual ponderación"

    st.markdown(f"**Resultado** · objetivo: {res.get('objective_label', res['objective'])} · "
                f"{res['n_obs']} observaciones ({res['first']:%d/%m/%Y} – {res['last']:%d/%m/%Y})")
    for n in res['notes']:
        st.warning(n)

    # ---- 1) Resumen ----
    vc, vp = res['vol_core'] * 100, res['vol_port'] * 100
    red = (1 - res['vol_port'] / res['vol_core']) * 100
    if red >= 0.5:
        st.success(f"Con esta cartera, la volatilidad anual estimada baja de **{vc:.1f}%** a **{vp:.1f}%** "
                   f"({red:.0f}% menos) respecto de {ref_name}.")
    elif red <= -0.5:
        st.info(f"Con estas restricciones la volatilidad estimada ({vp:.1f}%) queda por encima de la de {ref_name} "
                f"({vc:.1f}%). Probá aflojar los límites (máximo por activo, posición mínima o cantidad de activos).")
    else:
        st.info(f"La volatilidad estimada ({vp:.1f}%) prácticamente no cambia respecto de {ref_name} ({vc:.1f}%).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Volatilidad anual {ref_short}", f"{vc:.1f}%",
              help="Cuánto suele oscilar en un año la cartera de referencia con la que se compara. Más alto = "
                   "más sube y baja.")
    m2.metric("Volatilidad anual de la cartera", f"{vp:.1f}%", f"{-red:.1f}%", delta_color="inverse",
              help="Lo mismo, para la cartera armada. El porcentaje de abajo es la variación respecto de la "
                   "referencia (negativo = oscila menos).")
    m3.metric("Ratio de diversificación", f"{res['dr_port']:.2f}", f"{res['dr_port'] - res['dr_core']:+.2f}",
              help="1 = sin beneficio de diversificar (todo se mueve igual). 2 = la cartera es la mitad de "
                   "riesgosa que el promedio de sus partes. Más alto es mejor.")
    m4.metric("Activos en la cartera", f"{len(res['table'])}",
              help="Cantidad de ETFs distintos que quedaron con peso.")

    t = res['tail']
    st.caption(
        f"**En los peores momentos:** tomamos el 5% de los períodos en que a la referencia peor le fue "
        f"({t['n']} observaciones). El retorno medio fue {t['core'] * 100:.2f}% para la referencia y "
        f"{t['port'] * 100:.2f}% para la cartera. Ojo: se mide con los mismos datos con los que se armó la cartera, "
        f"así que es optimista; la validación fuera de muestra (más abajo) es la prueba más honesta.")

    bn = res.get('bench')
    if bn:
        st.markdown(f"**Comparación con el benchmark ({bn['name']})**")
        b1, b2, b3 = st.columns(3)
        b1.metric(f"Volatilidad anual de {bn['name']}", f"{bn['vol'] * 100:.1f}%",
                  help="Cuánto oscila el benchmark en un año, para comparar con la volatilidad de la cartera.")
        b2.metric("Correlación de la cartera con el benchmark", f"{bn['corr']:.2f}",
                  f"{bn['corr'] - bn['corr_core']:+.2f}", delta_color="off",
                  help="De -1 a +1: qué tan parecido se mueve tu cartera al benchmark. El número de abajo es el "
                       "cambio respecto de la referencia (el núcleo o la igual ponderación).")
        b3.metric("Beta de la cartera", f"{bn['beta']:.2f}", f"{bn['beta'] - bn['beta_core']:+.2f}",
                  delta_color="off",
                  help="Cuánto tiende a moverse la cartera cuando se mueve el benchmark. El número de abajo es el "
                       "cambio respecto de la referencia.")
        ej = 10 * bn['beta']
        st.caption(
            f"**Cómo leerlo:** una beta de {bn['beta']:.2f} significa que, históricamente, cuando {bn['name']} cayó "
            f"10%, esta cartera tendió a caer cerca de {ej:.0f}%. Con beta y correlación bajas, la cartera depende "
            "menos de ese mercado. Son promedios del pasado: en una crisis fuerte las correlaciones suelen subir.")

    # ---- 2) Qué comprar ----
    st.markdown("#### Qué comprar y en qué proporción")
    st.caption("Estos son los pesos que resultan de aplicar el objetivo y los límites elegidos. "
               "Los porcentajes de la columna *Peso* suman 100%.")
    _stretch(
        st.dataframe, res['table'], hide_index=True,
        column_config={
            'Ticker': st.column_config.TextColumn(help="Instrumento."),
            'Rol': st.column_config.TextColumn(
                help="Imprescindible: lo elegiste vos. Complemento: lo agregó la herramienta. "
                     "Posición: lo eligió la herramienta (no marcaste imprescindibles)."),
            'Peso (%)': st.column_config.NumberColumn(
                format="%.1f", help="Porcentaje de tu dinero que iría a este instrumento."),
            'Contribución al riesgo (%)': st.column_config.NumberColumn(
                format="%.1f", help="Qué parte del riesgo total de la cartera aporta este instrumento. "
                                    "Si es mucho mayor que su peso, es más riesgoso que su tamaño."),
            'Tipo': st.column_config.TextColumn(
                help="Normal, Inverso / volatilidad o Apalancado (estos dos últimos se desgastan con el tiempo)."),
            'Bloque': st.column_config.NumberColumn(
                format="%d", help="Grupo de ETFs que se comportan parecido. Complementos de un bloque distinto "
                                  "al de tu núcleo suelen diversificar más."),
        })
    st.download_button("Descargar pesos (CSV)", res['table'].to_csv(index=False).encode('utf-8'),
                       file_name="cartera_pesos.csv", mime="text/csv", key="pf_dl")
    _stretch(st.plotly_chart, _fig_weights(res['table']))
    st.caption("Cada activo tiene dos barras: cuánto pesa en la cartera y cuánto del riesgo aporta. Si la barra de "
               "riesgo es mucho más larga que la de peso, ese activo mueve la cartera más de lo que su tamaño sugiere.")

    # ---- 3) Ranking ----
    st.markdown("#### Ranking de diversificadores")
    st.caption(
        f"Ordena los ETFs del universo según cuánto reducirían la oscilación de {ref_name} si les destinaras el "
        f"{res['blend']:.0%}. Sirve para ver **qué ayuda más** y por qué, aunque no haya entrado en la cartera final. "
        "Pasá el mouse por el título de cada columna para ver qué significa. Los inversos/volatilidad y apalancados "
        "aparecen marcados: su correlación diaria sobreestima su utilidad como cobertura a plazos más largos.")
    rk = res['ranking']
    if rk.empty:
        st.info("No hay candidatos en el universo.")
    else:
        _stretch(st.dataframe, rk.round(2), hide_index=True, column_config=_ranking_config(res['blend']))

    # ---- 4) Correlaciones ----
    if len(res['corr_sel']) >= 2:
        st.markdown("#### Correlaciones entre los activos de la cartera")
        st.caption("Cada celda muestra cuánto se mueven juntos dos activos: **verde** (hacia +1) se mueven igual, "
                   "**rojo** (hacia -1) en sentido contrario, **oscuro** (cerca de 0) sin relación clara. "
                   "Una cartera diversificada tiene muchos valores bajos fuera de la diagonal.")
        _stretch(st.plotly_chart, _fig_corr(res['corr_sel']))

    # ---- 5) Fuera de muestra ----
    if res['oos'] is not None:
        _show_oos(res, has_core)

    with st.expander("Glosario: qué significa cada término"):
        st.markdown(GLOSSARY)
    with st.expander("Limitaciones a tener en cuenta"):
        st.markdown(
            "- Las correlaciones pasadas **no son estables**: en las crisis casi todo sube junto. "
            "Mirá la validación fuera de muestra y los peores días, no solo la volatilidad estimada.\n"
            "- **Minimizar el riesgo no es cubrir un riesgo concreto.** Definí primero contra qué querés cubrirte.\n"
            "- Se trabaja con retornos en **dólares del subyacente**: no se ve el CCL, los ratios de los CEDEARs, "
            "la liquidez ni las comisiones.\n"
            "- Los inversos, de volatilidad y apalancados tienen decaimiento: por eso quedan afuera por defecto.\n"
            "- La herramienta **ignora los retornos esperados**: una cartera de mínimo riesgo puede rendir poco."
        )


def _show_oos(res, has_core):
    est, reb = res.get('oos_params', (None, None))
    bn_name = (res.get('bench') or {}).get('name')
    st.markdown("#### Validación fuera de muestra: ¿habría funcionado en el pasado?")
    how = (f"cada {reb} meses, usando solo los {est} años anteriores" if est else "cada cierto tiempo, usando solo "
           "los años anteriores")
    st.markdown(
        "**Para qué sirve.** Si probás un método con los mismos datos con los que lo diseñaste, siempre se ve bien: "
        "es como estudiar con el examen resuelto. Esta es la prueba real. La herramienta simula que, en el pasado, "
        f"armó la cartera {how}, la mantuvo hasta la siguiente revisión y repitió. De esa forma se ve si la "
        "reducción de riesgo **se sostiene con datos que no conocía**.")
    _stretch(st.plotly_chart, _fig_oos(res['oos']))
    st.caption("Cada línea muestra cuánto habría ganado o perdido (en %) quien siguiera esa estrategia: "
               "la **cartera optimizada** (la de esta herramienta)"
               + (", **solo imprescindibles** (el núcleo sin nada más)" if has_core else "")
               + " y **igual ponderación** (el mismo dinero en cada ETF del universo, sin inversos ni "
                 "apalancados)"
               + (f", más el **benchmark** {bn_name} que elegiste (línea punteada)" if bn_name else "")
               + ". La optimizada busca menos oscilación, no más retorno: es normal que rinda menos "
                 "que una apuesta concentrada en lo que más subió.")
    summ = res['oos_summary']
    _stretch(st.dataframe, summ.round(2), hide_index=True, column_config={
        'Retorno acumulado (%)': st.column_config.NumberColumn(help="Ganancia o pérdida total en todo el período de prueba."),
        'Retorno anualizado (%)': st.column_config.NumberColumn(help="Retorno acumulado expresado como tasa anual."),
        'Volatilidad anual (%)': st.column_config.NumberColumn(help="Cuánto oscilaron los resultados. Menos es mejor si buscás estabilidad."),
        'Máxima caída (%)': st.column_config.NumberColumn(help="La peor baja desde un máximo hasta el mínimo siguiente. Menos negativa es mejor."),
    })

    ref_row = 'Solo imprescindibles' if has_core else 'Igual ponderación'
    try:
        a = summ[summ['Estrategia'] == 'Cartera optimizada'].iloc[0]
        b = summ[summ['Estrategia'] == ref_row].iloc[0]
        period = f"{res['oos'].index[0]:%m/%Y} a {res['oos'].index[-1]:%m/%Y}"
        txt = (f"**Cómo leerlo:** en el período de prueba ({period}) la cartera optimizada tuvo una volatilidad de "
               f"{a['Volatilidad anual (%)']:.1f}% frente a {b['Volatilidad anual (%)']:.1f}% de "
               f"{ref_row.lower()}, y una máxima caída de {a['Máxima caída (%)']:.1f}% frente a "
               f"{b['Máxima caída (%)']:.1f}%. ")
        if a['Volatilidad anual (%)'] < b['Volatilidad anual (%)']:
            txt += "La menor oscilación **se sostuvo fuera de muestra**, que es lo que se buscaba. "
        else:
            txt += "La menor oscilación **no se sostuvo** fuera de muestra: tomá con cautela el resultado de arriba. "
        txt += (f"En retorno anualizado: {a['Retorno anualizado (%)']:.1f}% frente a "
                f"{b['Retorno anualizado (%)']:.1f}%. ")
        bsel = summ[summ['Estrategia'].astype(str).str.startswith('Benchmark')]
        if len(bsel):
            c_ = bsel.iloc[0]
            txt += (f"El benchmark ({bn_name}) tuvo una volatilidad de {c_['Volatilidad anual (%)']:.1f}%, una máxima "
                    f"caída de {c_['Máxima caída (%)']:.1f}% y un retorno anualizado de "
                    f"{c_['Retorno anualizado (%)']:.1f}%. ")
        txt += "Es un solo período histórico: no garantiza lo que pase después."
        st.info(txt)
    except Exception:
        pass
    if res['oos_fails']:
        st.caption(f"En {res['oos_fails']} rebalanceo(s) el optimizador no encontró solución y se mantuvo "
                   "la cartera anterior.")
