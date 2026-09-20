"""
Simulador de rotación entre dos activos según el ratio A/B (solo long, sin short).

Idea en una línea: si el ratio A/B se dispara (A caro vs B) se vende A y se compra B;
si se hunde (A barato vs B) se vende B y se compra A. Nunca se vende algo que no se tiene.
Se compara contra: buy & hold de A, buy & hold de B y buy & hold 50/50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TRADING_DAYS = 252

# Colores coherentes con el resto de la app (tema oscuro)
COLOR_BH_A = "#ffaa60"
COLOR_BH_B = "#c080ff"
COLOR_BH_50 = "#a0a0a0"
STRATEGY_COLORS = ["#40c0ff", "#60d070", "#ff7070", "#ffe066", "#ff8ad8", "#7ee8d4", "#b0b0ff", "#ffa0a0", "#90ee90"]


# ===================== LÓGICA =====================

def compute_zscore(ratio: pd.Series, window: int):
    """
    Devuelve (z, media_móvil, desvío_móvil).
    z = cuántos desvíos estándar está el ratio de hoy respecto de su promedio reciente.
    Usa SOLO datos pasados (ventana móvil), así no hay "trampa" de mirar el futuro.
    """
    mean = ratio.rolling(window).mean()
    std = ratio.rolling(window).std()
    z = (ratio - mean) / std.replace(0, np.nan)
    return z, mean, std


def simulate_rotation(
    price_a: pd.Series,
    price_b: pd.Series,
    z: pd.Series,
    k: float,
    capital: float = 100_000.0,
    cash_pct: float = 0.0,
    fee_pct: float = 0.0,
    cash_rate_annual_pct: float = 0.0,
    start_mode: str = "signal",
):
    """
    Simula la estrategia de rotación long-only.

    Reglas:
      - Arranca comprando A o B (start_mode: 'A', 'B' o 'signal' = el que indique el ratio).
      - Si estoy en A y z >= +k  -> A está caro vs B  -> rotar a B.
      - Si estoy en B y z <= -k  -> A está barato vs B -> rotar a A.
      - La señal se detecta al cierre de un día y se ejecuta al cierre del día siguiente.
      - cash_pct: parte de la cartera que se deja en cash en cada rotación (0 = todo invertido).
      - fee_pct: comisión (%) sobre lo comprado/vendido de acciones en cada operación.

    Devuelve dict con 'equity' (Serie), 'trades' (DataFrame), 'fees' (float), 'n_rotations' (int).
    """
    idx = price_a.index
    pa = price_a.values.astype(float)
    pb = price_b.values.astype(float)
    zv = z.reindex(idx).values.astype(float)

    valid = np.where(~np.isnan(zv))[0]
    if len(valid) == 0:
        return None
    i0 = int(valid[0])

    fee = fee_pct / 100.0
    cash_w = cash_pct / 100.0
    cash_daily = (1.0 + cash_rate_annual_pct / 100.0) ** (1.0 / TRADING_DAYS) - 1.0

    va = vb = 0.0
    vc = float(capital)
    fees_paid = 0.0
    held = None  # activo en cartera realmente ('A' o 'B')
    trades = []

    def rebalance(target: str):
        nonlocal va, vb, vc, fees_paid, held
        total = va + vb + vc
        w_asset = 1.0 - cash_w
        new_a = total * w_asset if target == "A" else 0.0
        new_b = total * w_asset if target == "B" else 0.0
        traded = abs(new_a - va) + abs(new_b - vb)
        cost = traded * fee
        fees_paid += cost
        total -= cost
        # reparto del total neto (ya descontada la comisión) respetando los pesos
        va = total * w_asset if target == "A" else 0.0
        vb = total * w_asset if target == "B" else 0.0
        vc = total - va - vb
        held = target

    # Estado objetivo (lo que "quiero" tener) y orden pendiente de ejecutar
    if start_mode == "A":
        target = "A"
    elif start_mode == "B":
        target = "B"
    else:
        target = "B" if zv[i0] > 0 else "A"

    n = len(idx)
    equity = np.empty(n - i0)
    pending = None
    n_rot = 0

    for j, i in enumerate(range(i0, n)):
        if j == 0:
            rebalance(target)
            trades.append((idx[i], idx[i], f"Compra inicial de {target}", pa[i] / pb[i], zv[i], va + vb + vc))
        else:
            va *= pa[i] / pa[i - 1]
            vb *= pb[i] / pb[i - 1]
            vc *= 1.0 + cash_daily
            if pending is not None:
                p_target, p_date, p_z = pending
                if p_target != held:
                    rebalance(p_target)
                    n_rot += 1
                    trades.append((p_date, idx[i], f"Vende {'B' if p_target == 'A' else 'A'} y compra {p_target}",
                                   pa[i] / pb[i], p_z, va + vb + vc))
                pending = None

        zi = zv[i]
        if target == "A" and zi >= k:
            target = "B"
            pending = ("B", idx[i], zi)
        elif target == "B" and zi <= -k:
            target = "A"
            pending = ("A", idx[i], zi)

        equity[j] = va + vb + vc

    eq = pd.Series(equity, index=idx[i0:], name=f"Rotar ±{k:g}σ")
    trades_df = pd.DataFrame(trades, columns=["Fecha señal", "Fecha", "Operación", "Ratio", "Z (desvíos)", "Valor de la cartera"])
    return {"equity": eq, "trades": trades_df, "fees": fees_paid, "n_rotations": n_rot}


def buy_and_hold(price_a: pd.Series, price_b: pd.Series, start_idx, capital: float,
                 weight_a: float, fee_pct: float = 0.0):
    """Compra al inicio (weight_a en A, el resto en B) y no toca más. Devuelve (equity, comisión)."""
    pa = price_a.loc[start_idx:]
    pb = price_b.loc[start_idx:]
    fee_cost = capital * fee_pct / 100.0
    net = capital - fee_cost
    eq = net * (weight_a * pa / pa.iloc[0] + (1.0 - weight_a) * pb / pb.iloc[0])
    return eq, fee_cost


def compute_metrics(equity: pd.Series, capital: float, n_rotations: int | None, fees: float) -> dict:
    final = float(equity.iloc[-1])
    drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "Valor final": final,
        "Ganancia / pérdida": final - capital,
        "Retorno %": (final / capital - 1.0) * 100.0,
        "Peor caída %": drawdown * 100.0,
        "Rotaciones": n_rotations,
        "Comisiones pagadas": fees,
    }


def run_full_simulation(price_a, price_b, ratio, window, sigmas, capital, cash_pct, fee_pct,
                        cash_rate_annual_pct, start_mode, name_a, name_b):
    """Corre todas las estrategias y benchmarks. Devuelve (curves, metrics_df, trades_by_k, z, mean, std)."""
    z, mean, std = compute_zscore(ratio, window)
    first_valid = z.first_valid_index()
    if first_valid is None:
        return None

    curves, rows, trades_by_k = {}, {}, {}

    # Benchmarks (empiezan el mismo día que las estrategias para que la comparación sea justa)
    eq_a, fee_a = buy_and_hold(price_a, price_b, first_valid, capital, 1.0, fee_pct)
    eq_b, fee_b = buy_and_hold(price_a, price_b, first_valid, capital, 0.0, fee_pct)
    eq_50, fee_50 = buy_and_hold(price_a, price_b, first_valid, capital, 0.5, fee_pct)
    bench = [
        (f"Buy & hold {name_a}", eq_a, fee_a),
        (f"Buy & hold {name_b}", eq_b, fee_b),
        ("Buy & hold 50/50", eq_50, fee_50),
    ]
    for name, eq, fee in bench:
        curves[name] = eq
        rows[name] = compute_metrics(eq, capital, 0, fee)

    # Estrategias de rotación
    for k in sigmas:
        res = simulate_rotation(price_a, price_b, z, k, capital, cash_pct, fee_pct,
                                cash_rate_annual_pct, start_mode)
        if res is None:
            continue
        name = f"Rotar a ±{k:g}σ"
        curves[name] = res["equity"]
        rows[name] = compute_metrics(res["equity"], capital, res["n_rotations"], res["fees"])
        trades_by_k[k] = res["trades"]

    metrics_df = pd.DataFrame(rows).T
    return {
        "curves": curves,
        "metrics": metrics_df,
        "trades": trades_by_k,
        "z": z, "mean": mean, "std": std,
        "start": first_valid,
        "bench_names": [b[0] for b in bench],
    }


# ===================== GRÁFICOS =====================

def _base_layout(fig, title, ytitle, height=650):
    fig.update_layout(
        title=title,
        yaxis=dict(title=dict(text=ytitle, font=dict(color="#d0d0ff")), showgrid=True,
                   gridcolor="rgba(120,120,140,0.28)", tickfont=dict(color="#d0d0ff")),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(35,35,55,0.65)", bordercolor="rgba(90,90,130,0.45)", borderwidth=1),
        template="plotly_dark", height=height,
        plot_bgcolor="#0f0f1a", paper_bgcolor="#05050f", font=dict(color="#e0e0ff"),
        hovermode="x unified",
    )
    fig.add_annotation(text="MTaurus - X: @mtaurus_ok", xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=42, color="rgba(140,140,180,0.14)"), textangle=-30)
    return fig


def build_equity_figure(sim: dict, capital: float, name_a: str, name_b: str) -> go.Figure:
    fig = go.Figure()
    bench_style = {
        f"Buy & hold {name_a}": (COLOR_BH_A, "dash"),
        f"Buy & hold {name_b}": (COLOR_BH_B, "dash"),
        "Buy & hold 50/50": (COLOR_BH_50, "dash"),
    }
    strat_names = [n for n in sim["curves"] if n not in bench_style]
    for i, name in enumerate(strat_names):
        eq = sim["curves"][name]
        fig.add_trace(go.Scatter(
            x=eq.index, y=(eq / capital - 1) * 100, mode="lines", name=name,
            line=dict(color=STRATEGY_COLORS[i % len(STRATEGY_COLORS)], width=2.2),
            customdata=eq.values, hovertemplate="%{y:.1f}%  (valor: %{customdata:,.0f})",
        ))
    for name, (color, dash) in bench_style.items():
        eq = sim["curves"][name]
        fig.add_trace(go.Scatter(
            x=eq.index, y=(eq / capital - 1) * 100, mode="lines", name=name,
            line=dict(color=color, width=1.8, dash=dash),
            customdata=eq.values, hovertemplate="%{y:.1f}%  (valor: %{customdata:,.0f})",
        ))
    fig.add_hline(y=0, line=dict(color="rgba(200,200,220,0.35)", width=1))
    return _base_layout(fig, f"Resultado acumulado: rotar {name_a}/{name_b} vs quedarse quieto",
                        "Ganancia o pérdida acumulada (%)")


def build_signals_figure(ratio: pd.Series, sim: dict, k: float, name_a: str, name_b: str) -> go.Figure:
    start = sim["start"]
    r = ratio.loc[start:]
    mean = sim["mean"].loc[start:]
    std = sim["std"].loc[start:]
    upper, lower = mean + k * std, mean - k * std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=r, mode="lines", name=f"Ratio {name_a}/{name_b}",
                             line=dict(color="#40c0ff", width=2)))
    fig.add_trace(go.Scatter(x=mean.index, y=mean, mode="lines", name="Promedio reciente",
                             line=dict(color="#a0a0ff", width=1.3, dash="dash")))
    fig.add_trace(go.Scatter(x=upper.index, y=upper, mode="lines", name=f"+{k:g}σ (zona 'A caro')",
                             line=dict(color="#ff7070", width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=lower.index, y=lower, mode="lines", name=f"-{k:g}σ (zona 'A barato')",
                             line=dict(color="#60d070", width=1.3, dash="dot")))

    tr = sim["trades"][k]
    tr = tr[tr["Operación"].str.startswith(("Vende", "Compra inicial"))]
    to_a = tr[tr["Operación"].str.endswith("A")]
    to_b = tr[tr["Operación"].str.endswith("B")]
    fig.add_trace(go.Scatter(x=to_a["Fecha"], y=to_a["Ratio"], mode="markers", name=f"Pasa a {name_a}",
                             marker=dict(symbol="triangle-up", size=13, color="#60d070", line=dict(width=1, color="white"))))
    fig.add_trace(go.Scatter(x=to_b["Fecha"], y=to_b["Ratio"], mode="markers", name=f"Pasa a {name_b}",
                             marker=dict(symbol="triangle-down", size=13, color="#ff7070", line=dict(width=1, color="white"))))
    return _base_layout(fig, f"Puntos de rotación con umbral ±{k:g}σ", "Ratio", height=600)


# ===================== TEXTO EN LENGUAJE SIMPLE =====================

def plain_summary(sim: dict, name_a: str, name_b: str) -> str:
    m = sim["metrics"]
    bench = sim["bench_names"]
    strat = [n for n in m.index if n not in bench]
    best_all = m["Retorno %"].idxmax()
    worst_all = m["Retorno %"].idxmin()
    best_bh = m.loc[bench, "Retorno %"].idxmax()
    best_bh_ret = m.loc[best_bh, "Retorno %"]

    beat = [n for n in strat if m.loc[n, "Retorno %"] > best_bh_ret]
    lines = [
        f"- **La que mejor terminó:** {best_all} ({m.loc[best_all, 'Retorno %']:+.1f}%).",
        f"- **La que peor terminó:** {worst_all} ({m.loc[worst_all, 'Retorno %']:+.1f}%).",
        f"- **El mejor 'quedarse quieto' fue** {best_bh} ({best_bh_ret:+.1f}%).",
    ]
    if not strat:
        pass
    elif beat:
        lines.append(f"- **Rotar le ganó al mejor 'quedarse quieto' con:** {', '.join(beat)}.")
    else:
        lines.append("- **Ninguna versión de rotar le ganó al mejor 'quedarse quieto'** en este período.")
    return "\n".join(lines)


EXPLANATION_MD = """
### ¿Qué hace este simulador?
Responde una pregunta: **¿me convenía ir cambiando de un activo a otro según el ratio, o me convenía comprar uno y no tocarlo?**

**1. ¿Qué es el ratio?**
Es una división: precio de A ÷ precio de B. Si sube, A se encareció *respecto de B*. Si baja, A se abarató *respecto de B*.

**2. ¿Cuál es la idea de la estrategia?**
Muchas veces el ratio de dos activos parecidos oscila alrededor de un valor "normal". La apuesta es:
- Cuando el ratio está **muy alto** (A demasiado caro vs B): **vendés A y comprás B**.
- Cuando el ratio está **muy bajo** (A demasiado barato vs B): **vendés B y comprás A**.

**3. ¿Qué es "σ" (desvío estándar)?**
Es la medida de cuánto se suele alejar el ratio de su promedio. Es la "regla" para decidir qué es *muy alto* o *muy bajo*:
- **±1σ:** reacciona ante alejamientos chicos → **más rotaciones** (más operaciones, más comisiones).
- **±2σ:** solo reacciona ante alejamientos grandes → **pocas rotaciones**, pero puede quedarse mucho tiempo sin hacer nada.

Como no se sabe cuál umbral es mejor, el simulador prueba **varios a la vez** y los compara.

**4. Sin short (como en Argentina)**
Nunca se vende algo que no se tiene. Siempre se arranca **comprando** uno de los dos, y cada rotación es *vender lo que tengo para comprar el otro*.

**5. ¿Y el cash?**
Opcionalmente podés dejar una parte de la plata **sin invertir** (en cash) cada vez que rotás. Menos riesgo, pero también te perdés parte de las subas. Esto solo afecta a las estrategias de rotación: los "quedarse quieto" siempre están 100% invertidos.

**6. Contra qué se compara ("quedarse quieto" = buy & hold)**
- Comprar solo A el primer día y no tocar más.
- Comprar solo B el primer día y no tocar más.
- Comprar mitad A y mitad B el primer día y no tocar más.

**7. ¿Cómo leo los resultados?**
- **Gráfico de arriba:** cuánto ganaste o perdiste (%) en cada momento. Cuanto más arriba termina una línea, mejor.
- **Tabla:** valor final, ganancia/pérdida, **peor caída** (el peor momento: cuánto llegó a caer desde un máximo antes de recuperarse) y cantidad de rotaciones.
- **Gráfico de puntos de rotación:** el ratio con sus bandas, y triángulos donde la estrategia cambió de activo.

**⚠️ Ojo con esto**
- Es una **simulación con datos pasados**: no garantiza nada a futuro.
- El resultado depende mucho del período elegido y de si los activos realmente "vuelven" a su relación normal (a veces el ratio se va y no vuelve).
- La señal se detecta al cierre de un día y se opera al **cierre del día siguiente**. No incluye impuestos ni diferencias entre precio de compra y venta (spread), solo la comisión que cargues abajo.
- Los precios están ajustados por dividendos y splits.
"""


# ===================== INTERFAZ STREAMLIT =====================

def render_ratio_simulator(price_a: pd.Series, price_b: pd.Series, name_a: str, name_b: str):
    st.divider()
    st.header(f"🔄 Simulador: ¿conviene rotar entre {name_a} y {name_b}?")
    st.caption("Sin short: siempre se compra uno de los dos y, cuando la señal lo indica, se vende para comprar el otro.")

    with st.expander("📖 Explicación paso a paso (leer si es la primera vez)", expanded=False):
        st.markdown(EXPLANATION_MD)

    ratio = (price_a / price_b).dropna()

    st.subheader("Configuración")
    c1, c2 = st.columns(2)
    with c1:
        sigmas = st.multiselect(
            "Umbrales a probar (en desvíos estándar σ)",
            options=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
            default=[1.0, 1.5, 2.0],
            key="rbt_sigmas",
            help="Cuánto tiene que alejarse el ratio de su promedio para rotar. Chico = rota seguido; "
                 "grande = rota poco. Se prueban todos los elegidos y se comparan entre sí.",
        )
        window = st.number_input(
            "Días para calcular el 'promedio normal' del ratio",
            min_value=10, max_value=max(11, min(500, len(ratio) - 10)), value=min(60, max(10, len(ratio) // 4)),
            step=5, key="rbt_window",
            help="El ratio se compara contra su promedio de estos últimos días hábiles (60 ≈ 3 meses). "
                 "Ventana corta = se adapta rápido; larga = referencia más estable. "
                 "Solo se usan datos pasados, nunca futuros.",
        )
        start_options = ["El que indique el ratio ese día", f"Siempre {name_a}", f"Siempre {name_b}"]
        start_label = st.selectbox(
            "¿Con qué activo arranca la estrategia?",
            start_options,
            key="rbt_start",
            help="Como no se puede shortear, hay que empezar comprando uno. 'El que indique el ratio': "
                 "si A está caro arranca en B; si está barato, arranca en A.",
        )
    with c2:
        capital = st.number_input(
            "Capital inicial (moneda de los tickers)", min_value=1_000.0, value=100_000.0, step=10_000.0,
            key="rbt_capital", help="Plata con la que arranca cada estrategia. Sirve para expresar la ganancia en plata.",
        )
        fee_pct = st.number_input(
            "Comisión por operación (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="rbt_fee",
            help="Se cobra sobre lo que se compra y se vende en cada rotación. Todos los 'quedarse quieto' "
                 "pagan la comisión de su compra inicial. Con 0% se ignoran los costos.",
        )
        use_cash = st.checkbox(
            "Permitir dejar una parte en cash", value=False, key="rbt_use_cash",
            help="Cada vez que rota, la estrategia deja este porcentaje sin invertir.",
        )

    cash_pct, cash_rate = 0.0, 0.0
    if use_cash:
        cc1, cc2 = st.columns(2)
        with cc1:
            cash_pct = st.slider(
                "% de la cartera que queda en cash", 5, 95, 30, 5, key="rbt_cash_pct",
                help="Ejemplo: 30% → en cada rotación, 70% va al activo elegido y 30% queda en efectivo.",
            )
        with cc2:
            cash_rate = st.number_input(
                "Rendimiento anual del cash (%)", min_value=-50.0, max_value=200.0, value=0.0, step=1.0,
                key="rbt_cash_rate", help="0% = la plata en cash no rinde nada. Poné una tasa si la tuvieras en una caución o similar.",
            )

    if not sigmas:
        st.info("Elegí al menos un umbral para correr la simulación.")
        return
    sigmas = sorted(sigmas)

    start_mode = ["signal", "A", "B"][start_options.index(start_label)]

    sim = run_full_simulation(price_a, price_b, ratio, int(window), sigmas, float(capital), float(cash_pct),
                              float(fee_pct), float(cash_rate), start_mode, name_a, name_b)
    if sim is None:
        st.warning("No hay datos suficientes para la ventana elegida. Probá con una ventana más corta o un año de inicio más antiguo.")
        return

    st.caption(f"La simulación arranca el **{sim['start'].date()}** (los primeros {int(window)} días se usan para calcular el promedio del ratio) "
               f"y termina el **{ratio.index[-1].date()}**. Todas las estrategias y los 'quedarse quieto' arrancan ese mismo día.")

    st.subheader("Resultado")
    st.plotly_chart(build_equity_figure(sim, float(capital), name_a, name_b), use_container_width=True)
    st.markdown(plain_summary(sim, name_a, name_b))

    st.subheader("Tabla comparativa")
    m = sim["metrics"].copy()
    disp = pd.DataFrame(index=m.index)
    disp["Valor final"] = m["Valor final"].map(lambda v: f"{v:,.0f}")
    disp["Ganancia / pérdida"] = m["Ganancia / pérdida"].map(lambda v: f"{v:+,.0f}")
    disp["Retorno"] = m["Retorno %"].map(lambda v: f"{v:+.1f}%")
    disp["Peor caída"] = m["Peor caída %"].map(lambda v: f"{v:.1f}%")
    disp["Rotaciones"] = m["Rotaciones"].astype(int)
    disp["Comisiones"] = m["Comisiones pagadas"].map(lambda v: f"{v:,.0f}")
    st.dataframe(disp, use_container_width=True)
    st.caption("**Peor caída:** lo máximo que llegó a caer la cartera desde un pico antes de recuperarse (cuanto más cerca de 0%, más tranquila la ruta). "
               "**Rotaciones:** cuántas veces se vendió un activo para comprar el otro (en 'quedarse quieto' es 0).")
    if use_cash:
        st.caption(f"Las estrategias de rotación dejan {cash_pct}% en cash en cada rotación; los 'quedarse quieto' están 100% invertidos.")

    st.subheader("¿Dónde rotó? (puntos sobre la curva del ratio)")
    k_view = st.selectbox("Ver los puntos de rotación del umbral:", sigmas,
                          format_func=lambda x: f"±{x:g}σ", key="rbt_k_view")
    st.plotly_chart(build_signals_figure(ratio, sim, k_view, name_a, name_b), use_container_width=True)
    st.caption(f"▲ verde: la estrategia pasó a {name_a} (el ratio estaba bajo). ▼ rojo: pasó a {name_b} (el ratio estaba alto).")

    with st.expander("Ver el detalle de cada operación"):
        tr = sim["trades"][k_view].copy()
        tr["Fecha señal"] = tr["Fecha señal"].dt.date
        tr["Fecha"] = tr["Fecha"].dt.date
        tr["Ratio"] = tr["Ratio"].map(lambda v: f"{v:.4f}")
        tr["Z (desvíos)"] = tr["Z (desvíos)"].map(lambda v: f"{v:+.2f}")
        tr["Valor de la cartera"] = tr["Valor de la cartera"].map(lambda v: f"{v:,.0f}")
        tr = tr.rename(columns={"Fecha": "Fecha de la operación", "Z (desvíos)": "Desvíos al dar la señal",
                                "Ratio": "Ratio al operar"})
        st.dataframe(tr, use_container_width=True, hide_index=True)
        st.caption("La señal se ve al cierre de la 'Fecha señal' y la operación se hace al cierre del día siguiente.")
