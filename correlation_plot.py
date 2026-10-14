import argparse
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mtaurus_style  # noqa: F401  (tipografía compartida)
import os
import sys
from datetime import datetime, timedelta

# ================= CONFIG - DARK THEME =================
BG_COLOR = '#0f0f1a'           # Main plot background
PAPER_COLOR = '#05050f'        # Figure background
TEXT_COLOR = '#e0e0ff'         # Soft light text
GRID_COLOR = '#2a2a3a'         # Very subtle grid
ANNOT_COLOR = '#ffffff'        # Bold annotations (Pure white for max contrast)
WATERMARK_COLOR = (140/255, 140/255, 180/255, 0.11)  # RGBA tuple

# Red (Negative) <-> Green (Positive), center dark so white text is readable
DARK_DIVERGING = sns.diverging_palette(
    h_neg=10,
    h_pos=130,
    s=100,
    l=55,
    sep=1,
    center='dark',
    as_cmap=True
)

YEARS_BACK = 5
OUTPUT_FILE = "correlation_temp.png"
# Salidas opcionales (se generan solo si se piden)
DENDROGRAM_FILE = "dendrogram_temp.png"
CLUSTERS_CSV_FILE = "clusters_temp.csv"

# Con más tickers que esto, los números se omiten por defecto (no se leen en un feed/celular)
ANNOT_MAX_TICKERS = 30
# Líneas que separan los bloques (clusters)
BLOCK_LINE_COLOR = (224/255, 224/255, 255/255, 0.9)
BLOCK_LINE_WIDTH = 2.2
# Líneas finas de subbloques (segundo nivel, anidado dentro de los bloques)
SUB_LINE_COLOR = (224/255, 224/255, 255/255, 0.55)
SUB_LINE_WIDTH = 1.2
# Colores de las etiquetas de ticker, alternados entre bloques consecutivos
# Cada bloque usa un par (tono pleno, tono apagado): el pleno y el apagado se alternan entre
# los subbloques de ese bloque; el par cambia entre bloques consecutivos
BLOCK_LABEL_COLORS = ((TEXT_COLOR, '#8f96d8'), ('#ffaa60', '#c97b3c'))
# Ramas del dendrograma por encima del corte de bloques
DENDRO_ABOVE_COLOR = '#6a6a8a'


def get_data_and_corr(tickers, start_year=None):
    end_date = datetime.now()
    if start_year:
        start_date = datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=YEARS_BACK * 365)

    valid_data = {}
    print(f"Downloading data for: {tickers} from {start_date.date()}")

    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                try:
                    series = df['Close'][t]
                except KeyError:
                    series = df['Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                series = df.iloc[:, 0]

            if not series.empty and len(series) > 50:
                valid_data[t] = series
            else:
                print(f"⚠️ Warning: Insufficient data for {t} (skipping)")

        except Exception as e:
            print(f"❌ Error downloading {t}: {e}")

    if len(valid_data) < 2:
        print("❌ Error: Not enough valid tickers to build correlation matrix.")
        return None

    df_all = pd.DataFrame(valid_data)
    returns = df_all.pct_change()
    corr_matrix = returns.corr(method='pearson', min_periods=30)

    return corr_matrix


def _remove_market_mode(c):
    """
    Quita el primer componente (factor de mercado común) de una matriz de correlación y la
    renormaliza. Se usa solo para agrupar: el gráfico sigue mostrando la correlación original.
    """
    w, v = np.linalg.eigh(c)            # autovalores en orden ascendente: el último es el mayor
    c_res = c - w[-1] * np.outer(v[:, -1], v[:, -1])
    d = np.sqrt(np.clip(np.diag(c_res), 1e-12, None))
    return np.clip(c_res / np.outer(d, d), -1, 1)


def _inter_rho(corr, labels):
    """
    Mayor correlación promedio entre dos grupos distintos (sobre la correlación original).
    Es el número real detrás de "grupos con correlación promedio de ~X o menos entre sí".
    """
    c = corr.values
    labels = np.asarray(labels)
    ids = np.unique(labels)
    best = -1.0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = np.nanmean(c[np.ix_(labels == a, labels == b)])
            if np.isfinite(m):
                best = max(best, float(m))
    return round(best, 2)


def _order_by_clustering(corr, n_blocks=None, n_sub=None, residual=False):
    """
    Reordena tickers por clustering jerárquico (los que se parecen quedan juntos) y define
    dos niveles de agrupación anidados: bloques (corte alto del árbol) y subbloques (corte más bajo).
    Devuelve un dict con:
      corr:       matriz reordenada
      blocks/sub: ids de grupo por ticker en el nuevo orden (None si ese nivel no se pidió)
      rho_blocks/rho_sub: mayor correlación promedio entre dos grupos distintos de ese nivel
      Z/order/names: linkage, orden de hojas y nombres originales (para el dendrograma)
      cut_blocks/cut_sub: altura (distancia) del corte de cada nivel
    n_blocks: None = automático (~n/7, entre 3 y 10); 0 = sin bloques.
    n_sub:    None/0 = sin subbloques; N = cantidad de subbloques (debe ser mayor que los bloques).
    residual: True = agrupar sobre la correlación sin el factor de mercado (el primer componente),
              para que aparezca la estructura interna (sectores, países, bonos, etc.).
    """
    out = {'corr': corr, 'blocks': None, 'sub': None, 'rho_blocks': None, 'rho_sub': None,
           'Z': None, 'order': None, 'names': list(corr.index),
           'cut_blocks': None, 'cut_sub': None}
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
    except ImportError:
        return out
    n = len(corr)
    c = corr.fillna(0).values
    if residual:
        c = _remove_market_mode(c)
    dist = np.sqrt(np.clip(0.5 * (1 - c), 0, None))   # corr 1 -> 0, corr -1 -> 1
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method='average')
    try:
        Z = optimal_leaf_ordering(Z, cond)             # hojas contiguas más parecidas entre sí
    except Exception:
        pass
    order = leaves_list(Z)
    out['corr'] = corr.iloc[order, order]
    out['Z'], out['order'] = Z, order

    def cut(k):
        labels = fcluster(Z, t=k, criterion='maxclust')[order]
        # altura del corte: entre la última fusión hecha y la primera deshecha
        h = 0.5 * (Z[n - 1 - k, 2] + Z[n - k, 2])
        return labels, _inter_rho(out['corr'], labels), float(h)

    if n_blocks is None:
        n_blocks = int(np.clip(round(n / 7), 3, 10))
    if n_blocks and 1 < n_blocks < n:
        out['blocks'], out['rho_blocks'], out['cut_blocks'] = cut(n_blocks)
        if n_sub and n_blocks < n_sub < n:
            out['sub'], out['rho_sub'], out['cut_sub'] = cut(n_sub)
    return out


def _label_colors(n, blocks, sub):
    """Color de etiqueta por ticker (en orden): dos colores alternados por bloque y, dentro de
    cada bloque, variación leve de tono entre subbloques consecutivos."""
    cols = []
    blk_idx, sub_idx = 0, 0
    for i in range(n):
        if i > 0 and blocks[i] != blocks[i - 1]:
            blk_idx += 1
            sub_idx = 0
        elif i > 0 and sub is not None and sub[i] != sub[i - 1]:
            sub_idx += 1
        cols.append(BLOCK_LABEL_COLORS[blk_idx % 2][sub_idx % 2])
    return cols


def _draw_boundaries(ax, bounds, n, upper_masked, **kw):
    """Dibuja líneas horizontales y verticales en las posiciones dadas (respeta el triángulo)."""
    for b in bounds:
        if upper_masked:
            ax.hlines(b, 0, b, **kw)
            ax.vlines(b, b, n, **kw)
        else:
            ax.axhline(b, **kw)
            ax.axvline(b, **kw)


def _short_fmt(v):
    """0.85 -> '.85', -0.85 -> '-.85', 1.00 -> '1' (más angosto, entra en celdas chicas)."""
    if pd.isna(v):
        return ""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    if s in ("1.00", "-1.00"):
        return s[:-3]
    return s


def _save_clusters_csv(res, path):
    """CSV ticker -> bloque / subbloque (numerados en orden de aparición) y correlación
    promedio dentro de cada bloque (sobre la correlación original)."""
    c = res['corr']
    df = pd.DataFrame({'ticker': list(c.index)})
    for col, lab in (('bloque', res['blocks']), ('subbloque', res['sub'])):
        if lab is not None:
            ids = {v: i + 1 for i, v in enumerate(dict.fromkeys(lab))}
            df[col] = [ids[v] for v in lab]
    if res['blocks'] is not None:
        lab = np.asarray(res['blocks'])
        m = c.values
        rho = {}
        for l in np.unique(lab):
            idx = np.where(lab == l)[0]
            if len(idx) < 2:
                rho[l] = np.nan
            else:
                off = m[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]
                rho[l] = round(float(np.nanmean(off)), 2)
        df['rho_prom_intrabloque'] = [rho[l] for l in lab]
    df.to_csv(path, index=False)
    return path


def plot_dendrogram(res, title, out_path, residual=False):
    """
    Dendrograma del clustering jerárquico en un PNG aparte, mismo estilo oscuro que la matriz.
    Los colores de ramas (bloques) y de tickers coinciden con los de la matriz de correlación;
    las líneas punteadas marcan los cortes de bloques y subbloques.
    """
    from scipy.cluster.hierarchy import dendrogram
    Z, order, names = res['Z'], res['order'], res['names']
    n = len(names)
    ordered = [names[i] for i in order]
    blocks, sub = res['blocks'], res['sub']

    if blocks is not None:
        label_cols = _label_colors(n, blocks, sub)
        # color de rama por bloque (tono pleno), según orden de aparición
        blk_color = {}
        for lab in blocks:
            if lab not in blk_color:
                blk_color[lab] = BLOCK_LABEL_COLORS[len(blk_color) % 2][0]
        blk_orig = np.empty(n, dtype=blocks.dtype)
        blk_orig[order] = blocks
        rep = list(range(n))                    # una hoja representativa por cada nodo
        for j in range(n - 1):
            rep.append(rep[int(Z[j, 0])])
        h_cut = res['cut_blocks']

        def link_color_func(k):
            if Z[k - n, 2] <= h_cut:
                return blk_color[blk_orig[rep[k]]]
            return DENDRO_ABOVE_COLOR
    else:
        label_cols = [TEXT_COLOR] * n

        def link_color_func(k):
            return '#8f96d8'
    color_of = dict(zip(ordered, label_cols))

    fig_w = 11
    fig_h = max(6.0, 0.2 * n + 2.8)
    fs = float(np.clip(fig_h * 72 * 0.8 / n * 0.7, 6, 11))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)

    dendrogram(Z, orientation='right', labels=names, link_color_func=link_color_func,
               leaf_font_size=fs, ax=ax)
    ax.invert_yaxis()                           # mismo orden de arriba hacia abajo que la matriz

    ax.tick_params(axis='both', colors=TEXT_COLOR)
    for lbl in ax.get_yticklabels():
        lbl.set_color(color_of.get(lbl.get_text(), TEXT_COLOR))
    for sp in ax.spines.values():
        sp.set_color(GRID_COLOR)
    ax.grid(axis='x', color=GRID_COLOR, lw=0.6)
    ax.set_axisbelow(True)

    if res['cut_blocks'] is not None:
        ax.axvline(res['cut_blocks'], color=BLOCK_LINE_COLOR, lw=1.4, ls='--')
    if res['cut_sub'] is not None:
        ax.axvline(res['cut_sub'], color=SUB_LINE_COLOR, lw=1.0, ls=':')

    xl = "Distancia  d = √(½·(1 − ρ))"
    if residual:
        xl += "   (sobre la correlación sin factor de mercado)"
    ax.set_xlabel(xl, color=TEXT_COLOR, fontsize=11, labelpad=10)
    title_pad = 14
    if not residual:
        # eje superior con la correlación equivalente (solo tiene sentido sobre la correlación original)
        sec = ax.secondary_xaxis(
            'top',
            functions=(lambda d: 1 - 2 * np.asarray(d, dtype=float) ** 2,
                       lambda r: np.sqrt(np.clip((1 - np.asarray(r, dtype=float)) / 2, 0, None))))
        sec.set_xlabel("Correlación equivalente (ρ)", color=TEXT_COLOR, fontsize=10, labelpad=6)
        sec.tick_params(colors=TEXT_COLOR)
        title_pad = 38
    ax.set_title(f"Dendrograma de Correlación: {title}\n(agrupamiento jerárquico, enlace promedio)",
                 fontsize=15, color=TEXT_COLOR, weight='bold', pad=title_pad)

    import textwrap
    note = ("Cómo leerlo: cada rama une ETFs o grupos; mientras más corta la rama (más a la izquierda), "
            "más parecido es su comportamiento. Línea punteada gruesa: corte de bloques; línea punteada "
            "fina: corte de subbloques. Los colores de las ramas y de los tickers coinciden con los "
            "bloques de la matriz de correlación.")
    if residual:
        note += " Los grupos se calcularon quitando el factor de mercado común."
    note = textwrap.fill(note, width=125)
    fig.text(0.03, 0.012, note, ha='left', va='bottom', fontsize=9, color=TEXT_COLOR, alpha=0.8,
             linespacing=1.4)
    n_lines = note.count("\n") + 1
    plt.tight_layout(pad=1.8, rect=(0, (0.35 + 0.17 * n_lines) / fig_h, 1, 1))

    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=44, color=WATERMARK_COLOR,
             ha='center', va='center', rotation=30, weight='bold')
    fig.text(0.98, 0.012, f"Generado: {datetime.now().strftime('%Y-%m-%d')}", ha='right', va='bottom',
             fontsize=9, color='gray', alpha=0.65)

    abs_path = os.path.abspath(out_path)
    fig.savefig(abs_path, dpi=200, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return abs_path


def plot_correlation(corr_matrix, title, cluster=None, mask_upper=None,
                     annotate=None, n_blocks=None, footnote=True, n_subblocks=None,
                     residual=False, dendrogram_path=None, csv_path=None):
    """
    footnote:   texto explicativo al pie sobre cómo leer el gráfico (default: True)
    annotate:   mostrar los números en cada celda (default: True si hay <=30 tickers)
    n_blocks:   bloques a marcar con líneas gruesas al reordenar por similitud
                (default: automático, 0 = sin líneas)
    n_subblocks: segundo nivel, más fino, marcado con líneas finas (default: None = sin subbloques;
                debe ser mayor que la cantidad de bloques)
    cluster:    reordenar por similitud (default: True si hay >12 tickers)
    mask_upper: mostrar solo el triángulo inferior (default: False, matriz cuadrada completa)
    residual:   agrupar sobre la correlación sin el factor de mercado (default: False). La matriz
                sigue mostrando la correlación original.
    dendrogram_path: si se indica, guarda además el dendrograma en ese PNG (default: None = no)
    csv_path:   si se indica, guarda un CSV ticker -> bloque/subbloque (default: None = no)
    El tamaño de la figura y de las fuentes escala con la cantidad de tickers.
    """
    n = len(corr_matrix)

    if cluster is None:
        cluster = n > 12
    blocks = sub = rho_blocks = rho_sub = None
    res = None
    if cluster or dendrogram_path or csv_path:
        res = _order_by_clustering(corr_matrix, n_blocks, n_subblocks, residual)
    if cluster:
        corr_matrix, blocks, sub = res['corr'], res['blocks'], res['sub']
        rho_blocks, rho_sub = res['rho_blocks'], res['rho_sub']
    if mask_upper is None:
        mask_upper = False  # matriz cuadrada completa por defecto

    # ---------- Tamaños adaptativos ----------
    fig_w = max(13, 0.42 * n + 3)
    fig_h = fig_w * 11 / 13
    scale = fig_w / 13
    cell_pt = fig_h * 0.78 * 72 / n                      # lado aprox. de una celda en puntos
    annot_size = float(np.clip(cell_pt * 0.34, 6, 11))
    # Etiquetas: hasta ~62% del lado de la celda (el texto vertical necesita entrar en una columna),
    # con tope proporcional al tamaño de la figura para no inflarlas con pocos tickers
    tick_size = float(np.clip(cell_pt * 0.62, 7, 11 * scale))
    dpi = 300 if n <= 25 else 200

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    # Anotaciones: con muchos tickers se omiten; con 21-30, formato corto sin cero inicial
    if annotate is None:
        annotate = n <= ANNOT_MAX_TICKERS
    if not annotate:
        annot = False
        fmt = ''
    elif n > 20:
        annot = corr_matrix.apply(lambda col: col.map(_short_fmt)).values
        fmt = ''
    else:
        annot = True
        fmt = '.2f'

    mask = np.triu(np.ones((n, n), dtype=bool), k=1) if mask_upper else None

    heatmap = sns.heatmap(
        corr_matrix,
        cmap=DARK_DIVERGING,
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        annot=annot,
        fmt=fmt,
        annot_kws={'size': annot_size, 'weight': 'bold', 'color': ANNOT_COLOR},
        cbar_kws={
            'label': 'Pearson Correlation',
            'shrink': 0.7,
            'pad': 0.07,  # espacio extra para las etiquetas del costado derecho
            'fraction': 0.046,
        },
        square=True,
        linewidths=0.8 if n <= 25 else 0.3,
        linecolor='#1a1a2e',
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Líneas finas entre subbloques (debajo) y gruesas entre bloques (encima)
    if sub is not None:
        sub_b = [i for i in range(1, n) if sub[i] != sub[i - 1]]
        _draw_boundaries(ax, sub_b, n, mask_upper, color=SUB_LINE_COLOR, lw=SUB_LINE_WIDTH)
    if blocks is not None:
        blk_b = [i for i in range(1, n) if blocks[i] != blocks[i - 1]]
        _draw_boundaries(ax, blk_b, n, mask_upper, color=BLOCK_LINE_COLOR, lw=BLOCK_LINE_WIDTH)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.set_label('Pearson Correlation', color=TEXT_COLOR, weight='bold')

    ax.set_title(
        f"Matriz de Correlación: {title}\n({YEARS_BACK} años o período seleccionado)",
        fontsize=min(19 * scale, 30), color=TEXT_COLOR, weight='bold', pad=25
    )

    # Tickers en los cuatro costados (abajo, arriba, izquierda, derecha)
    ax.tick_params(
        axis='both', colors=TEXT_COLOR, labelsize=tick_size,
        top=True, bottom=True, left=True, right=True,
        labeltop=True, labelbottom=True, labelleft=True, labelright=True,
    )
    for tick in ax.xaxis.get_major_ticks():
        if n > 25:
            for lbl in (tick.label1, tick.label2):
                lbl.set_rotation(90)
                lbl.set_ha('center')
        else:
            tick.label1.set_rotation(45)
            tick.label1.set_ha('right')
            tick.label1.set_rotation_mode('anchor')
            tick.label2.set_rotation(45)
            tick.label2.set_ha('left')
            tick.label2.set_rotation_mode('anchor')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label2.set_rotation(0)

    # Etiquetas coloreadas por bloque (dos colores alternados) y, dentro de cada bloque,
    # por subbloque (variación leve de tono entre subbloques consecutivos)
    if blocks is not None:
        label_cols = _label_colors(n, blocks, sub)
        for axis in (ax.xaxis, ax.yaxis):
            for i, tick in enumerate(axis.get_major_ticks()):
                tick.label1.set_color(label_cols[i])
                tick.label2.set_color(label_cols[i])

    ax.set_xlabel("Tickers", fontsize=12 * min(scale, 1.6), color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Tickers", fontsize=12 * min(scale, 1.6), color=TEXT_COLOR, labelpad=12)

    # Ancho ajustado al contenido real: con pocos tickers la matriz (cuadrada, limitada por el alto)
    # no llena la figura y quedaba aire a los costados. Se mide lo que ocupa todo (matriz, etiquetas,
    # título y barra de color) y se achica el ancho; el alto y el tamaño de las fuentes no cambian.
    from matplotlib.transforms import Bbox
    plt.tight_layout(pad=1.8)
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _content = Bbox.union([ax.get_tightbbox(_r), cbar.ax.get_tightbbox(_r)])
    _new_w = float(np.clip(_content.width / fig.dpi + 0.8, 9.5, fig_w))
    if _new_w < fig_w - 0.2:
        fig.set_size_inches(_new_w, fig_h)
        fig_w = _new_w

    fig.text(
        0.5, 0.5,
        "MTaurus - X: @MTaurus_ok",
        fontsize=58 * scale,
        color=WATERMARK_COLOR,
        ha='center', va='center',
        rotation=30,
        weight='bold'
    )

    fig.text(
        0.98, 0.015,
        f"Generado: {datetime.now().strftime('%Y-%m-%d')}",
        ha='right', va='bottom',
        fontsize=9 * min(scale, 1.6), color='gray', alpha=0.65
    )

    # Texto explicativo al pie
    if footnote:
        import textwrap
        parts = [
            "Cómo leerlo: cada celda es la correlación (Pearson) entre los retornos diarios "
            "de dos ETFs. Verde (hacia +1): tienden a moverse juntos. Rojo (hacia −1): tienden "
            "a moverse en sentido contrario. Oscuro (cerca de 0): sin relación clara."
        ]
        if cluster:
            parts.append("Los tickers están ordenados por similitud.")
            if residual:
                parts.append(
                    "Los grupos se calcularon quitando el factor de mercado común, para que se vea "
                    "la estructura interna; los colores de la matriz muestran la correlación original."
                )
            if blocks is not None:
                parts.append(
                    "Las líneas gruesas separan grupos de ETFs con una correlación promedio de "
                    f"~{rho_blocks:.2f} o menos entre sí (cada grupo comparte color; se alternan dos tonos)."
                )
            if sub is not None:
                parts.append(
                    f"Las líneas finas separan subgrupos con una correlación promedio de ~{rho_sub:.2f} "
                    "o menos entre sí (dentro de cada grupo, el tono del color varía levemente entre subgrupos)."
                )
        parts.append("Correlación no implica causalidad y cambia según el período elegido.")

        from matplotlib.font_manager import FontProperties
        fs = float(np.clip(11 * scale, 10, 18))
        joined = " ".join(parts)
        max_px = fig_w * 0.94 * fig.dpi
        # Se mide con la fuente real (cada tipografía tiene otro ancho y otra altura de línea)
        try:
            renderer = fig.canvas.get_renderer()
            prop = FontProperties(size=fs)
            lines, cur = [], ""
            for word in joined.split(" "):
                trial = f"{cur} {word}" if cur else word
                if cur and renderer.get_text_width_height_descent(trial, prop, False)[0] > max_px:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
            text = "\n".join(lines)
        except Exception:
            renderer = None
            text = textwrap.fill(joined, width=int(fig_w * 72 * 0.94 / (0.55 * fs)))
        foot = fig.text(0.03, 0.12 / fig_h, text, ha='left', va='bottom', fontsize=fs,
                        color=TEXT_COLOR, alpha=0.8, linespacing=1.4)
        if renderer is not None:
            text_h_in = foot.get_window_extent(renderer).height / fig.dpi
        else:
            text_h_in = (text.count("\n") + 1) * fs * 1.45 / 72
        bottom_in = 0.12 + text_h_in + 0.3   # pulgadas reservadas al pie
        plt.tight_layout(pad=1.8, rect=(0, bottom_in / fig_h, 1, 1))
    else:
        plt.tight_layout(pad=1.8)

    abs_path = os.path.abspath(OUTPUT_FILE)
    plt.savefig(abs_path, dpi=dpi, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)

    # Salidas opcionales (aparte de la matriz)
    if (dendrogram_path or csv_path) and (res is None or res.get('Z') is None):
        print("⚠️ No se pudo generar el dendrograma/CSV de clusters (falta scipy).")
    else:
        if dendrogram_path:
            d_path = plot_dendrogram(res, title, dendrogram_path, residual)
            print(f"Dendrograma: {d_path}")
        if csv_path:
            c_path = _save_clusters_csv(res, csv_path)
            print(f"Clusters CSV: {os.path.abspath(c_path)}")

    return abs_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers (space separated)')
    parser.add_argument('--title', type=str, default="Acciones", help='Título del gráfico')
    parser.add_argument('--start_year', type=int, default=None, help='Año de inicio (opcional)')
    parser.add_argument('--n_blocks', type=int, default=None, help='Cantidad de bloques (0 = sin líneas, omitir = automático)')
    parser.add_argument('--n_subblocks', type=int, default=None, help='Cantidad de subbloques (líneas finas); omitir = sin subbloques')
    parser.add_argument('--residual', action='store_true',
                        help='Agrupar sobre la correlación sin el factor de mercado (la matriz muestra la correlación original)')
    parser.add_argument('--dendrogram', action='store_true',
                        help=f'Generar además el dendrograma en un PNG aparte ({DENDROGRAM_FILE})')
    parser.add_argument('--csv', action='store_true',
                        help=f'Guardar además un CSV ticker -> bloque/subbloque ({CLUSTERS_CSV_FILE})')
    args = parser.parse_args()

    try:
        unique_tickers = list(dict.fromkeys([t.upper() for t in args.tickers]))
        corr_matrix = get_data_and_corr(unique_tickers, args.start_year)
        if corr_matrix is None or corr_matrix.empty:
            sys.exit(1)

        # Las salidas opcionales se imprimen antes, para que la línea |ruta| siga siendo la última
        path = plot_correlation(corr_matrix, args.title, n_blocks=args.n_blocks,
                                n_subblocks=args.n_subblocks, residual=args.residual,
                                dendrogram_path=DENDROGRAM_FILE if args.dendrogram else None,
                                csv_path=CLUSTERS_CSV_FILE if args.csv else None)
        print(f"|{path}|")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_correlation(tickers, title="Acciones", start_year=None, n_blocks=None, n_subblocks=None,
                    residual=False, dendrogram=False, csv=False):
    """Wrapper for Streamlit: builds correlation heatmap.
    n_blocks:    None/0 = automático; N = cantidad de bloques marcados con líneas gruesas.
    n_subblocks: None/0 = sin subbloques; N = cantidad de subbloques (líneas finas), mayor que los bloques.
    residual:    True = agrupar sobre la correlación sin el factor de mercado.
    dendrogram:  True = además guarda el dendrograma en os.path.abspath(DENDROGRAM_FILE).
    csv:         True = además guarda el CSV de clusters en os.path.abspath(CLUSTERS_CSV_FILE).
    Devuelve siempre la ruta del heatmap (los archivos opcionales quedan en las rutas indicadas)."""
    unique_tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    corr_matrix = get_data_and_corr(unique_tickers, start_year)
    if corr_matrix is None or corr_matrix.empty:
        return None
    path = plot_correlation(corr_matrix, title, n_blocks=(n_blocks or None),
                            n_subblocks=(n_subblocks or None), residual=residual,
                            dendrogram_path=DENDROGRAM_FILE if dendrogram else None,
                            csv_path=CLUSTERS_CSV_FILE if csv else None)
    return path

if __name__ == "__main__":
    main()
