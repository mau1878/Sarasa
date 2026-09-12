import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os
import json
from datetime import datetime, timedelta
import warnings
import textwrap

# Ignorar warnings de yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===================== CONFIG =====================
BG_COLOR = '#1a1a1a'
TEXT_COLOR = 'white'
plt.style.use('dark_background')

# ==================== DIVIDENDS ADJUSTMENTS ====================
DIVIDENDS = {
    'BYMA.BA': [
        (datetime(2026, 4, 14), 22.68)
    ]
    # Podés agregar más excepciones acá abajo:
    # 'GGAL.BA': [(datetime(2025, 10, 5), 15.00)],
}

def apply_dividend_adjustments(df):
    if df is None or df.empty:
        return df
    for ticker in df.columns:
        if ticker in DIVIDENDS:
            for ex_date, amount in DIVIDENDS[ticker]:
                mask = df.index <= ex_date
                df.loc[mask, ticker] -= amount
    return df

def load_ticker_names(json_path):
    mapping = {}
    if not os.path.exists(json_path):
        return mapping
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def extract_names(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        mapping[k.upper()] = v
                    elif isinstance(v, (dict, list)):
                        extract_names(v)
            elif isinstance(obj, list):
                pass
        extract_names(data)
    except Exception as e:
        print(f"Warning: No se pudo leer nombres del JSON: {e}")
    return mapping

def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers if isinstance(tickers, list) else [tickers]
        return apply_dividend_adjustments(data)
    except Exception:
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = tickers if isinstance(tickers, list) else [tickers]
            return apply_dividend_adjustments(data)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

def fetch_ccl_proxy(start_date, end_date):
    try:
        ccl_data = yf.download(['YPFD.BA', 'YPF'], start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        if ccl_data.empty or 'YPFD.BA' not in ccl_data.columns or 'YPF' not in ccl_data.columns:
            return None
        ccl_ratio = ccl_data['YPFD.BA'] / ccl_data['YPF']
        ccl_ratio = ccl_ratio.interpolate(method='time').ffill().bfill()
        return ccl_ratio
    except Exception as e:
        print(f"Warning: Error fetching CCL: {e}")
        return None

def calculate_performance(data, reference_prices=None):
    data = data.dropna(how='all')
    if data.empty:
        return pd.Series()
    data_filled = data.ffill().bfill()
    if reference_prices is not None and not reference_prices.empty:
        first_price = reference_prices.reindex(data_filled.columns)
        first_price = first_price.fillna(data_filled.iloc[0])
    else:
        first_price = data_filled.iloc[0]
    last_price = data_filled.iloc[-1]
    returns = ((last_price - first_price) / first_price) * 100
    return returns.sort_values(ascending=False)

def create_bar_plot(performance, title, period_label, name_mapping, filename="performance_chart.png"):
    if performance.empty:
        return None
    df = performance.to_frame(name='Return')
    labels = []
    for ticker in df.index:
        real_name = name_mapping.get(ticker, ticker)
        wrapped_name = "\n".join(textwrap.wrap(real_name, width=12))
        labels.append(wrapped_name)
    df['Display Name'] = labels

    colors = ["#ff3333", "#ffffff", "#00e673"]
    cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", colors)
    vmin = df['Return'].min()
    vmax = df['Return'].max()
    if vmin >= 0: vmin = -0.01
    if vmax <= 0: vmax = 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    palette_colors = [cmap(norm(v)) for v in df['Return']]

    num_bars = len(df)
    calculated_width = 10 + (num_bars * 0.5)
    fig_width = max(12, min(22, calculated_width))
    fig_height = 10

    if num_bars > 40:
        base_max_size = 10
    elif num_bars > 25:
        base_max_size = 11
    elif num_bars > 15:
        base_max_size = 12
    else:
        base_max_size = 14

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(df['Display Name'], df['Return'], color=palette_colors, width=0.8, zorder=3)

    y_range = vmax - vmin
    y_buffer = y_range * 0.15
    ax.set_ylim(vmin - y_buffer, vmax + y_buffer)

    for bar in bars:
        height = bar.get_height()
        offset = y_range * 0.02 if height >= 0 else -y_range * 0.02
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f'{height:.1f}%',
                ha='center', va=va, color=TEXT_COLOR, fontsize=11, rotation=90, fontweight='bold')

    ax.axhline(0, color='white', linewidth=2, alpha=0.8, linestyle='-', zorder=4)
    for x in range(len(df) - 1):
        ax.axvline(x + 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.2)

    ax.set_title(f"{title}\n{period_label}", fontsize=24, color=TEXT_COLOR, pad=25, fontweight='bold')
    ax.set_ylabel("Retorno (%)", fontsize=16, color=TEXT_COLOR, labelpad=15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(axis='x', colors=TEXT_COLOR, rotation=45)
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=12)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    xtick_labels = ax.get_xticklabels()
    for label in xtick_labels:
        text = label.get_text()
        num_lines = text.count('\n') + 1
        if num_lines == 1:
            label.set_fontsize(base_max_size)
        elif num_lines == 2:
            label.set_fontsize(max(8, base_max_size - 2))
        else:
            label.set_fontsize(max(7, base_max_size - 4))

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('white')

    ax.text(0.5, 0.5, '@mtaurus_ok', transform=ax.transAxes,
            fontsize=50, color='gray', alpha=0.15, ha='center', va='center', rotation=30)

    plt.tight_layout()
    output_path = os.path.abspath(filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


# ===================== UPDATED RUNNER WITH CUSTOM START DATE =====================
def run_performance(
    tickers,
    period="YTD",
    group_name="Activos",
    tickers_file=None,
    ccl_adjust=False,
    custom_start_date=None  # ← NEW
):
    name_mapping = load_ticker_names(tickers_file) if tickers_file else {}

    now = datetime.now()
    end_date = now + timedelta(days=1)
    start_date = None
    target_ref_date = None
    custom_label = None

    if custom_start_date:
        start_date = custom_start_date
        custom_label = f"desde {custom_start_date.strftime('%d/%m/%Y')}"
    else:
        if period == "YTD":
            target_ref_date = datetime(now.year - 1, 12, 31).date()
            start_date = (now.replace(month=1, day=1) - timedelta(days=15)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "1Y":
            start_date = now - timedelta(days=365)
        elif period == "2Y":
            start_date = now - timedelta(days=365 * 2)
        elif period == "5Y":
            start_date = now - timedelta(days=365 * 5)
        elif period == "MTD":
            if now.month == 1:
                prev_month_end = datetime(now.year - 1, 12, 31)
            else:
                prev_month_end = datetime(now.year, now.month, 1) - timedelta(days=1)
            target_ref_date = prev_month_end.date()
            start_date = (now - timedelta(days=70)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "WTD":
            weekday = now.weekday()
            days_back = (weekday - 4) % 7 + 7
            if weekday == 4:
                days_back = 7
            target_ref_date = (now - timedelta(days=days_back)).date()
            start_date = (now - timedelta(days=25)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_date = now - timedelta(days=365)

    data = fetch_data(tickers, start_date, end_date)
    if data is None or data.empty:
        return None
    data = data.dropna(axis=1, how="all")
    if data.empty:
        return None

    if ccl_adjust:
        ccl_ratio = fetch_ccl_proxy(start_date, end_date)
        if ccl_ratio is not None:
            ccl_ratio_aligned = ccl_ratio.reindex(data.index).interpolate(method='time').ffill().bfill()
            data = data.div(ccl_ratio_aligned, axis=0)
            group_name += " (Ajustado CCL)"
        else:
            print("Warning: Could not fetch CCL proxy correctly.")

    reference_prices = None
    if target_ref_date is not None and not data.empty:
        mask = pd.Series(data.index.date <= target_ref_date, index=data.index)
        valid_idx = data.index[mask]
        if len(valid_idx) > 0:
            ref_date = valid_idx.max()
            reference_prices = data.loc[ref_date]

    performance = calculate_performance(data, reference_prices)
    if performance.empty:
        return None

    last_date_fetched = data.index[-1]
    real_end_str = last_date_fetched.strftime('%d/%m/%Y')

    if custom_label:
        period_label = f"Periodo: {custom_label} - {real_end_str}"
    else:
        if reference_prices is not None:
            real_start_str = reference_prices.name.strftime('%d/%m/%Y')
        else:
            real_start_str = data.index[0].strftime('%d/%m/%Y')
        period_label = f"Periodo: {real_start_str} - {real_end_str}"

    path = create_bar_plot(
        performance,
        f"Performance: {group_name}",
        period_label,
        name_mapping
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', type=str, required=True)
    parser.add_argument('--group_name', type=str, default="Activos")
    parser.add_argument('--tickers_file', type=str, default=None)
    parser.add_argument('--ccl_adjust', action='store_true')
    args = parser.parse_args()
    # main() is kept for backward CLI compatibility (no custom date here)
    path = run_performance(
        args.tickers,
        period=args.period,
        group_name=args.group_name,
        tickers_file=args.tickers_file,
        ccl_adjust=args.ccl_adjust
    )
    if path:
        print(path)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
