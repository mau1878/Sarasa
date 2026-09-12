import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==== CONFIG - Dark theme refined ====
BG_COLOR = '#0f0f1a'  # Very dark background
PAPER_COLOR = '#05050f'  # Slightly different for figure
TEXT_COLOR = '#e0e0ff'  # Soft light purple-white
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.22)  # Matplotlib-friendly RGBA tuple
LINE_COLOR = '#ff7070'  # Softer vivid red
FILL_COLOR = '#ff4040'  # Fill under drawdown
ACCENT_COLOR = '#40c0ff'  # Optional accent (not used here, but consistent)
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)  # Matplotlib-safe RGBA

plt.style.use('dark_background')


def fetch_data(ticker, start_year=None):
    """Downloads historical data for the ticker."""
    if start_year:
        start_date = f"{start_year}-01-01"
    else:
        start_date = "1900-01-01"  # Default start

    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

    print(f"Downloading data for {ticker}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df['Close'][ticker].to_frame('Close')
            except KeyError:
                df = df.iloc[:, 0].to_frame('Close')  # Fallback
        elif 'Close' in df.columns:
            df = df[['Close']]
        else:
            return pd.DataFrame()

        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


def calculate_drawdown(df):
    """Calculates the drawdown series."""
    df = df.copy()
    df['Running_Max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] / df['Running_Max']) - 1
    return df


def plot_drawdown(df, ticker, output_dir="output", from_ath=False):
    """Generates the Underwater (Drawdown) Plot."""

    current_dd = df['Drawdown'].iloc[-1]
    max_dd = df['Drawdown'].min()
    max_dd_date = df['Drawdown'].idxmin()

    # Create Figure
    fig = plt.figure(figsize=(13, 7.5), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    # Plot Drawdown Line
    ax.plot(df.index, df['Drawdown'], color=LINE_COLOR, linewidth=1.4, label='Drawdown')

    # Fill Area
    ax.fill_between(df.index, df['Drawdown'], 0, color=FILL_COLOR, alpha=0.25)

    # Highlight Max Drawdown
    # Only plot max dd annotation if it exists in the current window
    if not pd.isnull(max_dd_date):
        ax.scatter(max_dd_date, max_dd, color='#ffffff', s=60, zorder=10, edgecolor='#ff7070', linewidth=1.2)
        ax.annotate(
            f'Max DD: {max_dd:.1%}\n({max_dd_date.strftime("%Y-%m")})',
            xy=(max_dd_date, max_dd),
            xytext=(15, 15),
            textcoords='offset points',
            color=TEXT_COLOR,
            fontsize=10,
            fontweight='bold',
            arrowprops=dict(arrowstyle="->", color='#ff7070', linewidth=1.2, alpha=0.8)
        )

    # Highlight Current Drawdown
    last_date = df.index[-1]
    ax.scatter(last_date, current_dd, color='#ffffff', s=45, zorder=10, edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.annotate(
        f'Current: {current_dd:.1%}',
        xy=(last_date, current_dd),
        xytext=(15, -25),
        textcoords='offset points',
        color=TEXT_COLOR,
        fontsize=11,
        fontweight='bold'
    )

    # Formatting
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    title_suffix = " (Current Cycle)" if from_ath else ""
    ax.set_title(f"{ticker} – Underwater Plot{title_suffix}",
                 fontsize=20, color=TEXT_COLOR, fontweight='bold', pad=20)
    ax.set_ylabel("Drawdown %", fontsize=13, color=TEXT_COLOR, labelpad=12)
    ax.set_xlabel("Date", fontsize=13, color=TEXT_COLOR, labelpad=10)

    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.8)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(TEXT_COLOR)
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10)

    # Watermark
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok",
             fontsize=52, color=WATERMARK_COLOR,
             ha='center', va='center', rotation=30)

    # Footer
    fig.text(0.98, 0.015, f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
             ha='right', fontsize=9, color='gray', alpha=0.7)

    plt.tight_layout(pad=1.5)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"drawdown_{ticker.replace('.', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)

    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol (e.g., AAPL, SPY, GGAL.BA)')
    parser.add_argument('--start_year', type=int, default=None, help='Start year for data')
    parser.add_argument('--from_last_ath', action='store_true', help='Plot only from the most recent All-Time High')
    args = parser.parse_args()

    ticker = args.ticker.upper()

    df = fetch_data(ticker, args.start_year)
    if df.empty:
        print(f"No data found for {ticker}")
        sys.exit(1)

    df = calculate_drawdown(df)

    # --- Logic for filtering from last ATH ---
    if args.from_last_ath:
        # Find indices where Drawdown is 0 (meaning price is at ATH)
        ath_indices = df[df['Drawdown'] == 0].index

        if not ath_indices.empty:
            last_ath_date = ath_indices[-1]

            # Check if we are currently at ATH (last date is an ATH)
            if last_ath_date == df.index[-1]:
                print("Info: Asset is currently at All-Time High. Showing full history instead of a single point.")
            else:
                print(f"Filtering data from last ATH date: {last_ath_date.date()}")
                df = df.loc[last_ath_date:]
        else:
            print("Warning: No ATH found in the selected period. Showing full data.")

    filepath = plot_drawdown(df, ticker, from_ath=args.from_last_ath)
    print(filepath)
def run_drawdown(ticker, start_year=None, from_last_ath=False):
    """Wrapper for Streamlit: returns path to drawdown plot."""
    ticker = ticker.upper()
    df = fetch_data(ticker, start_year)
    if df.empty:
        return None

    df = calculate_drawdown(df)

    if from_last_ath:
        ath_indices = df[df["Drawdown"] == 0].index
        if not ath_indices.empty:
            last_ath_date = ath_indices[-1]
            if last_ath_date != df.index[-1]:
                df = df.loc[last_ath_date:]

    path = plot_drawdown(df, ticker, from_ath=from_last_ath)
    return path


if __name__ == "__main__":
    main()