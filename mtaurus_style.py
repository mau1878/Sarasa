"""
Estilo tipográfico compartido de MTaurus.

Un solo lugar para elegir la tipografía de TODOS los gráficos del repo:

  * matplotlib / seaborn: registra los TTF de ./fonts y fija rcParams['font.family'].
    Si a la fuente le falta algún carácter, matplotlib usa DejaVu Sans solo para ese carácter.
  * plotly: fija la familia en las plantillas 'plotly' y 'plotly_dark'. Para los casos en que un
    layout define la fuente a mano, usar PLOTLY_FONT_FAMILY.
  * Streamlit: inject_streamlit_font(st) carga la fuente en el navegador (Google Fonts) para que los
    gráficos plotly interactivos la muestren.

Para cambiar de tipografía alcanza con cambiar FONT_KEY (más abajo). Las cuatro opciones están
incluidas en ./fonts con su licencia (SIL Open Font License 1.1).

Exportación de PNG con plotly (fig.write_image, usado en ratio_plot.py e inflation_adjusted_plot.py):
kaleido renderiza con un Chrome interno que solo ve las fuentes instaladas en el sistema.
  - Linux: este módulo copia las fuentes a ~/.local/share/fonts/mtaurus y corre fc-cache solo.
  - Windows / macOS: instalar una vez los .ttf de la fuente elegida (doble clic > Instalar).
    Sin eso, esos PNG usan la fuente de reemplazo del navegador; el resto de los gráficos no se afecta.

Además define la paleta de colores del "tema Taurus" para los gráficos matplotlib (BG_COLOR, PAPER_COLOR,
TEXT_COLOR, ...): los scripts la importan con `from mtaurus_style import BG_COLOR, ...`.
"""
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Tipografías disponibles (todas OFL, incluidas en ./fonts).
FONTS = {
    # Semicondensada, formas algo cuadradas: compacta para tickers, con carácter propio.
    "saira": {
        "files": "SairaSemiCondensed",
        "css": ["Saira Semi Condensed", "Saira SemiCondensed"],
        "google": "Saira+Semi+Condensed:wght@400;500;700",
    },
    # Grotesca de ancho normal, con detalles irregulares en a, g, y y los números.
    "familjen": {
        "files": "FamiljenGrotesk",
        "css": ["Familjen Grotesk"],
        "google": "Familjen+Grotesk:wght@400;500;700",
    },
    # Semicondensada más neutra y nítida.
    "sofia": {
        "files": "SofiaSansSemiCondensed",
        "css": ["Sofia Sans Semi Condensed"],
        "google": "Sofia+Sans+Semi+Condensed:wght@400;500;700",
    },
    # Muy condensada: buena para títulos, apretada para el texto chico.
    "bigshoulders": {
        "files": "BigShouldersText",
        "css": ["Big Shoulders Text"],
        "google": "Big+Shoulders+Text:wght@400;500;700",
    },
}

FONT_KEY = "saira"  # <- cambiar acá para cambiar la tipografía de todo el repo

_CFG = FONTS[FONT_KEY]
# Familia para plotly / CSS: con comillas y con respaldo genérico
PLOTLY_FONT_FAMILY = ", ".join(f'"{n}"' for n in _CFG["css"]) + ", sans-serif"

# ---------------------------------------------------------------------------
# Paleta compartida ("tema Taurus") para los gráficos matplotlib del repo.
# Un gráfico que necesite un valor distinto (otro alpha de grilla, por ejemplo) lo redefine localmente.
# ---------------------------------------------------------------------------
BG_COLOR = '#0f0f1a'      # fondo de ejes
PAPER_COLOR = '#05050f'   # fondo de figura y de savefig
TEXT_COLOR = '#e0e0ff'    # títulos, labels, spines
TICK_COLOR = '#c0c0ff'    # tick labels
ANNOT_COLOR = '#ffffff'   # anotaciones sobre celdas / barras
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.22)
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)
ZERO_LINE_COLOR = '#ff4d4d'
# Divergente apagado (rojo - gris azulado - verde), centrado en cero con TwoSlopeNorm
DIV_COLORS = [(0.7, 0.15, 0.15), (0.3, 0.3, 0.35), (0.15, 0.6, 0.25)]

_mpl_ready = False
_plotly_ready = False


def _font_files():
    return sorted(FONT_DIR.glob(f"{_CFG['files']}-*.ttf"))


def setup_matplotlib():
    """Registra los TTF y fija la fuente por defecto de matplotlib (idempotente)."""
    global _mpl_ready
    if _mpl_ready:
        return
    files = _font_files()
    if not files:
        warnings.warn(f"mtaurus_style: no se encontraron fuentes en {FONT_DIR}; se usa la fuente por defecto.")
        return
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    family = None
    for p in files:
        fm.fontManager.addfont(str(p))
        if family is None:
            family = fm.FontProperties(fname=str(p)).get_name()
    mpl.rcParams["font.family"] = [family, "DejaVu Sans"]  # DejaVu solo como respaldo por carácter
    _mpl_ready = True


def install_system_fonts():
    """Linux: deja las fuentes visibles para kaleido/Chrome (write_image). Falla en silencio."""
    try:
        if not sys.platform.startswith("linux"):
            return
        dest = Path.home() / ".local" / "share" / "fonts" / "mtaurus"
        dest.mkdir(parents=True, exist_ok=True)
        copied = False
        for p in _font_files():
            target = dest / p.name
            if not target.exists():
                shutil.copy2(p, target)
                copied = True
        if copied and shutil.which("fc-cache"):
            subprocess.run(["fc-cache", "-f", str(dest)], check=False, capture_output=True, timeout=60)
    except Exception:
        pass


def setup_plotly():
    """Fija la familia tipográfica en las plantillas de plotly (idempotente)."""
    global _plotly_ready
    if _plotly_ready:
        return
    try:
        import plotly.io as pio
    except ImportError:
        return
    for name in ("plotly", "plotly_dark"):
        try:
            pio.templates[name].layout.font.family = PLOTLY_FONT_FAMILY
        except Exception:
            pass
    install_system_fonts()
    _plotly_ready = True


def inject_streamlit_font(st):
    """Carga la fuente en el navegador para los gráficos plotly dentro de Streamlit."""
    st.markdown(
        "<style>@import url('https://fonts.googleapis.com/css2?family="
        f"{_CFG['google']}&display=swap');</style>",
        unsafe_allow_html=True,
    )


def setup():
    setup_matplotlib()
    setup_plotly()


# Al importar el módulo queda todo configurado: alcanza con `import mtaurus_style` en cada script.
setup()
