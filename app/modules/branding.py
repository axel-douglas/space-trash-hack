# --- path guard universal ---
import sys, pathlib
_here = pathlib.Path(__file__).resolve()
p = _here.parent
while p.name != "app" and p.parent != p:
    p = p.parent
repo_root = p.parent if p.name == "app" else _here.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# --------------------------------

from pathlib import Path
import streamlit as st

# Paleta
PRIMARY   = "#0A84FF"
PRIMARY_2 = "#3AA6FF"
OK        = "#00E38C"
WARN      = "#FFB020"
RISK      = "#FF5C5C"
INK       = "#0E1016"
SURFACE   = "#0F1220"
CARD      = "#171B2C"
BORDER    = "#2A2F45"

STATIC_DIR = (repo_root / "app" / "static")

def _logo_svg_path() -> Path:
    return STATIC_DIR / "logo_rexai.svg"

def inject_branding():
    """Inyecta CSS de marca (sin :has() ni hacks) y muestra el header con logo."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    css = f"""
    <style>
      :root {{
        --primary: {PRIMARY};
        --primary-2: {PRIMARY_2};
        --ok: {OK};
        --warn: {WARN};
        --risk: {RISK};
        --ink: {INK};
        --surface: {SURFACE};
        --card: {CARD};
        --border: {BORDER};
      }}
      .stApp {{
        background: linear-gradient(180deg, #0B0E1C 0%, #0B0E1C 60%, #0A0D1A 100%);
        color: #E8ECF4;
      }}
      .rex-header {{
        display:flex; gap:16px; align-items:center; padding:6px 8px 2px 4px;
      }}
      .rex-title {{ font-weight: 800; letter-spacing:.3px; font-size: 22px; margin: 0; }}
      .rex-sub {{ color:#A9B2C8; margin-top:2px; font-size:13px; }}
      .rex-card {{
        background: var(--card);
        border:1px solid var(--border);
        border-radius:16px; padding:16px 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.22);
      }}
      .rex-chip {{
        display:inline-flex; align-items:center; gap:8px;
        border-radius:999px; padding:4px 10px; font-size:12px;
        background:#1E2438; border:1px solid var(--border);
        margin-right:8px; margin-top:6px;
      }}
      .rex-chip.ok {{ border-color: {OK}; }}
      .rex-chip.warn {{ border-color: {WARN}; }}
      .rex-chip.risk {{ border-color: {RISK}; }}
      [data-testid="stSidebar"] > div:first-child {{
        background: #0D1122;
        border-right: 1px solid var(--border);
      }}
      .rex-logo {{ width: 30px; height: 30px; margin-right:4px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Header con logo
    logo_path = _logo_svg_path()
    svg = logo_path.read_text(encoding="utf-8") if logo_path.exists() else ""
    st.markdown(
        f"""
        <div class="rex-header">
          <div class="rex-logo">{svg}</div>
          <div>
            <div class="rex-title">REX-AI Mars</div>
            <div class="rex-sub">Recycling & Experimentation eXpert — Jezero Base</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
