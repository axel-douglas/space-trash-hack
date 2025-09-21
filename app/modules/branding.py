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

PRIMARY   = "#0A84FF"
PRIMARY_2 = "#3AA6FF"
OK        = "#00E38C"
WARN      = "#FFB020"
RISK      = "#FF5C5C"
CARD      = "#171B2C"
BORDER    = "#2A2F45"

STATIC_DIR = (repo_root / "app" / "static")
LOGO_PATH  = STATIC_DIR / "logo_rexai.svg"

def inject_branding():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: linear-gradient(180deg,#0B0E1C 0%,#0B0E1C 60%,#0A0D1A 100%);
            color:#E8ECF4;
          }}
          .rex-header {{ display:flex; gap:16px; align-items:center; padding:6px 8px 2px 4px; }}
          .rex-title {{ font-weight:800; letter-spacing:.3px; font-size:22px; margin:0; }}
          .rex-sub {{ color:#A9B2C8; margin-top:2px; font-size:13px; }}
          .rex-card {{
             background:{CARD}; border:1px solid {BORDER};
             border-radius:16px; padding:16px; box-shadow:0 8px 24px rgba(0,0,0,.22);
          }}
          .rex-chip {{ display:inline-flex; align-items:center; gap:8px; border-radius:999px;
             padding:4px 10px; font-size:12px; background:#1E2438; border:1px solid {BORDER};
             margin-right:8px; margin-top:6px;
          }}
          [data-testid="stSidebar"] > div:first-child {{
            background:#0D1122; border-right:1px solid {BORDER};
          }}
          .rex-logo svg {{ width:160px; height:auto; color:{PRIMARY}; display:block; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    svg = ""
    try:
        if LOGPATH := (LOGO_PATH if LOGO_PATH.exists() else None):
            svg = LOGPATH.read_text(encoding="utf-8")
    except Exception:
        pass

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
        unsafe_allow_html=True,
    )
