import streamlit as st
from typing import Literal

CSS = """
<style>
.card {
  background:#151A21; border:1px solid #242C37; border-radius:16px; padding:18px; margin-bottom:12px;
}
.kpi { display:flex; gap:14px; }
.kpi .pill { background:#0E1117; border:1px solid #242C37; border-radius:999px; padding:6px 12px; }
.badge-ok { color:#0FD68B; }
.badge-warn { color:#F39C12; }
.badge-risk { color:#E74C3C; }
.btn-primary { background:#00B894; color:#0E1117; border-radius:12px; padding:10px 16px; font-weight:700; }
.small { opacity:0.8; font-size:0.9rem; }
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def card(title:str, body:str=""):
    st.markdown(f"""<div class="card"><h4>{title}</h4><div class="small">{body}</div></div>""", unsafe_allow_html=True)

def pill(label:str, kind:Literal["ok","warn","risk"]="ok"):
    klass = {"ok":"badge-ok","warn":"badge-warn","risk":"badge-risk"}[kind]
    st.markdown(f"""<span class="pill {klass}">{label}</span>""", unsafe_allow_html=True)

def section(title:str, subtitle:str=""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
