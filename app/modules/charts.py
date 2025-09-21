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

import plotly.graph_objects as go
import pandas as pd

def predictions_ci_chart(df: pd.DataFrame, title="Score predictions"):
    """
    df: columns => batch(str), mean(float), lo(float), hi(float), observed(optional float)
    """
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    # Intervalos
    fig.add_trace(go.Scatter(
        x=df["batch"], y=df["mean"], mode="markers",
        marker=dict(size=9, color="#B388FF"),
        name="Mean prediction"
    ))
    # Barras de confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([df["batch"], df["batch"][::-1]]),
        y=pd.concat([df["hi"], df["lo"][::-1]]),
        fill="toself", fillcolor="rgba(179,136,255,0.25)",
        line=dict(color="rgba(179,136,255,0.0)"),
        hoverinfo="skip",
        name="Confidence interval"
    ))
    if "observed" in df.columns and df["observed"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["batch"], y=df["observed"], mode="markers",
            marker=dict(size=9, color="#45D483"),
            name="Trial"
        ))
    fig.update_layout(
        title=title, template="plotly_dark", height=420,
        margin=dict(l=20,r=20,t=60,b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
