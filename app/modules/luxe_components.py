"""Premium UI components for the target designer page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


@dataclass
class TargetPresetMeta:
    """Metadata used to render the Tesla-style preset cards."""

    icon: str
    tagline: str


_PRESET_META: Dict[str, TargetPresetMeta] = {
    "Container": TargetPresetMeta(
        icon="📦",
        tagline="Para almacenaje hermético y modular sin sacrificar estilo.",
    ),
    "Utensil": TargetPresetMeta(
        icon="🍴",
        tagline="Diseñado para tareas delicadas con acabado pulido lunar.",
    ),
    "Interior": TargetPresetMeta(
        icon="🛋️",
        tagline="Habitáculos confortables que optimizan espacio y calor.",
    ),
    "Tool": TargetPresetMeta(
        icon="🛠️",
        tagline="Robustez industrial lista para cualquier misión orbital.",
    ),
}


def _ensure_css_once() -> None:
    """Inject the CSS needed for the luxury cards and gauges once."""

    if st.session_state.get("_luxe_css_loaded"):
        return

    st.markdown(
        """
        <style>
        .luxe-card-grid {display:flex; gap:1rem; flex-wrap:wrap;}
        .luxe-card {
            border-radius: 18px;
            padding: 1rem;
            width: 220px;
            background: linear-gradient(145deg, rgba(15,25,36,0.9), rgba(48,74,102,0.85));
            color: #f5f9ff;
            position: relative;
            box-shadow: 0 25px 45px rgba(2,12,27,0.45);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .luxe-card.is-active {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 35px 60px rgba(0, 150, 255, 0.45);
            border-color: rgba(94,174,255,0.8);
        }
        .luxe-card h4 {margin: 0; font-size: 1.1rem;}
        .luxe-card .tagline {opacity: 0.85; font-size: 0.85rem; margin-top: 0.4rem;}
        .luxe-card .image {
            width: 100%;
            height: 120px;
            border-radius: 14px;
            margin-bottom: 0.8rem;
            background: radial-gradient(circle at top left, rgba(120,195,255,0.8), rgba(8,15,27,0.6));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            text-shadow: 0 12px 30px rgba(0,0,0,0.35);
        }
        .target-3d-card {
            perspective: 1200px;
        }
        .target-3d-card .inner {
            background: linear-gradient(160deg, rgba(8,21,35,0.9), rgba(45,90,120,0.88));
            border-radius: 24px;
            padding: 1.5rem;
            min-height: 260px;
            color: #eaf4ff;
            box-shadow: 0 35px 55px rgba(3, 12, 32, 0.55);
            border: 1px solid rgba(255,255,255,0.08);
            transform: rotateY(-12deg) rotateX(6deg);
            transform-style: preserve-3d;
            position: relative;
        }
        .target-3d-card .inner::after {
            content: "";
            position: absolute;
            inset: 10px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.05);
            pointer-events: none;
        }
        .circular-indicator {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            background: conic-gradient(var(--accent-color, #59b1ff) calc(var(--value, 0) * 1%), rgba(255,255,255,0.08) 0);
            color: #0e2137;
            margin-right: 0.6rem;
        }
        .circular-indicator span {
            font-size: 0.85rem;
        }
        .slider-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }
        .slider-row .stSlider {flex: 1;}
        .feedback-pill {
            display: inline-flex;
            gap: 0.3rem;
            align-items: center;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(88, 184, 255, 0.12);
            color: #cfe9ff;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_luxe_css_loaded"] = True


def _render_preset_cards(presets: Iterable[Dict]) -> str:
    """Return the currently selected preset name after rendering cards."""

    presets = list(presets)
    st.session_state.setdefault("target_preset_choice", presets[0]["name"])

    cols = st.columns(len(presets))
    selected_name = st.session_state["target_preset_choice"]

    for col, preset in zip(cols, presets):
        meta = _PRESET_META.get(
            preset["name"],
            TargetPresetMeta(icon="🛰️", tagline="Optimizado para desafíos orbitales."),
        )
        active_class = "is-active" if preset["name"] == selected_name else ""
        col.markdown(
            f"""
            <div class="luxe-card {active_class}">
                <div class="image">{meta.icon}</div>
                <h4>{preset['name']}</h4>
                <div class="tagline">{meta.tagline}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if col.button("Elegir", key=f"preset_card_{preset['name']}"):
            st.session_state["target_preset_choice"] = preset["name"]
            selected_name = preset["name"]

    return selected_name


def _render_indicator(value: float, max_value: float, accent: str, label: str) -> None:
    percentage = 100 * value / max_value if max_value else 0
    st.markdown(
        f"<div class='circular-indicator' style='--value:{percentage}; --accent-color:{accent};'>"
        f"<span>{percentage:.0f}%</span></div><div class='feedback-pill'>{label}: {value:.2f}</div>",
        unsafe_allow_html=True,
    )


def _gauge(title: str, value: float, max_value: float, unit: str, color: str = "#59b1ff") -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": unit},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(6,20,33,0.65)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
            },
            title={"text": title},
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _build_audio_clip() -> bytes:
    """Create a short sine beep encoded as WAV bytes."""

    sample_rate = 44100
    duration = 0.25
    frequency = 880
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = np.int16(tone * 32767)

    from io import BytesIO
    import wave

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def target_configurator(
    presets: List[Dict],
    scenario_options: Tuple[str, ...] | List[str] | None = None,
) -> Dict:
    """Render the luxe target configurator and return the resulting target spec."""

    if not presets:
        st.warning("No hay presets disponibles para configurar el objetivo.")
        return {}

    _ensure_css_once()
    scenario_options = tuple(scenario_options or ())

    st.subheader("Target Configurator ✨")
    st.caption("Seleccioná un preset y refiná el objetivo con feedback inmediato.")

    selected_name = _render_preset_cards(presets)
    selected_preset = next(p for p in presets if p["name"] == selected_name)

    current_target = st.session_state.get("target")
    default_scenario = scenario_options[0] if scenario_options else ""
    if current_target is None:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": default_scenario,
            "crew_time_low": False,
        }
    elif current_target.get("name") != selected_name:
        st.session_state["target"] = {
            **selected_preset,
            "scenario": current_target.get("scenario", default_scenario),
            "crew_time_low": current_target.get("crew_time_low", False),
        }

    current_target = st.session_state["target"]

    previous_values = st.session_state.get("_target_prev_values", {})

    base = {
        key: float(selected_preset[key]) if "max_" not in key else selected_preset[key]
        for key in ("rigidity", "tightness", "max_water_l", "max_energy_kwh", "max_crew_min")
    }

    main_col, summary_col = st.columns([3, 1])

    with main_col:
        preview_col, controls_col = st.columns([1.2, 1.8])
        with preview_col:
            st.markdown(
                f"""
                <div class="target-3d-card">
                    <div class="inner">
                        <h3>{selected_name}</h3>
                        <p>Render conceptual en vivo del objeto seleccionado.</p>
                        <div style="margin-top:2rem; font-size:0.85rem; opacity:0.85;">
                            Rigidez base: {base['rigidity']:.2f}<br/>
                            Estanqueidad base: {base['tightness']:.2f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            scenario = ""
            if scenario_options:
                default_scenario = current_target.get("scenario", scenario_options[0])
                if default_scenario not in scenario_options:
                    default_idx = 0
                else:
                    default_idx = scenario_options.index(default_scenario)
                scenario = st.selectbox(
                    "Escenario del reto",
                    scenario_options,
                    index=default_idx,
                )
            crew_low = st.toggle(
                "Crew-time Low",
                value=current_target.get("crew_time_low", False),
                help="Prioriza procesos con poco tiempo de tripulación.",
            )
        with controls_col:
            st.markdown("#### Ajustes dinámicos")
            rigidity = st.slider(
                "Rigidez deseada",
                0.0,
                1.0,
                float(current_target.get("rigidity", selected_preset["rigidity"])),
                0.05,
            )
            _render_indicator(rigidity, 1.0, "#4ecdc4", "Rigidez")

            tightness = st.slider(
                "Estanqueidad deseada",
                0.0,
                1.0,
                float(current_target.get("tightness", selected_preset["tightness"])),
                0.05,
            )
            _render_indicator(tightness, 1.0, "#ff8c69", "Estanqueidad")

            max_water = st.slider(
                "Agua máxima (L)",
                0.0,
                3.0,
                float(current_target.get("max_water_l", selected_preset["max_water_l"])),
                0.1,
            )
            _render_indicator(max_water, 3.0, "#59b1ff", "Agua")

            max_energy = st.slider(
                "Energía máxima (kWh)",
                0.0,
                3.0,
                float(current_target.get("max_energy_kwh", selected_preset["max_energy_kwh"])),
                0.1,
            )
            _render_indicator(max_energy, 3.0, "#f8d66d", "Energía")

            max_crew = st.slider(
                "Tiempo máximo de tripulación (min)",
                5,
                60,
                int(current_target.get("max_crew_min", selected_preset["max_crew_min"])),
                1,
            )
            _render_indicator(max_crew, 60.0, "#d277ff", "Crew")

            audio_enabled = st.checkbox(
                "Audio feedback", value=st.session_state.get("_target_audio", False)
            )
            haptic_enabled = st.checkbox(
                "Vibración háptica", value=st.session_state.get("_target_haptic", False)
            )
            st.session_state["_target_audio"] = audio_enabled
            st.session_state["_target_haptic"] = haptic_enabled

            if audio_enabled:
                st.audio(_build_audio_clip(), format="audio/wav", sample_rate=44100)

    with main_col:
        st.markdown("#### Simulación visual")
        gauges = st.columns(3)
        with gauges[0]:
            _gauge("Agua", max_water, 3.0, " L")
        with gauges[1]:
            _gauge("Energía", max_energy, 3.0, " kWh", color="#f8d66d")
        with gauges[2]:
            _gauge("Crew", max_crew, 60.0, " min", color="#d277ff")

        feedback_area = st.empty()

    current_values = {
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
    }

    if previous_values and current_values != previous_values:
        messages = []
        if st.session_state.get("_target_audio"):
            messages.append("🔊 Audio feedback (simulado)")
        if st.session_state.get("_target_haptic"):
            messages.append("🤲 Haptic pulse (simulado)")
        if messages:
            feedback_area.success(" ".join(messages))
    else:
        feedback_area.empty()

    st.session_state["_target_prev_values"] = current_values

    with summary_col:
        st.markdown("### Resumen")
        st.caption("Comparación contra el preset seleccionado.")

        def metric(label: str, value: float, base_value: float, unit: str = "") -> None:
            delta = value - base_value
            st.metric(label, f"{value:.2f}{unit}", f"{delta:+.2f}{unit}")

        metric("Rigidez", rigidity, base["rigidity"], "")
        metric("Estanqueidad", tightness, base["tightness"], "")
        metric("Agua máx.", max_water, base["max_water_l"], " L")
        metric("Energía máx.", max_energy, base["max_energy_kwh"], " kWh")
        metric("Crew máx.", max_crew, base["max_crew_min"], " min")

    target = {
        "name": selected_name,
        "rigidity": rigidity,
        "tightness": tightness,
        "max_water_l": max_water,
        "max_energy_kwh": max_energy,
        "max_crew_min": max_crew,
        "scenario": scenario if scenario_options else current_target.get("scenario", ""),
        "crew_time_low": crew_low,
    }

    st.session_state["target"] = target
    return target
