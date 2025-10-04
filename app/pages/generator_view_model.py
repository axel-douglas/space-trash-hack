from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import pandas as pd
import streamlit as st

from app.modules.generator.service import GeneratorService
from app.modules.utils import safe_float

__all__ = ["GeneratorViewModel"]


@dataclass
class GeneratorViewModel:
    """Encapsula la interacción con ``st.session_state`` y el servicio."""

    session: MutableMapping[str, Any]
    service: GeneratorService = field(default_factory=GeneratorService)
    button_state_key: str = "generator_button_state"
    button_trigger_key: str = "generator_button_trigger"
    button_error_key: str = "generator_button_error"

    @classmethod
    def from_streamlit(
        cls, service: GeneratorService | None = None
    ) -> "GeneratorViewModel":
        session = st.session_state
        existing = session.get("_generator_view_model_instance")
        if isinstance(existing, cls):
            if service is not None and existing.service is not service:
                existing.service = service
            return existing

        instance = cls(session=session, service=service or GeneratorService())
        session["_generator_view_model_instance"] = instance
        return instance

    def pop_playbook_prefill(
        self,
    ) -> tuple[Mapping[str, object] | None, str | None]:
        raw = self.session.pop("_playbook_generator_filters", None)
        filters: dict[str, object] | None = None
        label: str | None = None
        if isinstance(raw, Mapping):
            candidate_filters = raw.get("filters")
            if isinstance(candidate_filters, Mapping):
                filters = dict(candidate_filters)
            scenario_hint = raw.get("scenario")
            if isinstance(scenario_hint, str) and scenario_hint.strip():
                label = scenario_hint.strip()
        return filters, label

    def apply_playbook_prefilters(
        self,
        filters: Mapping[str, object] | None,
        target: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return False

        for key, value in filters.items():
            self.session[key] = value

        if not isinstance(target, Mapping):
            return True

        energy_limit = safe_float(target.get("max_energy_kwh"))
        water_limit = safe_float(target.get("max_water_l"))
        crew_limit = safe_float(target.get("max_crew_min"))

        if filters.get("showroom_limit_energy") and energy_limit is not None:
            self.session["showroom_energy_limit_value"] = float(energy_limit)
        if filters.get("showroom_limit_water") and water_limit is not None:
            self.session["showroom_water_limit_value"] = float(water_limit)
        if filters.get("showroom_limit_crew") and crew_limit is not None:
            self.session["showroom_crew_limit_value"] = float(crew_limit)
        return True

    def get_target(self) -> dict[str, Any] | None:
        target = self.session.get("target")
        if isinstance(target, Mapping):
            return dict(target)
        return None

    def set_target(self, target: Mapping[str, Any]) -> None:
        self.session["target"] = dict(target)

    def ensure_defaults(self) -> None:
        self.session.setdefault("candidates", [])
        history = self.session.get("optimizer_history")
        if isinstance(history, pd.DataFrame):
            return
        if history is None:
            self.session["optimizer_history"] = pd.DataFrame()
        else:
            self.session["optimizer_history"] = pd.DataFrame(history)

    @property
    def candidates(self) -> list[dict[str, Any]]:
        value = self.session.get("candidates", [])
        if isinstance(value, list):
            return value
        if isinstance(value, pd.DataFrame):
            records = value.to_dict("records")
            self.session["candidates"] = records
            return records
        try:
            as_list = list(value)
        except TypeError:
            as_list = []
        self.session["candidates"] = as_list
        return as_list

    @property
    def optimizer_history(self) -> pd.DataFrame:
        history = self.session.get("optimizer_history")
        if isinstance(history, pd.DataFrame):
            return history
        if history is None:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(history)
        self.session["optimizer_history"] = df
        return df

    @property
    def button_state(self) -> str:
        state = self.session.get(self.button_state_key, "idle")
        return str(state)

    @property
    def button_error(self) -> str | None:
        error = self.session.get(self.button_error_key)
        return str(error) if isinstance(error, str) else None

    def set_button_state(self, state: str) -> None:
        self.session[self.button_state_key] = state

    def set_button_error(self, message: str) -> None:
        self.session[self.button_error_key] = message

    def clear_button_error(self) -> None:
        self.session.pop(self.button_error_key, None)

    def trigger_generation(self) -> None:
        self.set_button_state("loading")
        self.session[self.button_trigger_key] = True
        self.clear_button_error()

    def reset_trigger(self) -> None:
        self.session[self.button_trigger_key] = False

    @property
    def should_generate(self) -> bool:
        return bool(self.session.get(self.button_trigger_key))

    def get_prediction_mode(self, default: str) -> str:
        stored = self.session.get("prediction_mode", default)
        return str(stored)

    def set_prediction_mode(self, mode: str) -> None:
        self.session["prediction_mode"] = mode

    def get_seed_input(self) -> str:
        seed = self.session.get("generator_seed_input", "")
        return str(seed)

    def set_seed_input(self, seed: str) -> None:
        self.session["generator_seed_input"] = seed

    def parse_seed(self) -> tuple[int | None, str | None]:
        seed_raw = self.get_seed_input().strip()
        if not seed_raw:
            return None, None
        try:
            return int(seed_raw, 0), None
        except ValueError:
            return None, "Ingresá un entero válido para la semilla (por ejemplo 42 o 0x2A)."

    def store_results(
        self, candidates: list[dict[str, Any]], history: pd.DataFrame
    ) -> None:
        self.session["candidates"] = candidates
        self.session["optimizer_history"] = history
        self.set_button_state("success")
        self.reset_trigger()
        self.clear_button_error()

    def set_error(self, message: str) -> None:
        self.set_button_state("error")
        self.set_button_error(message)
        self.reset_trigger()

    def set_selected(self, candidate: dict[str, Any], badge: Mapping[str, Any]) -> None:
        self.session["selected"] = {"data": candidate, "safety": dict(badge)}

    def set_ranking_focus(self, row: Mapping[str, Any]) -> None:
        self.session["generator_ranking_focus"] = dict(row)

    def generate_candidates(
        self,
        waste_df: pd.DataFrame,
        proc_df: pd.DataFrame,
        target: Mapping[str, Any],
        *,
        n_candidates: int,
        crew_time_low: bool,
        optimizer_evals: int,
        use_ml: bool,
        seed: int | None,
    ) -> tuple[list[dict], pd.DataFrame]:
        return self.service.generate_candidates(
            waste_df,
            proc_df,
            dict(target),
            n=n_candidates,
            crew_time_low=crew_time_low,
            optimizer_evals=optimizer_evals,
            use_ml=use_ml,
            seed=seed,
        )
