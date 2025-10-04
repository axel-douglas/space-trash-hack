"""CLI helper to generate candidate recipes and persist them as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _ensure_project_root() -> Path:
    """Ensure the repository root is available on ``sys.path`` when run as a script."""

    module_path = Path(__file__).resolve()
    root = module_path.parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


_ensure_project_root()

from app.modules.generator import PredProps, generate_candidates
from app.modules.io import load_process_df, load_targets, load_waste_df
from app.modules.ranking import rank_candidates

DEFAULT_OUTPUT = Path("data/candidates.json")


def _load_target(args: argparse.Namespace) -> Dict[str, Any]:
    if args.target_file:
        payload = json.loads(Path(args.target_file).read_text(encoding="utf-8"))
        return {
            "name": payload.get("name", "custom"),
            "rigidity": float(payload.get("rigidity", 0.75)),
            "tightness": float(payload.get("tightness", 0.75)),
            "max_energy_kwh": float(payload.get("max_energy_kwh", 8.0)),
            "max_water_l": float(payload.get("max_water_l", 5.0)),
            "max_crew_min": float(payload.get("max_crew_min", 60.0)),
            "crew_time_low": bool(payload.get("crew_time_low", False)),
        }

    presets = load_targets()
    if not presets:
        raise RuntimeError("No se encontraron targets en data/targets_presets.json")

    if args.target_name:
        for preset in presets:
            if preset.get("name") == args.target_name:
                base = preset
                break
        else:  # pragma: no cover - defensive
            raise ValueError(f"Preset '{args.target_name}' no encontrado en data/targets_presets.json")
    else:
        base = presets[0]

    target = {
        "name": base.get("name", "preset"),
        "rigidity": float(args.rigidity if args.rigidity is not None else base.get("rigidity", 0.75)),
        "tightness": float(args.tightness if args.tightness is not None else base.get("tightness", 0.75)),
        "max_energy_kwh": float(args.max_energy if args.max_energy is not None else base.get("max_energy_kwh", 8.0)),
        "max_water_l": float(args.max_water if args.max_water is not None else base.get("max_water_l", 5.0)),
        "max_crew_min": float(args.max_crew if args.max_crew is not None else base.get("max_crew_min", 60.0)),
        "crew_time_low": bool(args.crew_time_low if args.crew_time_low is not None else base.get("crew_time_low", False)),
    }
    return target


def _load_waste(path: str | None) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        return df
    return load_waste_df()


def _load_processes(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    return load_process_df()


def _serialize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, PredProps):
            return value.as_dict()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_convert(v) for v in value]
        return value

    return {k: _convert(v) for k, v in candidate.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera recetas Rex-AI y las serializa a JSON")
    parser.add_argument("--waste", help="Ruta opcional a inventario de residuos en CSV")
    parser.add_argument("--processes", help="Ruta opcional al catálogo de procesos en CSV")
    parser.add_argument("--target-file", dest="target_file", help="Archivo JSON con objetivo personalizado")
    parser.add_argument("--target-name", dest="target_name", help="Nombre del preset de target a usar")
    parser.add_argument("--rigidity", type=float, help="Override de rigidez objetivo")
    parser.add_argument("--tightness", type=float, help="Override de estanqueidad objetivo")
    parser.add_argument("--max-energy", dest="max_energy", type=float, help="Energía máxima kWh")
    parser.add_argument("--max-water", dest="max_water", type=float, help="Agua máxima L")
    parser.add_argument("--max-crew", dest="max_crew", type=float, help="Tiempo crew máximo (min)")
    parser.add_argument("--crew-time-low", dest="crew_time_low", action="store_true", help="Priorizar recetas con poco tiempo de crew")
    parser.add_argument("--n", type=int, default=120, help="Cantidad de candidatos a generar")
    parser.add_argument("--optimizer-evals", dest="optimizer_evals", type=int, default=0, help="Iteraciones de optimización adicional")
    parser.add_argument("--heuristic", action="store_true", help="Forzar modo heurístico (sin ML)")
    parser.add_argument("--top", type=int, default=20, help="Recortar top-N recetas")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Archivo JSON de salida")
    parser.add_argument("--seed", type=int, help="Semilla RNG opcional para reproducibilidad")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = _load_target(args)

    waste_df = _load_waste(args.waste)
    process_df = _load_processes(args.processes)

    candidates, history = generate_candidates(
        waste_df,
        process_df,
        target,
        n=args.n,
        crew_time_low=target.get("crew_time_low", False),
        optimizer_evals=args.optimizer_evals,
        use_ml=not args.heuristic,
        seed=args.seed,
    )

    ranked = rank_candidates(
        candidates,
        target,
        top_n=args.top,
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "target": target,
        "use_ml": not args.heuristic,
        "n_requested": args.n,
        "optimizer_evals": args.optimizer_evals,
        "seed": args.seed,
        "candidates": [_serialize_candidate(c) for c in ranked],
        "history": history.to_dict(orient="records") if isinstance(history, pd.DataFrame) else [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Se guardaron {len(ranked)} candidatos en {args.output}")


if __name__ == "__main__":
    main()
