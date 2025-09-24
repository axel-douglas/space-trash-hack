"""Re-score candidate JSON payloads with custom weights and penalties."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.modules.generator import PredProps
from app.modules.ranking import rank_candidates

DEFAULT_OUTPUT = Path("data/candidates_ranked.json")


def _load_payload(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "candidates" not in data or "target" not in data:
        raise ValueError("El JSON debe contener 'candidates' y 'target'")
    return data


def _parse_mapping(value: str | None) -> Dict[str, float]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("El argumento debe ser un objeto JSON {clave: valor}")
    return {str(k): float(v) for k, v in parsed.items()}


def _prepare_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for cand in candidates:
        base = dict(cand)
        props_payload = base.get("props")
        if isinstance(props_payload, dict):
            base["props"] = PredProps.from_mapping(props_payload)
        prepared.append(base)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-ranking multiobjetivo de recetas Rex-AI")
    parser.add_argument("input", type=Path, help="Archivo JSON con candidatos generados")
    parser.add_argument("--weights", help="JSON inline con pesos personalizados (ej. '{\\"rigidez\\":1.5}')")
    parser.add_argument("--penalties", help="JSON inline con penalizaciones personalizadas")
    parser.add_argument("--top", type=int, default=20, help="Cantidad de recetas a conservar")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta de salida JSON")
    parser.add_argument("--summary", action="store_true", help="Imprimir tabla resumen en stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.input)
    target = payload.get("target", {})
    weights = _parse_mapping(args.weights)
    penalties = _parse_mapping(args.penalties)

    candidates = _prepare_candidates(payload.get("candidates", []))
    ranked = rank_candidates(
        candidates,
        target,
        weights=weights,
        penalties=penalties,
        top_n=args.top,
        as_dict=True,
    )

    summary_rows = []
    for idx, cand in enumerate(ranked, start=1):
        aux = cand.get("auxiliary", {}) or {}
        props = cand.get("props", {}) or {}
        summary_rows.append(
            {
                "rank": idx,
                "score": round(float(cand.get("score", 0.0)), 3),
                "process": f"{cand.get('process_id', '')} · {cand.get('process_name', '')}",
                "rigidez": round(float(props.get("rigidez", 0.0)), 3),
                "estanqueidad": round(float(props.get("estanqueidad", 0.0)), 3),
                "energy_kwh": round(float(props.get("energy_kwh", 0.0)), 3),
                "water_l": round(float(props.get("water_l", 0.0)), 3),
                "crew_min": round(float(props.get("crew_min", 0.0)), 3),
                "passes_seal": bool(aux.get("passes_seal", True)),
                "process_risk": round(float(aux.get("process_risk", 0.0)), 3),
                "risk_label": aux.get("process_risk_label", "n/a"),
            }
        )

    output_payload = {
        **payload,
        "ranked_at": datetime.utcnow().isoformat(),
        "weights": weights,
        "penalties": penalties,
        "top_n": args.top,
        "candidates": ranked,
        "summary": summary_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.summary:
        for row in summary_rows:
            print(
                f"#{row['rank']:02d} | score={row['score']:.3f} | rigidez={row['rigidez']:.2f} | "
                f"estanqueidad={row['estanqueidad']:.2f} | energía={row['energy_kwh']:.2f} kWh | "
                f"agua={row['water_l']:.2f} L | crew={row['crew_min']:.1f} min | "
                f"seal={'✅' if row['passes_seal'] else '⚠️'} | riesgo={row['risk_label']}"
            )

    print(f"Ranking actualizado guardado en {args.output}")


if __name__ == "__main__":
    main()
