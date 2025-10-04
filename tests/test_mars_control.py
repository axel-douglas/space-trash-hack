import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import yaml

from app.modules import mars_control


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    # Ensure every test starts from a clean baseline cache.
    monkeypatch.setattr(mars_control, "_BASELINE_CACHE", None, raising=False)
    monkeypatch.setattr(mars_control, "_JEZERO_GEODATA_CACHE", None, raising=False)


def test_load_logistics_baseline_parses_yaml(tmp_path, monkeypatch):
    dataset = {
        "flights": [
            {
                "flight_id": "MC-999",
                "capsule_id": "ares",
                "origin": "Hub",
                "destination": "Base",
                "departure": "2043-04-12T06:30:00Z",
                "arrival": "2043-04-12T18:05:00Z",
                "status": "en_route",
                "payload_mass_kg": 1234.5,
                "manifest_ref": "manifest-test",
            }
        ],
        "capsules": [
            {
                "capsule_id": "ares",
                "name": "Ares",
                "capacity_kg": 1500,
                "status": "active",
                "location": "Orbit",
                "notes": "Ready",
            }
        ],
        "processes": [
            {
                "process_id": "refinery",
                "name": "Refinery",
                "throughput_kg_per_day": 50,
                "energy_kwh_per_kg": 1.2,
                "crew_hours_per_day": 5,
            }
        ],
        "orders": [
            {
                "order_id": "ia-1",
                "issued_at": "2043-04-12T05:00:00Z",
                "priority": "high",
                "directive": "Priorizar",
                "target": "manifest-test",
                "context": "Testing",
            }
        ],
        "event_templates": {"inbound": [{"title": "Arribo", "description": "Desc", "delta_mass_kg": 42}]},
    }
    dataset_path = tmp_path / "mars_logistics.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset), encoding="utf-8")

    monkeypatch.setattr(mars_control, "DATA_ROOT", tmp_path, raising=False)

    data = mars_control.load_logistics_baseline(refresh=True)

    assert len(data.flights) == 1
    flight = data.flights[0]
    assert flight.flight_id == "MC-999"
    assert flight.capsule_id == "ares"
    assert flight.payload_mass_kg == pytest.approx(1234.5)
    assert isinstance(flight.departure, datetime)
    assert flight.departure.tzinfo == timezone.utc

    assert len(data.capsules) == 1
    capsule = data.capsules[0]
    assert capsule.capacity_kg == pytest.approx(1500.0)
    assert capsule.notes == "Ready"

    assert len(data.processes) == 1
    process = data.processes[0]
    assert process.energy_kwh_per_kg == pytest.approx(1.2)

    assert len(data.ai_orders) == 1
    order = data.ai_orders[0]
    assert order.priority == "high"
    assert order.target == "manifest-test"

    assert "inbound" in data.event_templates
    assert data.event_templates["inbound"][0]["title"] == "Arribo"


def test_aggregate_inventory_by_category_returns_flows():
    inventory = pd.DataFrame(
        [
            {
                "category": "Foam Packaging",
                "material_family": "PVDF",
                "flags": "foam;pvdf",
                "key_materials": "PVDF 85%; binder",
                "mass_kg": 120.0,
                "volume_l": 320.0,
                "moisture_pct": 5,
                "difficulty_factor": 1.5,
            },
            {
                "category": "Metal Struts",
                "material_family": "Alloy",
                "flags": "structural",
                "key_materials": "Ti-6Al-4V",
                "mass_kg": 80.0,
                "volume_l": 150.0,
                "moisture_pct": 1,
                "difficulty_factor": 2.2,
                "_problematic": True,
            },
        ]
    )

    payload = mars_control.aggregate_inventory_by_category(inventory)

    normalized = payload["normalized"]
    categories = payload["categories"]
    flows = payload["flows"]

    assert not normalized.empty
    assert set(["purity_index", "cross_contamination_risk", "recycle_mass_kg"]).issubset(normalized.columns)

    assert categories.shape[0] == 2
    assert categories["total_mass_kg"].sum() == pytest.approx(200.0)
    assert {"Foam Packaging", "Metal Struts"} == set(categories["category"])

    assert not flows.empty
    assert flows.groupby("destination_key")["mass_kg"].sum().sum() == pytest.approx(200.0)
    assert set(payload["material_groups"]).issuperset({"espumas", "metales"})
    assert set(payload["destinations"]).issuperset({"recycle", "reuse", "stock"})


def test_apply_simulation_tick_generates_events(monkeypatch):
    logistics = mars_control.MarsLogisticsData(
        flights=[],
        capsules=[],
        processes=[],
        ai_orders=[],
        event_templates={
            "inbound": [
                {
                    "title": "Arribo parcial",
                    "description": "Carga recibida",
                    "delta_mass_kg": 100,
                    "capsule_id": "ares",
                }
            ],
            "orders": [
                {
                    "title": "Orden IA",
                    "description": "Revisar",
                    "reference": "ia-1",
                }
            ],
        },
    )

    monkeypatch.setattr(mars_control, "load_logistics_baseline", lambda: logistics)
    monkeypatch.setattr(
        mars_control,
        "compute_mission_summary",
        lambda inventory=None, logistics=None: {"mass_kg": 320.0},
    )

    session: dict[str, object] = {}

    first_tick = mars_control.apply_simulation_tick(session=session)
    assert len(first_tick) == 2
    assert first_tick[0].tick == 1
    assert first_tick[0].category == "inbound"
    assert first_tick[0].metadata["mass_delta"] == pytest.approx(100.0)

    cached_tick = mars_control.apply_simulation_tick(session=session)
    assert len(cached_tick) == 2
    assert {event.tick for event in cached_tick} == {2}

    injected = mars_control.apply_simulation_tick(
        {"inject_event": {"category": "manual", "title": "Test", "details": "Injected"}},
        session=session,
    )
    assert len(injected) == 3
    assert any(event.category == "manual" for event in injected)

    history = mars_control.iterate_events(since_tick=1, session=session)
    assert all(event.tick >= 2 for event in history)


def test_score_manifest_batch_builds_policy_summary():
    policy_df = pd.DataFrame(
        [
            {
                "item_name": "PVDF Foam",
                "current_score": 0.4,
                "recommended_score": 0.72,
                "recommended_quota": 0.6,
                "action": "substitute",
                "justification": "Compatibilidad validada",
            },
            {
                "item_name": "Metal Strut",
                "current_score": 0.55,
                "recommended_score": 0.7,
                "recommended_quota": 0.4,
                "action": "reprioritize",
                "justification": "Reducir penalización",
            },
        ]
    )

    manifest_df = pd.DataFrame(
        [
            {"item": "PVDF Foam", "mass_kg": 10.0, "material_utility_score": 0.4},
            {"item": "Metal Strut", "mass_kg": 15.0, "material_utility_score": 0.55},
        ]
    )

    class FakeService:
        def analyze_manifest(self, manifest):  # pragma: no cover - simple stub
            return {
                "manifest": manifest_df,
                "scored_manifest": manifest_df,
                "policy_recommendations": policy_df,
                "material_passport": {
                    "total_items": 2,
                    "total_mass_kg": 25.0,
                    "mean_material_utility_score": 0.475,
                },
            }

    results = mars_control.score_manifest_batch(FakeService(), [manifest_df])
    assert len(results) == 1
    summary = results[0]["summary"]
    policy_summary = summary["policy_summary"]

    assert summary["total_mass_kg"] == pytest.approx(25.0)
    assert summary["item_count"] == 2
    assert policy_summary["total_actions"] == 2
    assert policy_summary["mean_score_gain"] == pytest.approx(0.235, rel=1e-6)
    assert {"substitute", "reprioritize"} == set(policy_summary["actions_by_type"])
    assert len(policy_summary["top_actions"]) >= 1

    serialized = json.dumps(results[0]["material_passport"])
    round_trip = json.loads(serialized)
    assert round_trip["total_items"] == 2
    assert round_trip["total_mass_kg"] == pytest.approx(25.0)
