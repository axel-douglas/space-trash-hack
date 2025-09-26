"""
Model loading utilities for Rex-AI predictions.

Este módulo carga artefactos entrenados (pipeline sklearn + metadata) y expone
un registro de modelo con helpers de inferencia y explicabilidad livianos.
El diseño es robusto: si no hay modelo o falta metadata, devuelve fallbacks
seguros para que la UI nunca crashee.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:  # Optional runtime for accelerated inference
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except Exception:  # pragma: no cover - optional dependency missing
    ort = None  # type: ignore[assignment]
    HAS_ONNXRUNTIME = False

try:  # Optional dependency for embeddings
    import torch
    from torch import nn

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False

LOGGER = logging.getLogger(__name__)

# Rutas estándar
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MODEL_DIR = DATA_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "rexai_regressor.joblib"
METADATA_PATH = MODEL_DIR / "metadata_gold.json"
LEGACY_METADATA_PATH = MODEL_DIR / "metadata.json"
XGBOOST_PATH = MODEL_DIR / "rexai_xgboost.joblib"   # opcional
AUTOENCODER_PATH = MODEL_DIR / "rexai_autoencoder.pt"
TABTRANSFORMER_PATH = MODEL_DIR / "rexai_tabtransformer.pt"
TIGHTNESS_CLASSIFIER_PATH = MODEL_DIR / "rexai_class_tightness.joblib"
RIGIDITY_CLASSIFIER_PATH = MODEL_DIR / "rexai_class_rigidity.joblib"
LIGHTGBM_ONNX_PATH = MODEL_DIR / "rexai_lightgbm.onnx"

# Orden y nombres de objetivos que espera la UI
TARGET_COLUMNS: List[str] = ["rigidez", "estanqueidad", "energy_kwh", "water_l", "crew_min"]

DEFAULT_TIGHTNESS_SCORE_MAP = {0: 0.35, 1: 0.85}
DEFAULT_RIGIDITY_SCORE_MAP = {1: 0.35, 2: 0.65, 3: 0.9}


def _get_secret_value(key: str) -> str | None:
    value = os.environ.get(key)
    if value:
        return value

    secrets = getattr(st, "secrets", None)
    if secrets is None:
        return None

    try:
        secret_value = secrets[key]
    except FileNotFoundError:
        return None
    except (KeyError, AttributeError, TypeError):
        return None

    if secret_value is None:
        return None

    return str(secret_value)


def ensure_model_bundle(model_dir: Path | str = MODEL_DIR) -> None:
    """Descarga y extrae el bundle de modelos si está configurado."""

    url = _get_secret_value("MODEL_BUNDLE_URL")
    if not url:
        return

    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    expected_pipeline = target_dir / PIPELINE_PATH.name
    if expected_pipeline.exists():
        return

    LOGGER.info("Descargando bundle de modelos desde %s", url)

    expected_hash = _get_secret_value("MODEL_BUNDLE_SHA256")
    hasher = hashlib.sha256() if expected_hash else None

    tmp_path: Path | None = None
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    tmp_file.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)

        if hasher is not None:
            digest = hasher.hexdigest()
            if digest.lower() != expected_hash.strip().lower():
                LOGGER.warning(
                    "Hash SHA256 del bundle no coincide (esperado %s, obtenido %s)",
                    expected_hash,
                    digest,
                )
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                return

        if tmp_path is None:
            LOGGER.warning("Descarga del bundle de modelos desde %s no produjo archivo temporal", url)
            return

        try:
            with zipfile.ZipFile(tmp_path) as bundle:
                bundle.extractall(target_dir)
        except Exception as exc:
            LOGGER.warning("No se pudo extraer el bundle de modelos desde %s: %s", url, exc)
            return
    except Exception as exc:
        LOGGER.warning("No se pudo descargar el bundle de modelos desde %s: %s", url, exc)
        return
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.debug("No se pudo eliminar el archivo temporal %s", tmp_path)

    LOGGER.info("Bundle de modelos extraído en %s", target_dir)


def _parse_score_map(raw: Any, fallback: Dict[int, float]) -> Dict[int, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    parsed: Dict[int, float] = {}
    for key, value in raw.items():
        try:
            parsed[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed or dict(fallback)


@dataclass(slots=True)
class PredictionResult:
    rigidez: float
    estanqueidad: float
    energy_kwh: float
    water_l: float
    crew_min: float
    source: str
    metadata: Dict[str, Any]
    uncertainty: Dict[str, float]
    confidence_interval: Dict[str, Tuple[float, float]]
    feature_importance: List[Tuple[str, float]]
    comparisons: Dict[str, Dict[str, float]]
    latent_vector: Tuple[float, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rigidez": self.rigidez,
            "estanqueidad": self.estanqueidad,
            "energy_kwh": self.energy_kwh,
            "water_l": self.water_l,
            "crew_min": self.crew_min,
            "source": self.source,
            "metadata": self.metadata,
            "uncertainty": self.uncertainty,
            "confidence_interval": self.confidence_interval,
            "feature_importance": self.feature_importance,
            "comparisons": self.comparisons,
            "latent_vector": list(self.latent_vector),
        }


class ModelRegistry:
    """
    Registro simple para exponer un pipeline sklearn multi-salida entrenado.
    - No depende de PyTorch (opcional) para evitar bloqueos en Cloud.
    - Provee atributos usados por la UI: ready, feature_names, trained_label(), etc.
    """

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None                   # sklearn Pipeline
        self.preprocessor = None               # step 'preprocess' si existe
        self.metadata: Dict[str, Any] = {}     # metadata.json parseada
        self.feature_names: List[str] = []     # columnas post-transform
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.residual_std: np.ndarray | None = None
        self.feature_importance_avg: List[Tuple[str, float]] = []
        self.label_summary: Dict[str, Dict[str, Any]] = {}
        self.label_columns: Dict[str, str] = {}
        self.xgb_models: Dict[str, Any] = {}   # comparador opcional
        self.autoencoder = None
        self.autoencoder_meta: Dict[str, Any] = {}
        self.tabtransformer = None
        self.tab_meta: Dict[str, Any] = {}
        self.classifier_meta: Dict[str, Any] = {}
        self.tightness_clf = None
        self.rigidity_clf = None
        self.lightgbm_session = None
        self.lightgbm_input_name: str | None = None
        self.lightgbm_output_names: List[str] = []
        self.lightgbm_meta: Dict[str, Any] = {}
        self.tightness_classes: np.ndarray = np.array([])
        self.rigidity_classes: np.ndarray = np.array([])
        self.tightness_score_map: Dict[int, float] = dict(DEFAULT_TIGHTNESS_SCORE_MAP)
        self.rigidity_score_map: Dict[int, float] = dict(DEFAULT_RIGIDITY_SCORE_MAP)
        self._load()

    # -------------------------- Estado --------------------------------
    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def trained_label(self) -> str:
        metadata_label = self.metadata.get("trained_label")
        trained_on = self.metadata.get("trained_on")
        trained_at = self.metadata.get("trained_at")

        label_parts: List[str] = []

        if metadata_label:
            label_parts.append(str(metadata_label))
        elif trained_on:
            label_parts.append(str(trained_on))

        if trained_at:
            ts_label = str(trained_at)
            if isinstance(trained_at, str):
                try:
                    dt = datetime.fromisoformat(trained_at)
                except ValueError:
                    pass
                else:
                    ts_label = dt.strftime("%d %b %Y %H:%M UTC")
            label_parts.append(ts_label)

        return " · ".join(label_parts) if label_parts else "—"

    def uncertainty_label(self) -> str:
        # Texto para Home; si hay residual_std lo indicamos
        has_u = bool(self.metadata.get("residual_std"))
        return "reportada" if has_u else "no-reportada"

    # -------------------------- Carga ---------------------------------
    def _load(self) -> None:
        ensure_model_bundle(self.model_dir)
        # Pipeline
        pipeline_path = PIPELINE_PATH
        if not pipeline_path.exists():
            LOGGER.info("No hay modelo Rex-AI en %s, se inicializará uno demo", PIPELINE_PATH)
            generated_path: Path | str | None = None
            try:
                from app.modules.model_training import bootstrap_demo_model

                generated_path = bootstrap_demo_model()
            except Exception as exc:  # pragma: no cover - bootstrap opcional
                LOGGER.warning(
                    "Fallo bootstrap del modelo demo en %s: %s", PIPELINE_PATH, exc
                )
            if generated_path:
                pipeline_path = Path(generated_path)

        if pipeline_path.exists():
            try:
                self.pipeline = joblib.load(pipeline_path)
                # si es Pipeline([...('preprocess', ...), ('regressor', ...)])
                self.preprocessor = getattr(self.pipeline, "named_steps", {}).get("preprocess")
            except Exception as exc:
                LOGGER.warning("No se pudo cargar pipeline %s: %s", pipeline_path, exc)
                self.pipeline = None
                self.preprocessor = None
        else:
            LOGGER.info("No hay modelo Rex-AI en %s", pipeline_path)

        # Metadata
        metadata_path = METADATA_PATH if METADATA_PATH.exists() else LEGACY_METADATA_PATH
        if metadata_path.exists():
            try:
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.warning("Fallo parsing metadata %s: %s", metadata_path, exc)
                self.metadata = {}
        else:
            self.metadata = {}

        labeling = self.metadata.get("labeling") or {}
        if isinstance(labeling, dict):
            columns = labeling.get("columns", {})
            if isinstance(columns, dict):
                self.label_columns = {str(k): str(v) for k, v in columns.items()}

            summary_raw = labeling.get("summary", {})
            parsed_summary: Dict[str, Dict[str, Any]] = {}
            if isinstance(summary_raw, dict):
                for source, payload in summary_raw.items():
                    if not isinstance(payload, dict):
                        continue
                    try:
                        count_val = int(payload.get("count") or payload.get("n") or 0)
                    except (TypeError, ValueError):
                        count_val = 0
                    try:
                        mean_weight = float(payload.get("mean_weight") or payload.get("mean") or 0.0)
                    except (TypeError, ValueError):
                        mean_weight = 0.0
                    try:
                        min_weight = float(payload.get("min_weight") or payload.get("min") or 0.0)
                    except (TypeError, ValueError):
                        min_weight = 0.0
                    try:
                        max_weight = float(payload.get("max_weight") or payload.get("max") or 0.0)
                    except (TypeError, ValueError):
                        max_weight = 0.0

                    parsed_summary[str(source)] = {
                        "count": count_val,
                        "mean_weight": mean_weight,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                    }

            if parsed_summary:
                self.label_summary = parsed_summary

        # Fallbacks robustos
        feats = (
            self.metadata.get("post_transform_features")
            or self.metadata.get("feature_names")
            or self.metadata.get("feature_columns")
            or []
        )
        if isinstance(feats, dict):
            feats = list(feats.keys())
        self.feature_names = list(feats)

        self.feature_means = {
            str(k): float(v) for k, v in self.metadata.get("feature_means", {}).items()
        }
        self.feature_stds = {
            str(k): float(v) for k, v in self.metadata.get("feature_stds", {}).items()
        }
        residual_std = self.metadata.get("residual_std", {})
        self.residual_std = np.array(
            [float(residual_std.get(t, 0.0)) for t in TARGET_COLUMNS], dtype=float
        )

        imp_avg = (
            self.metadata.get("random_forest", {})
            .get("feature_importance", {})
            .get("average", [])
        )
        # lista de (nombre, peso)
        self.feature_importance_avg = [(str(n), float(w)) for n, w in imp_avg]

        # Ensemble opcional XGBoost (si existe)
        if XGBOOST_PATH.exists():
            try:
                payload = joblib.load(XGBOOST_PATH)
                self.xgb_models = payload.get("models", {})
            except Exception as exc:
                LOGGER.warning("No se pudo cargar ensemble XGBoost: %s", exc)
                self.xgb_models = {}

        self._load_autoencoder()
        self._load_classifiers()
        self._load_lightgbm()

    # ------------------------ Inferencia -------------------------------
    def predict(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Devuelve dict con predicciones + bandas de confianza + explicabilidad.
        Si no hay modelo cargado, retorna {} para que el caller use heurísticas.
        """
        if not self.ready:
            return {}

        try:
            frame, matrix = self._prepare_frame(features)
            preds = self.pipeline.predict(frame)  # type: ignore[union-attr]
        except Exception as exc:
            LOGGER.warning("Fallo en predicción; usar heurísticas. Motivo: %s", exc)
            return {}

        preds = np.asarray(preds, dtype=float).reshape(-1)
        if preds.size < len(TARGET_COLUMNS):
            LOGGER.warning("Tamaño de predicción inválido: %s", preds.size)
            return {}

        # Incertidumbre a partir de varianza entre árboles + residual de entrenamiento (si hay)
        tree_std = self._rf_uncertainty(matrix)
        combined_std = np.asarray(self._combine_uncertainty(tree_std), dtype=float)

        # Contribuciones (aprox) usando importancia media y desviación del mean
        importance = self._feature_contributions(matrix[0])

        # Comparadores opcionales (XGBoost)
        variants = self._predict_variants(matrix)
        classifier_variants = self._apply_classifiers(matrix, preds, combined_std)
        if classifier_variants:
            variants.update(classifier_variants)

        # Intervalos 95% (tras aplicar clasificadores)
        ci = self._confidence_interval(preds, combined_std)

        result = PredictionResult(
            rigidez=float(np.clip(preds[0], 0.0, 1.0)),
            estanqueidad=float(np.clip(preds[1], 0.0, 1.0)),
            energy_kwh=float(max(0.0, preds[2])),
            water_l=float(max(0.0, preds[3])),
            crew_min=float(max(0.0, preds[4])),
            source=str(self.metadata.get("model_name", "rexai-rf-ensemble")),
            metadata={
                "trained_at": self.metadata.get("trained_at"),
                "n_samples": self.metadata.get("n_samples"),
                "features": self.feature_names,
                "targets": TARGET_COLUMNS,
                "label_summary": self.label_summary,
                "label_columns": self.label_columns,
            },
            uncertainty={t: float(combined_std[i]) for i, t in enumerate(TARGET_COLUMNS)},
            confidence_interval=ci,
            feature_importance=importance,
            comparisons=variants,
            latent_vector=(),  # sin PyTorch mantemos vacío
        )
        return result.as_dict()

    def label_distribution_label(self) -> str:
        if not self.label_summary:
            return "—"

        def _sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
            count = item[1].get("count")
            try:
                sortable = -int(count) if count is not None else 0
            except (TypeError, ValueError):
                sortable = 0
            return sortable, str(item[0])

        parts: List[str] = []
        for source, stats in sorted(self.label_summary.items(), key=_sort_key):
            label = str(source)
            count = stats.get("count")
            mean_weight = stats.get("mean_weight")
            fragment = label
            try:
                if count is not None:
                    fragment = f"{label}×{int(count)}"
            except (TypeError, ValueError):
                fragment = label

            try:
                if mean_weight is not None:
                    fragment = f"{fragment} (w≈{float(mean_weight):.2f})"
            except (TypeError, ValueError):
                pass

            parts.append(fragment)

        return " · ".join(parts)

    # ---------------------- helpers internos --------------------------
    def _prepare_frame(self, features: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
        frame = pd.DataFrame([features])
        if self.preprocessor is None:
            # Sin preprocesador, devolvemos zeros para longitud consistente
            return frame, np.zeros((1, max(1, len(self.feature_names))), dtype=float)

        matrix = self.preprocessor.transform(frame)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return frame, np.asarray(matrix, dtype=float)

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        """Transforma un *DataFrame* usando el preprocesador entrenado."""

        if frame.empty:
            return np.zeros((0, max(1, len(self.feature_names))), dtype=float)

        if self.preprocessor is None:
            return np.zeros((len(frame), max(1, len(self.feature_names))), dtype=float)

        matrix = self.preprocessor.transform(frame)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=float)

    def has_autoencoder(self) -> bool:
        return bool(HAS_TORCH and self.autoencoder is not None and self.preprocessor is not None)

    def encode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((matrix.shape[0], 0), dtype=float)

        if not self.has_autoencoder():
            return np.zeros((matrix.shape[0], 0), dtype=float)

        tensor = torch.tensor(matrix, dtype=torch.float32)
        with torch.no_grad():
            latent = self.autoencoder.encode(tensor).cpu().numpy()
        return np.asarray(latent, dtype=float)

    def decode_latent(self, latent: Sequence[float]) -> Dict[str, Any]:
        if not latent:
            return {}

        if not self.has_autoencoder():
            return {}

        vector = np.asarray(latent, dtype=float).reshape(1, -1)
        tensor = torch.tensor(vector, dtype=torch.float32)
        try:
            with torch.no_grad():
                decoded = self.autoencoder.decode(tensor).cpu().numpy()
        except Exception as exc:  # pragma: no cover - fallos raros de torch
            LOGGER.warning("Fallo decodificando vector latente: %s", exc)
            return {}

        try:
            original = self.preprocessor.inverse_transform(decoded)
        except Exception as exc:  # pragma: no cover - transformadores sin inverse
            LOGGER.warning("No se pudo invertir el preprocesamiento: %s", exc)
            return {}

        if isinstance(original, pd.DataFrame):
            row = original.iloc[0].to_dict()
        else:
            arr = np.asarray(original)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            names: Iterable[str]
            names = getattr(self.preprocessor, "feature_names_in_", None) or []
            if not names:
                meta_columns = self.metadata.get("feature_columns") or []
                names = [str(col) for col in meta_columns]
            names = list(names)
            if len(names) < arr.shape[1]:
                names.extend(str(i) for i in range(len(names), arr.shape[1]))
            row = {str(names[i]): arr[0, i] for i in range(arr.shape[1])}

        feature_columns = self.metadata.get("feature_columns") or list(row.keys())
        cleaned: Dict[str, Any] = {}
        for column in feature_columns:
            value = row.get(column)
            if column == "process_id":
                if value is None or value == "":
                    continue
                cleaned[column] = str(value).strip().upper()
                continue

            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue

            if column.endswith("_frac"):
                numeric = float(np.clip(numeric, 0.0, 1.0))
            elif column.endswith("_pct"):
                numeric = float(np.clip(numeric, 0.0, 100.0))
            else:
                numeric = float(max(numeric, 0.0))
            cleaned[column] = numeric

        return cleaned

    def _rf_uncertainty(self, matrix: np.ndarray) -> np.ndarray:
        """Desviación entre árboles del RandomForestRegressor multi-salida."""
        regressor = getattr(self.pipeline, "named_steps", {}).get("regressor") if self.pipeline else None
        if regressor is None or not hasattr(regressor, "estimators_"):
            return np.zeros((matrix.shape[0], len(TARGET_COLUMNS)), dtype=float)

        stds = np.zeros((matrix.shape[0], len(TARGET_COLUMNS)), dtype=float)
        # estimators_ es una lista de RandomForestRegressor, uno por target
        for target_idx, estimator in enumerate(regressor.estimators_):
            try:
                tree_preds = np.stack([tree.predict(matrix) for tree in estimator.estimators_], axis=0)
                stds[:, target_idx] = tree_preds.std(axis=0)
            except Exception:
                stds[:, target_idx] = 0.0
        return stds

    def _combine_uncertainty(self, tree_std: np.ndarray) -> np.ndarray:
        """Combina varianza entre árboles con std residual del entrenamiento."""
        residual = self.residual_std if self.residual_std is not None else np.zeros(len(TARGET_COLUMNS))
        residual = residual.reshape(1, -1)
        # Proteger longitudes
        if residual.shape[1] < tree_std.shape[1]:
            residual = np.pad(residual, ((0, 0), (0, tree_std.shape[1] - residual.shape[1])), constant_values=0.0)
        return np.sqrt(tree_std**2 + residual**2)[0]

    def _confidence_interval(self, preds: np.ndarray, std: np.ndarray) -> Dict[str, Tuple[float, float]]:
        ci: Dict[str, Tuple[float, float]] = {}
        for i, t in enumerate(TARGET_COLUMNS):
            sigma = float(std[i])
            lo = preds[i] - 1.96 * sigma
            hi = preds[i] + 1.96 * sigma
            if t in {"rigidez", "estanqueidad"}:
                lo = float(np.clip(lo, 0.0, 1.0))
                hi = float(np.clip(hi, 0.0, 1.0))
            else:
                lo = float(max(0.0, lo))
                hi = float(max(0.0, hi))
            ci[t] = (lo, hi)
        return ci

    def _feature_contributions(self, vector: np.ndarray) -> List[Tuple[str, float]]:
        if not self.feature_importance_avg or not self.feature_names:
            return []
        contribs: List[Tuple[str, float]] = []
        idx = {name: i for i, name in enumerate(self.feature_names)}
        for name, weight in self.feature_importance_avg[:8]:
            i = idx.get(name)
            if i is None:
                continue
            val = float(vector[i])
            mean = self.feature_means.get(name, 0.0)
            contribs.append((name, float((val - mean) * weight)))
        return contribs

    def _predict_variants(self, matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Predicciones alternativas con modelos adicionales."""
        out: Dict[str, Dict[str, float]] = {}

        if self.xgb_models:
            preds = []
            for t in TARGET_COLUMNS:
                model = self.xgb_models.get(t)
                if model is None:
                    preds.append(np.zeros(matrix.shape[0]))
                else:
                    preds.append(np.asarray(model.predict(matrix), dtype=float))
            stacked = np.stack(preds, axis=1)[0]
            out["xgboost"] = {
                t: (
                    float(np.clip(stacked[i], 0.0, 1.0))
                    if t in {"rigidez", "estanqueidad"}
                    else float(max(0.0, stacked[i]))
                )
                for i, t in enumerate(TARGET_COLUMNS)
            }

        if self.lightgbm_session is not None and self.lightgbm_input_name:
            try:
                matrix32 = np.asarray(matrix, dtype=np.float32)
                outputs = self.lightgbm_session.run(
                    None, {self.lightgbm_input_name: matrix32}
                )
                if outputs:
                    values = np.asarray(outputs[0], dtype=float)
                    if values.ndim == 1:
                        values = values.reshape(1, -1)
                    if values.shape[0] > 0:
                        row = values[0]
                        out["lightgbm_gpu"] = {
                            target: (
                                float(np.clip(row[idx], 0.0, 1.0))
                                if target in {"rigidez", "estanqueidad"}
                                else float(max(0.0, row[idx]))
                            )
                            for idx, target in enumerate(TARGET_COLUMNS)
                        }
            except Exception as exc:  # pragma: no cover - runtime errors raros
                LOGGER.warning("Fallo inferencia LightGBM ONNX: %s", exc)

        return out

    def _apply_classifiers(
        self,
        matrix: np.ndarray,
        preds: np.ndarray,
        std: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        updates: Dict[str, Dict[str, float]] = {}

        if self.tightness_clf is not None:
            try:
                probabilities = self.tightness_clf.predict_proba(matrix)
                if probabilities.ndim == 2 and probabilities.shape[0] > 0:
                    row = probabilities[0]
                    classes = self.tightness_classes if self.tightness_classes.size else np.arange(len(row))
                    score_map = self.tightness_score_map or dict(DEFAULT_TIGHTNESS_SCORE_MAP)
                    fallback = next(iter(score_map.values())) if score_map else 0.5

                    expected = 0.0
                    prob_map: Dict[str, float] = {}
                    for cls_val, prob in zip(classes, row):
                        cls_int = int(cls_val)
                        score = float(score_map.get(cls_int, fallback))
                        prob_f = float(prob)
                        expected += score * prob_f
                        prob_map[str(cls_int)] = prob_f

                    variance = 0.0
                    for cls_val, prob in zip(classes, row):
                        cls_int = int(cls_val)
                        score = float(score_map.get(cls_int, fallback))
                        variance += (score - expected) ** 2 * float(prob)

                    preds[1] = float(np.clip(expected, 0.0, 1.0))
                    if variance > 0.0:
                        std[1] = float(np.sqrt(max(variance, 1e-9)))

                    updates["tightness_classifier"] = {
                        "pass_prob": float(prob_map.get("1", 0.0)),
                        "expected": float(preds[1]),
                    }
            except Exception as exc:
                LOGGER.warning("Fallo clasificador estanqueidad: %s", exc)

        if self.rigidity_clf is not None:
            try:
                probabilities = self.rigidity_clf.predict_proba(matrix)
                if probabilities.ndim == 2 and probabilities.shape[0] > 0:
                    row = probabilities[0]
                    classes = self.rigidity_classes if self.rigidity_classes.size else np.arange(len(row))
                    score_map = self.rigidity_score_map or dict(DEFAULT_RIGIDITY_SCORE_MAP)
                    fallback = next(iter(score_map.values())) if score_map else 0.6

                    expected = 0.0
                    prob_map: Dict[str, float] = {}
                    for cls_val, prob in zip(classes, row):
                        cls_int = int(cls_val)
                        score = float(score_map.get(cls_int, fallback))
                        prob_f = float(prob)
                        expected += score * prob_f
                        prob_map[str(cls_int)] = prob_f

                    variance = 0.0
                    for cls_val, prob in zip(classes, row):
                        cls_int = int(cls_val)
                        score = float(score_map.get(cls_int, fallback))
                        variance += (score - expected) ** 2 * float(prob)

                    preds[0] = float(np.clip(expected, 0.0, 1.0))
                    if variance > 0.0:
                        std[0] = float(np.sqrt(max(variance, 1e-9)))

                    if prob_map:
                        dominant = max(prob_map.items(), key=lambda kv: kv[1])
                        updates["rigidity_classifier"] = {
                            "level": float(int(dominant[0])),
                            "confidence": float(dominant[1]),
                        }
            except Exception as exc:
                LOGGER.warning("Fallo clasificador rigidez: %s", exc)

        return updates

    def embed(self, features: Mapping[str, Any]) -> Tuple[float, ...]:
        if not self.has_autoencoder():
            return ()

        try:
            frame = pd.DataFrame([features])
            matrix = self.transform_features(frame)
            latent = self.encode_matrix(matrix)
            if latent.size == 0:
                return ()
            return tuple(float(x) for x in latent.reshape(-1))
        except Exception as exc:
            LOGGER.warning("Fallo generando embedding: %s", exc)
            return ()

    def _load_autoencoder(self) -> None:
        if not HAS_TORCH:
            self.autoencoder = None
            return

        meta = self.metadata.get("artifacts", {}).get("autoencoder", {})
        if not meta:
            self.autoencoder = None
            return

        path = meta.get("path")
        model_path = self.model_dir / path if path else AUTOENCODER_PATH
        if not model_path.exists():
            self.autoencoder = None
            return

        feature_count = len(self.feature_names)
        if feature_count == 0:
            self.autoencoder = None
            return

        latent_dim = int(meta.get("latent_dim", 12))

        try:
            model = _Autoencoder(feature_count, latent_dim)
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.autoencoder = model
            self.autoencoder_meta = meta
        except Exception as exc:
            LOGGER.warning("No se pudo cargar autoencoder: %s", exc)
            self.autoencoder = None

    def _load_classifiers(self) -> None:
        meta = self.metadata.get("classifiers", {})
        if not isinstance(meta, dict):
            meta = {}
        self.classifier_meta = meta

        tight_meta = meta.get("tightness_pass", {}) if isinstance(meta.get("tightness_pass"), dict) else {}
        rigid_meta = meta.get("rigidity_level", {}) if isinstance(meta.get("rigidity_level"), dict) else {}

        self.tightness_score_map = _parse_score_map(
            tight_meta.get("score_map"), DEFAULT_TIGHTNESS_SCORE_MAP
        )
        self.rigidity_score_map = _parse_score_map(
            rigid_meta.get("score_map"), DEFAULT_RIGIDITY_SCORE_MAP
        )

        tight_path = tight_meta.get("path") if isinstance(tight_meta, dict) else None
        tight_model_path = self.model_dir / tight_path if tight_path else TIGHTNESS_CLASSIFIER_PATH
        if tight_model_path.exists():
            try:
                self.tightness_clf = joblib.load(tight_model_path)
                classes = getattr(self.tightness_clf, "classes_", None)
                self.tightness_classes = np.asarray(classes, dtype=float) if classes is not None else np.array([])
            except Exception as exc:
                LOGGER.warning("No se pudo cargar clasificador tightness: %s", exc)
                self.tightness_clf = None
                self.tightness_classes = np.array([])
        else:
            self.tightness_clf = None
            self.tightness_classes = np.array([])

        rigid_path = rigid_meta.get("path") if isinstance(rigid_meta, dict) else None
        rigid_model_path = self.model_dir / rigid_path if rigid_path else RIGIDITY_CLASSIFIER_PATH
        if rigid_model_path.exists():
            try:
                self.rigidity_clf = joblib.load(rigid_model_path)
                classes = getattr(self.rigidity_clf, "classes_", None)
                self.rigidity_classes = np.asarray(classes, dtype=float) if classes is not None else np.array([])
            except Exception as exc:
                LOGGER.warning("No se pudo cargar clasificador rigidez: %s", exc)
                self.rigidity_clf = None
                self.rigidity_classes = np.array([])
        else:
            self.rigidity_clf = None
            self.rigidity_classes = np.array([])

    def _load_lightgbm(self) -> None:
        meta = self.metadata.get("artifacts", {}).get("lightgbm_gpu", {})
        if not isinstance(meta, dict):
            meta = {}

        self.lightgbm_meta = meta
        self.lightgbm_session = None
        self.lightgbm_input_name = None
        self.lightgbm_output_names = []

        if not HAS_ONNXRUNTIME:
            return

        path = meta.get("path") if isinstance(meta, dict) else None
        model_path = self._resolve_artifact_path(path, LIGHTGBM_ONNX_PATH)
        if model_path is None or not model_path.exists():
            return

        try:
            session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - runtime errors are rare
            LOGGER.warning("No se pudo cargar modelo ONNX LightGBM: %s", exc)
            return

        self.lightgbm_session = session
        inputs = session.get_inputs()
        if inputs:
            self.lightgbm_input_name = inputs[0].name
        outputs = session.get_outputs()
        self.lightgbm_output_names = [out.name for out in outputs] if outputs else []

    def _resolve_artifact_path(self, candidate: str | None, default: Path) -> Path | None:
        if candidate:
            candidate_path = Path(candidate)
            search_order = []
            if candidate_path.is_absolute():
                search_order.append(candidate_path)
            else:
                search_order.append(self.model_dir / candidate_path)
                search_order.append((DATA_ROOT.parent / candidate_path).resolve())
        else:
            search_order = []

        search_order.append(default)

        for option in search_order:
            try:
                resolved = option if option.is_absolute() else option.resolve()
            except FileNotFoundError:
                resolved = option
            if resolved.exists():
                return resolved
        return None


class _Autoencoder(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        if not HAS_TORCH:  # pragma: no cover - sin torch no se instancia
            raise RuntimeError("PyTorch es requerido para embeddings")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 192),
            nn.ReLU(),
            nn.Linear(192, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


# Instancia global usada por la app
MODEL_REGISTRY = ModelRegistry()

__all__ = ["MODEL_REGISTRY", "ModelRegistry", "PredictionResult"]
