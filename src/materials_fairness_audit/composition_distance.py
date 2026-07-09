from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import warnings

import numpy as np
from pymatgen.core import Composition, Element
from sklearn.neighbors import NearestNeighbors


ELEMENT_COUNT = 118


def composition_vector_from_formula(formula: str) -> np.ndarray:
    return composition_vector_from_mapping(Composition(formula).fractional_composition.as_dict())


def composition_vector_from_mapping(mapping: Mapping[str, float]) -> np.ndarray:
    vector = np.zeros(ELEMENT_COUNT, dtype=np.float32)
    total = float(sum(float(amount) for amount in mapping.values()))
    if total <= 0:
        return vector
    for symbol, amount in mapping.items():
        vector[Element(symbol).Z - 1] = float(amount) / total
    return vector


def build_formula_matrix(formulas: Sequence[str]) -> np.ndarray:
    return np.vstack([composition_vector_from_formula(formula) for formula in formulas]).astype(np.float32)


def nearest_neighbor_distances(
    train_matrix: np.ndarray,
    query_matrix: np.ndarray,
    *,
    chunk_size: int = 5000,
    metric: str = "euclidean",
    algorithm: str = "kd_tree",
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_matrix) == 0 or len(query_matrix) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    model = NearestNeighbors(n_neighbors=1, metric=metric, algorithm=algorithm)
    model.fit(train_matrix)

    all_distances: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for start in range(0, len(query_matrix), chunk_size):
        stop = min(start + chunk_size, len(query_matrix))
        distances, indices = model.kneighbors(query_matrix[start:stop], return_distance=True)
        all_distances.append(distances[:, 0].astype(np.float32))
        all_indices.append(indices[:, 0].astype(np.int64))
    return np.concatenate(all_distances), np.concatenate(all_indices)


def quantile_bin_edges(values: Iterable[float], *, quantiles: int = 10) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.array([0.0, 1.0], dtype=float)
    if np.allclose(array, array[0]):
        return np.array([float(array[0]), float(array[0]) + 1e-9], dtype=float)

    raw_edges = np.quantile(array, np.linspace(0.0, 1.0, quantiles + 1))
    edges = np.unique(np.round(raw_edges, 8))
    if len(edges) < 2:
        edges = np.array([float(array.min()), float(array.max()) + 1e-9], dtype=float)
    elif edges[-1] <= edges[0]:
        edges[-1] = edges[0] + 1e-9
    return edges.astype(float)


def bin_labels(edges: Sequence[float]) -> list[str]:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        labels.append(f"[{left:.4f}, {right:.4f}]")
    return labels


def reduced_formula_from_mapping(mapping: Mapping[str, float]) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Composition(mapping).reduced_formula
