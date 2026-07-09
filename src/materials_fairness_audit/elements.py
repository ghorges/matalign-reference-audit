from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from pymatgen.core import Composition, Element


PLUS_U_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}
THREE_D_TRANSITION_METALS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
F_BLOCK = {symbol for symbol in ("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu").split()}
HEAVY_FIVE_D = {
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
}


@dataclass(slots=True)
class FormulaFeatures:
    formula: str
    reduced_formula: str
    elements: tuple[str, ...]
    n_elements: int
    has_3d_transition_metal: bool
    has_4f_lanthanide: bool
    has_5d_heavy_element: bool
    avg_electronegativity: float | None
    std_electronegativity: float | None
    has_plus_u_element: bool


def composition_to_symbols(formula: str) -> tuple[str, ...]:
    composition = Composition(formula)
    return tuple(sorted(str(element) for element in composition.elements))


def reduced_formula(formula: str) -> str:
    return Composition(formula).reduced_formula


def ionic_radius_range(symbol: str) -> float | None:
    element = Element(symbol)
    radii = [float(value) for value in (element.ionic_radii or {}).values() if value is not None]
    if not radii and element.average_ionic_radius is not None:
        radii = [float(element.average_ionic_radius)]
    if not radii:
        return None
    return max(radii) - min(radii)


def parse_formula_features(formula: str) -> FormulaFeatures:
    composition = Composition(formula)
    symbols = tuple(sorted(str(element) for element in composition.elements))
    elements = [Element(symbol) for symbol in symbols]
    electronegativities = [element.X for element in elements if element.X is not None]

    avg_x = mean(electronegativities) if electronegativities else None
    std_x = pstdev(electronegativities) if len(electronegativities) > 1 else 0.0 if electronegativities else None

    return FormulaFeatures(
        formula=formula,
        reduced_formula=composition.reduced_formula,
        elements=symbols,
        n_elements=len(symbols),
        has_3d_transition_metal=any(symbol in THREE_D_TRANSITION_METALS for symbol in symbols),
        has_4f_lanthanide=any(symbol in F_BLOCK for symbol in symbols),
        has_5d_heavy_element=any(symbol in HEAVY_FIVE_D for symbol in symbols),
        avg_electronegativity=avg_x,
        std_electronegativity=std_x,
        has_plus_u_element=any(symbol in PLUS_U_ELEMENTS for symbol in symbols),
    )


def element_property_record(symbol: str) -> dict[str, int | str | bool]:
    element = Element(symbol)
    return {
        "symbol": symbol,
        "atomic_number": element.Z,
        "period": element.row,
        "group": element.group if element.group is not None else -1,
        "is_plus_u": symbol in PLUS_U_ELEMENTS,
        "is_f_block": symbol in F_BLOCK,
        "has_heavy_soc": symbol in HEAVY_FIVE_D,
        "n_common_oxidation_states": len(element.common_oxidation_states or ()),
        "ionic_radius_range": ionic_radius_range(symbol),
    }
