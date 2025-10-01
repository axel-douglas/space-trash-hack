import pandas as pd
import pandas.testing as pdt

from app.modules.problematic import problematic_mask


def _legacy_problematic_row(row: pd.Series) -> bool:
    cat = str(row.get("category", "")).lower()
    fam = str(row.get("material_family", "")).lower()
    flg = str(row.get("flags", "")).lower()
    rules = [
        ("pouches" in cat) or ("multilayer" in flg) or ("pe-pet-al" in fam),
        ("foam" in cat) or ("zotek" in fam) or ("closed_cell" in flg),
        ("eva" in cat)
        or ("ctb" in flg)
        or ("nomex" in fam)
        or ("nylon" in fam)
        or ("polyester" in fam),
        ("glove" in cat) or ("nitrile" in fam),
        ("wipe" in flg) or ("textile" in cat),
    ]
    return any(rules)


def test_problematic_mask_matches_legacy_logic():
    df = pd.DataFrame(
        [
            {"category": "Thermal Pouches", "material_family": "PE-PET-Al", "flags": None},
            {"category": "Foam block", "material_family": "ZOTEK F30", "flags": "closed_cell"},
            {"category": "Glove", "material_family": "Nitrile", "flags": ""},
            {"category": "Utility", "material_family": "Nomex", "flags": "wipe"},
            {"category": "Cargo", "material_family": "Polymer", "flags": "multilayer"},
            {"category": "Random", "material_family": "Steel", "flags": ""},
            {"category": None, "material_family": None, "flags": None},
        ]
    )

    expected = df.apply(_legacy_problematic_row, axis=1)
    vectorized = problematic_mask(df)

    pdt.assert_series_equal(vectorized, expected.astype(bool), check_names=False)

