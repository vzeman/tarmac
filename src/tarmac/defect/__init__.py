from __future__ import annotations

DEFECT_LABELS = ["crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"]
NONE_LABEL = "none"
SEED = 42

CONCRETE_SPECIFIC_DEFECT_LABELS = ["spalling", "efflorescence", "exposed_rebar", "corrosion"]
CONCRETE_STRUCTURAL_SURFACE_TYPES = {"concrete"}
NON_CONCRETE_DEFECT_SURFACE_TYPES = {"asphalt", "unpaved", "paving_stones", "sett", "gravel", "mud"}
STRUCTURAL_DEFECT_DOMAINS = {"bridge", "building"}

# Label applicability used by analyze and assess. CODEBRIM-backed non-crack labels
# are only considered on concrete/structural imagery; crack remains pavement-wide.
DEFECT_LABEL_APPLICABILITY = {
    "crack": "all",
    "spalling": "concrete_structural",
    "efflorescence": "concrete_structural",
    "exposed_rebar": "concrete_structural",
    "corrosion": "concrete_structural",
}


def infer_defect_domain(
    source_path: str | None = None,
    filename: str | None = None,
    surface_type: str | None = None,
) -> str:
    source = f"{source_path or ''}/{filename or ''}".lower()
    if "codebrim" in source or "bridge" in source:
        return "bridge"
    if "building" in source or "wall" in source:
        return "building"
    if "sdnet2018" in source:
        if "/d" in source or "deck" in source:
            return "bridge"
        if "/w" in source or "wall" in source:
            return "building"
        if "/p" in source or "pavement" in source:
            return "pavement"
    if "runway" in source or "crackairport" in source or "airport" in source:
        return "runway"
    if _normalise_surface_type(surface_type) in {"asphalt", "concrete", "paving_stones", "sett", "unpaved"}:
        return "pavement"
    return "unknown"


def is_concrete_structural_context(surface_type: str | None, domain: str | None = None) -> bool:
    surface = _normalise_surface_type(surface_type)
    if surface in NON_CONCRETE_DEFECT_SURFACE_TYPES:
        return False
    return surface in CONCRETE_STRUCTURAL_SURFACE_TYPES or str(domain or "").lower() in STRUCTURAL_DEFECT_DOMAINS


def is_defect_label_applicable(
    label: str,
    surface_type: str | None,
    domain: str | None = None,
    source_path: str | None = None,
    filename: str | None = None,
) -> bool:
    rule = DEFECT_LABEL_APPLICABILITY.get(label, "all")
    if rule == "all":
        return True
    resolved_domain = domain or infer_defect_domain(
        source_path=source_path,
        filename=filename,
        surface_type=surface_type,
    )
    if rule == "concrete_structural":
        return is_concrete_structural_context(surface_type=surface_type, domain=resolved_domain)
    return False


def _normalise_surface_type(surface_type: str | None) -> str:
    return str(surface_type or "unknown").strip().lower()
