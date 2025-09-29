# Compile the design token source of truth and document the exported variables.

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "app" / "static"
DOCS_DIR = REPO_ROOT / "docs"

TOKEN_SOURCE = STATIC_DIR / "design_tokens.scss"
CSS_OUTPUT = STATIC_DIR / "theme.css"
DOC_OUTPUT = DOCS_DIR / "design-system.md"

TOKEN_SECTION_METADATA: Dict[str, Tuple[str, str]] = {
    "color-primary": ("Escala primaria", "Azules base usados para capas principales."),
    "color-neon": ("Escala neón", "Acentos brillantes para estados y CTA."),
    "color-mist": ("Escala neblina", "Neutros fríos para fondos y bordes."),
    "font-size": ("Tipografía fluida", "Clamps responsivos por jerarquía."),
    "space": ("Ritmo de espaciado", "Escalas en rem para layout."),
    "shadow": ("Sombras multicapa", "Capas apiladas para profundidad."),
    "state": ("Tokens de estado", "Transformaciones para hover/press/focus."),
    "surface": ("Superficies", "Valores base para superficies oscuras."),
    "glass": ("Glassmorphism", "Tokens para tarjetas translúcidas."),
    "app": ("Lienzo", "Fondo base de la aplicación."),
}

TOKEN_NAME_RE = re.compile(r"--([a-z0-9-]+):\s*([^;]+);")


def _load_sass() -> "module":  # type: ignore[override]
    try:
        import sass  # type: ignore
    except ImportError as exc:  # pragma: no cover - guidance for users
        raise SystemExit(
            "El paquete 'sass' no está instalado. Ejecuta 'pip install --upgrade sass'"
        ) from exc
    return sass


def compile_theme() -> str:
    sass = _load_sass()
    css = sass.compile(filename=str(TOKEN_SOURCE), output_style="expanded")
    CSS_OUTPUT.write_text(css, encoding="utf-8")
    return css


def _extract_tokens(css: str) -> Dict[str, List[Tuple[str, str]]]:
    """Return an ordered mapping of token prefix to pairs of name/value."""

    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for name, value in TOKEN_NAME_RE.findall(css):
        prefix = name.split("-")[0]
        if name.startswith("color-"):
            prefix = "-".join(name.split("-")[:2])
        elif name.startswith("font-size-"):
            prefix = "font-size"
        elif name.startswith("space-"):
            prefix = "space"
        elif name.startswith("shadow-"):
            prefix = "shadow"
        elif name.startswith("state-"):
            prefix = "state"
        elif name.startswith("surface-"):
            prefix = "surface"
        elif name.startswith("glass-"):
            prefix = "glass"
        elif name.startswith("app-"):
            prefix = "app"

        buckets.setdefault(prefix, []).append((name, value.strip()))
    return buckets


def _format_table(rows: Iterable[Tuple[str, str]]) -> str:
    body = ["| Token | Valor |", "| --- | --- |"]
    for token_name, value in rows:
        body.append(f"| `--{token_name}` | `{value}` |")
    return "\n".join(body)


def write_docs(css: str) -> None:
    buckets = _extract_tokens(css)

    sections: List[str] = [
        "# Sistema de diseño",
        "",
        "Este documento se genera desde `scripts/build_theme.py` y describe los tokens disponibles",
        "tras compilar `app/static/design_tokens.scss`. Actualiza el SCSS y vuelve a ejecutar el",
        "script para refrescar las tablas.",
        "",
        "## Tokens",
    ]

    for prefix, rows in sorted(buckets.items()):
        title, description = TOKEN_SECTION_METADATA.get(prefix, (prefix.title(), ""))
        sections.append(f"### {title}")
        if description:
            sections.append(description)
        sections.append("")
        sections.append(_format_table(rows))
        sections.append("")

    sections.extend(
        [
            "## Uso",
            "",
            "```bash",
            "python scripts/build_theme.py",
            "```",
            "",
            "El comando anterior recompila el CSS y regenera esta documentación.",
        ]
    )

    DOC_OUTPUT.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="No recompila el CSS, solo reconstruye la documentación a partir del archivo actual.",
    )
    args = parser.parse_args(argv)

    if args.docs_only:
        css = CSS_OUTPUT.read_text(encoding="utf-8")
    else:
        css = compile_theme()

    write_docs(css)
    return 0


if __name__ == "__main__":
    sys.exit(main())
