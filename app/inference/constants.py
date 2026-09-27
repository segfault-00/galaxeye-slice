"""
Single source of truth for the class taxonomy.

Both ``app.inference.engine`` and ``scripts/build_model.py`` import from here
so the index ↔ label contract is defined exactly once.
"""

CLASSES: list[str] = [
    "Forest",
    "River",
    "Residential",
    "Industrial",
    "AnnualCrop",
    "SeaLake",
    "Highway",
]
