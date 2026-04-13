# Code/config.py
# ─────────────────────────────────────────────────────────────
# Single source of truth for the active model variant.
# All pipeline scripts import VARIANT from here.
#
# "baseline" → original K and H implementation
# "stable"   → numerically stable K and H (for simulation/pricing)
#
# Changing this one line switches the entire pipeline.
# ─────────────────────────────────────────────────────────────

import sys

VARIANT = "baseline"


def confirm_variant():
    answer = input(f"\nConfig variant is set to: '{VARIANT}' — proceed? [y/n]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)
