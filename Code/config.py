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

import os
import sys

VARIANT = "stable"


def confirm_variant():
    # When called from run_all_dims.py (non-interactive), skip the prompt.
    if os.environ.get("SKIP_VARIANT_CONFIRM") == "1":
        print(f"[run_all_dims] Variant confirmed automatically: '{VARIANT}'")
        return
    answer = input(f"\nConfig variant is set to: '{VARIANT}' — proceed? [y/n]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)
