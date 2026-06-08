"""
Single source of truth for the active model variant.
All pipeline scripts import VARIANT from here.

    "baseline" — original K, H and R implementation
    "stable"   — numerically stable K, H and R (for simulation/pricing)
"""

import os
import sys

# Can be overridden by the MODEL_VARIANT environment variable.
VARIANT = os.environ.get("MODEL_VARIANT", "stable")


def confirm_variant():
    """Prompt the user to confirm the active variant, unless running in non-interactive mode."""
    # In non-interactive mode, skip the prompt.
    if os.environ.get("SKIP_VARIANT_CONFIRM") == "1":
        print(f"[run_all_dims] Variant confirmed automatically: '{VARIANT}'")
        return
    answer = input(f"\nConfig variant is set to: '{VARIANT}' — proceed? [y/n]: ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)
