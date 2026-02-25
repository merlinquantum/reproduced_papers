"""
Simple CLI dispatcher for HQPINN experiments.

Usage:
    python -m HQPINN
"""


def main() -> None:
    """Entry point for the HQPINN command-line interface."""
    print("Available experiments:")
    print("  dho-cc          -> DHO, Classical–Classical)")
    print("  dho-ci          -> DHO, Classical-Interferometer")
    print("  dho-cp          -> DHO, Classical–PennyLane")
    print("  dho-cperc       -> DHO, Classical–Perceval")
    print("  dho-ii          -> DHO, Interferometer–Interferometer")
    print("  dho-percperc    -> DHO, Perceval–Perceval")
    print("  dho-pp          -> DHO, PennyLane–PennyLane")
    print("  see-cc          -> SEE, Classical–Classical")
    print("  see-ci          -> SEE, Classical–Interferometer")
    print("  see-cp          -> SEE, Classical–PennyLane")
    print("  see-ii          -> SEE, Interferometer–Interferometer")
    print("  see-pp          -> SEE, PennyLane–PennyLane")

    print()
    choice = input("Which experiment do you want to run? ").strip()

    # DHO experiments
    if choice == "dho-pp":
        # PennyLane–PennyLane
        from .DHO.a2_dho_pp import run

        run()

    elif choice == "dho-cc":
        # Classical–Classical
        from .DHO.a2_dho_cc import run

        run()

    elif choice == "dho-cp":
        # Classical–PennyLane
        from .DHO.a2_dho_cp import run

        run()

    elif choice == "dho-cperc":
        # Classical–Perceval
        from .DHO.a2_dho_cperc import run

        run()

    elif choice == "dho-ii":
        from .DHO.a2_dho_ii import run

        mode = input("Train or run? [train/run/remote] ").strip().lower()
        backend = input("Backend? [sim:ascella/qpu:belenos] ").strip().lower()
        run(mode=mode, backend=backend)

    elif choice == "dho-percperc":
        # Perceval–Perceval
        from .DHO.a2_dho_percperc import run

        run()

    elif choice == "dho-ci":
        # Classical-Interferometer
        from .DHO.a2_dho_ci import run

        run()

    # SEE experiments
    elif choice == "see-cc":
        # Classical–Classical SEE
        from .SEE.see_cc import run

        run()

    elif choice == "see-cp":
        # Classical–PennyLane SEE
        from .SEE.see_cp import run

        run()

    elif choice == "see-ii":
        # Interferometer–Interferometer SEE
        from .SEE.see_ii import run

        run()

    elif choice == "see-pp":
        # PennyLane–PennyLane SEE
        from .SEE.see_pp import run

        run()

    elif choice == "see-ci":
        # Classical–Interferometer SEE
        from .SEE.see_ci import run

        run()

    else:
        print(f"Unknown experiment: {choice}")
        print(
            "Please choose one of: dho-pp, dho-cc, dho-cp, dho-cperc, "
            "dho-ii, dho-perc-perc, dho-ci, see-cc, see-pp, see-ci, see-ii, see-cp."
        )
