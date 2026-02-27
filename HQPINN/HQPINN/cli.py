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
    print("  dee-cc          -> DEE, Classical–Classical")
    print("  dee-ci          -> DEE, Classical–Interferometer")
    print("  dee-ii          -> DEE, Interferometer–Interferometer")

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

        backend = input("Backend? [sim:ascella] ").strip()
        run(mode=mode, backend=backend)

    elif choice == "dho-percperc":
        # Perceval–Perceval
        from .DHO.a2_dho_percperc import run

        run()

    elif choice == "dho-ci":
        # Classical-Interferometer
        from .DHO.a2_dho_ci import run

        run()

    # ==========================
    # SEE experiments
    # ==========================
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

        backend = "Local"
        n_photons = 2

        mode = input("Train or run? [train/run/remote] ").strip().lower()

        if mode == "remote":
            backend = input("Backend? [sim:ascella] ").strip()
            n_photons = int(input("Number of photons? [1/../6] "))
        if mode == "run":
            n_photons = int(input("Number of photons? [1/../6] "))
        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )

    elif choice == "see-pp":
        # PennyLane–PennyLane SEE
        from .SEE.see_pp import run

        run()

    elif choice == "see-ci":
        # Classical–Interferometer SEE
        from .SEE.see_ci import run

        backend = "Local"
        model_size = "10-4-2"

        mode = input("Train or run? [train/run/remote] ").strip().lower()

        if mode == "remote":
            backend = input("Backend? [sim:ascella] ").strip()
            model_size = input("Model size? [10-4-2/10-7-2/20-4-2] ").strip()
        if mode == "run":
            model_size = input("Model size? [10-4-2/10-7-2/20-4-2] ").strip()

        run(
            mode=mode,
            backend=backend,
            model_size=model_size,
        )

    # ==========================
    # DEE experiments
    # ==========================
    elif choice == "dee-cc":
        # Classical–Classical DEE
        from .DEE.dee_cc import run

        run()

    elif choice == "dee-ii":
        # Interferometer–Interferometer DEE
        from .DEE.dee_ii import run

        backend = "Local"
        n_photons = 2

        mode = input("Train or run? [train/run/remote] ").strip().lower()

        if mode == "remote":
            backend = input("Backend? [sim:ascella] ").strip()
            n_photons = int(input("Number of photons? [1/../6] "))
        if mode == "run":
            n_photons = int(input("Number of photons? [1/../6] "))
        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )

    elif choice == "dee-ci":
        # Classical–Interferometer DEE
        from .DEE.dee_ci import run

        backend = "Local"
        model_size = "10-4-1"

        mode = input("Train or run? [train/run/remote] ").strip().lower()

        if mode == "remote":
            backend = input("Backend? [sim:ascella] ").strip()
            model_size = input("Model size? [10-4-1/10-7-1/20-4-1] ").strip()
        if mode == "run":
            model_size = input("Model size? [10-4-1/10-7-1/20-4-1] ").strip()

        run(
            mode=mode,
            backend=backend,
            model_size=model_size,
        )

    else:
        print(f"Unknown experiment: {choice}")
        print(
            "Please choose one of: dho-pp, dho-cc, dho-cp, dho-cperc, "
            "dho-ii, dho-percperc, dho-ci, see-cc, see-pp, see-ci, see-ii, see-cp, dee-cc, dee-ci, dee-ii."
        )
