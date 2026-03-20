"""
Simple CLI dispatcher for HQPINN experiments.

Usage:
    python -m HQPINN
"""


def _ask_mode() -> str:
    mode = input("Mode? [train/run/remote] ").strip().lower()
    if mode not in {"train", "run", "remote"}:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
    return mode


def _ask_backend(mode: str) -> str:
    if mode == "remote":
        backend = input("Backend? [sim:ascella] ").strip()
        return backend or "sim:ascella"
    return "local"


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
    print("  dee-cp          -> DEE, Classical–PennyLane")
    print("  dee-ii          -> DEE, Interferometer–Interferometer")
    print("  dee-pp          -> DEE, PennyLane–PennyLane")
    print("  taf-cc          -> TAF, Classical–Classical")
    print("  taf-ci          -> TAF, Classical–Interferometer")
    print("  taf-cp          -> TAF, Classical–PennyLane")
    print("  taf-ii          -> TAF, Interferometer–Interferometer")
    print("  taf-pp          -> TAF, PennyLane–PennyLane")

    print()
    choice = input("Which experiment do you want to run? ").strip()

    # DHO experiments
    if choice == "dho-pp":
        from .DHO.a2_dho_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cc":
        from .DHO.a2_dho_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cp":
        from .DHO.a2_dho_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cperc":
        from .DHO.a2_dho_cperc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-ii":
        from .DHO.a2_dho_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-percperc":
        from .DHO.a2_dho_percperc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-ci":
        from .DHO.a2_dho_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    # SEE experiments
    elif choice == "see-cc":
        from .SEE.see_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-cp":
        from .SEE.see_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-ii":
        from .SEE.see_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(
                mode=mode,
                backend=backend,
                n_photons=n_photons,
            )

    elif choice == "see-pp":
        from .SEE.see_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-ci":
        from .SEE.see_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    # DEE experiments
    elif choice == "dee-cc":
        from .DEE.dee_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "dee-ii":
        from .DEE.dee_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(
                mode=mode,
                backend=backend,
                n_photons=n_photons,
            )

    elif choice == "dee-ci":
        from .DEE.dee_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-1/10-7-1/20-4-1] ").strip() or "10-4-1"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    elif choice == "dee-cp":
        from .DEE.dee_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    elif choice == "dee-pp":
        from .DEE.dee_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    # TAF experiments
    elif choice == "taf-cc":
        from .TAF.taf_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [40-4/40-7/80-4] ").strip() or "40-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-ci":
        from .TAF.taf_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-cp":
        from .TAF.taf_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-ii":
        from .TAF.taf_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(mode=mode, backend=backend, n_photons=n_photons)

    elif choice == "taf-pp":
        from .TAF.taf_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/4/6] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    else:
        print(f"Unknown experiment: {choice}")
        print(
            "Please choose one of: dho-pp, dho-cc, dho-cp, dho-cperc, "
            "dho-ii, dho-percperc, dho-ci, see-cc, see-pp, see-ci, see-ii, see-cp, "
            "dee-cc, dee-ci, dee-cp, dee-ii, dee-pp, taf-cc, taf-ci, taf-cp, taf-ii, taf-pp."
        )
