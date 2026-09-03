"""Real-hardware path for producing latent banks.

The paper draws its hardware latents from an ORCA PT-2 (16 photons in 32 time
bins).  On the Quandela stack the equivalent is a QPU reached through Perceval's
``RemoteProcessor`` (or a Scaleway session), which MerLin wraps with
:class:`merlin.MerlinProcessor`.

Because the latent distribution is *fixed* -- no gradient ever flows into the
photonic circuit -- the hardware only ever has to do one thing: emit a large
bank of samples once, before training.  That makes the simulated and the
hardware paths fully interchangeable::

    bank = sample_hardware("qpu:belenos", n_modes=32, n_photons=16,
                           n_samples=500_000, token=os.environ["QUANDELA_TOKEN"])
    latent = SampleBankLatent(bank, name="boson sampler (QPU)")

Everything downstream is unchanged.
"""

from __future__ import annotations

import os
import time

import numpy as np
import perceval as pcvl
from lib.circuits import delay_line_unitary, haar_unitary, to_circuit
from lib.latents import SampleBankLatent

__all__ = ["sample_hardware", "hardware_latent", "list_platforms"]


def list_platforms(token: str | None = None):  # pragma: no cover - needs network
    """Return the Quandela Cloud platforms visible to ``token``."""
    token = token or os.environ.get("QUANDELA_TOKEN", "")
    pcvl.save_token(token)
    return pcvl.RemoteConfig.list_platforms()


def sample_hardware(  # pragma: no cover - needs network
    platform: str,
    n_modes: int,
    n_photons: int,
    n_samples: int,
    *,
    architecture: str = "1-1",
    seed: int = 0,
    token: str | None = None,
    post_select: bool = True,
    max_shots_per_call: int = 100_000,
    poll: float = 5.0,
    max_empty_batches: int = 3,
) -> np.ndarray:
    """Draw a bank of photon-count patterns from a real QPU.

    Parameters
    ----------
    platform : str
        Quandela Cloud platform name, e.g. ``"qpu:ascella"`` or ``"qpu:belenos"``.
    architecture : {"haar", "1-1", "1-3-9"}
        Circuit loaded onto the device. Default value is "1-1", the delay-line
        configuration of the ORCA PT-2 used for the paper's hardware runs.
    token : str | None
        Quandela Cloud token; falls back to ``$QUANDELA_TOKEN``.
    post_select : bool
        Follow the paper: populate every input mode and keep only shots with at
        least ``n_photons`` detections. Default value is True. Set False to
        inject exactly ``n_photons`` and keep lossy shots as well.

        Note the two branches give ``n_photons`` different meanings -- a
        detection threshold when True, an injection count when False. That is
        deliberate; see the Notes below.
    max_empty_batches : int
        Abort after this many consecutive batches in which post-selection
        rejected every shot. Default value is 3.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_samples, n_modes)``.

    Notes
    -----
    Real devices are lossy, so many shots contain fewer photons than were sent
    in. The paper post-selects: *"we also used post-selection where we populated
    all 32 input time bins and discarded all results in which fewer than 16
    photons were measured."* Because lost photons in a linear network behave as
    if they were never present, this *"effectively mimics roughly 16 photons in
    32 channels with randomized input locations"*, and the authors note that
    *"Despite this input randomization, quantum effects such as photon bunching
    still occur and affect the output statistics."*

    Following that recipe means the input state here populates **every** mode
    and the filter requires ``n_photons`` detections, rather than injecting
    ``n_photons`` photons and accepting whatever survives.
    """
    token = token or os.environ.get("QUANDELA_TOKEN", "")
    if not token:
        raise RuntimeError("no Quandela Cloud token (set $QUANDELA_TOKEN)")
    pcvl.save_token(token)

    rng = np.random.default_rng(seed)
    u = (
        haar_unitary(n_modes, rng)
        if architecture == "haar"
        else delay_line_unitary(n_modes, rng, architecture)
    )

    proc = pcvl.RemoteProcessor(platform)
    proc.set_circuit(to_circuit(u))
    if post_select:
        # The paper's recipe, quoted in the docstring: populate EVERY input time
        # bin and keep only the shots in which at least ``n_photons`` were
        # detected. So in this branch ``n_photons`` is a *detection threshold*,
        # not an injection count -- injecting n_photons instead would remove the
        # input randomisation the authors rely on, and on a lossy device would
        # make the acceptance rate collapse, since it would then require every
        # injected photon to survive.
        proc.with_input(pcvl.BasicState([1] * n_modes))
        proc.min_detected_photons_filter(n_photons)
    else:
        # Inject exactly n_photons and keep whatever survives, lossy shots
        # included. Here n_photons *is* the injection count.
        proc.with_input(pcvl.BasicState([1] * n_photons + [0] * (n_modes - n_photons)))
        proc.min_detected_photons_filter(0)

    out = np.zeros((n_samples, n_modes), dtype=np.int16)
    done = 0
    empty_batches = 0
    sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=max_shots_per_call)
    while done < n_samples:
        k = min(max_shots_per_call, n_samples - done)
        job = sampler.samples.execute_async(k)
        while not job.is_complete:
            time.sleep(poll)
        res = job.get_results()["results"]
        take = min(k, len(res))
        if take == 0:
            # Post-selection can reject an entire batch on a lossy device. Give
            # up rather than loop forever asking for shots that never arrive.
            empty_batches += 1
            if empty_batches >= max_empty_batches:
                raise RuntimeError(
                    f"{empty_batches} consecutive empty batches from {platform!r} "
                    f"after {done}/{n_samples} samples; post-selection at "
                    f"n_photons={n_photons} is rejecting every shot"
                )
            continue
        empty_batches = 0
        out[done : done + take] = np.asarray(
            [list(s) for s in res[:take]], dtype=np.int16
        )
        done += take
    return out


def hardware_latent(  # pragma: no cover - needs network
    platform: str,
    n_modes: int,
    n_photons: int,
    n_samples: int,
    **kwargs,
) -> SampleBankLatent:
    """Convenience wrapper returning a ready-to-train latent distribution."""
    bank = sample_hardware(platform, n_modes, n_photons, n_samples, **kwargs)
    return SampleBankLatent(bank, name=f"boson sampler ({platform})")
