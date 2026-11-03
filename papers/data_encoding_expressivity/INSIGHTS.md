# Insights — data encoding and expressive power (Schuld, Sweke, Meyer 2021)

Reusable lessons for similar reproductions (Fourier-type expressivity, data re-uploading, encoding studies, trainability comparisons).

## Scientific results worth reusing

- **Test theory claims exactly, not by training.** The spectrum, the size bound and the universality construction are identities; FFT support checks on random circuits reach machine precision (stray coefficients below 1e-15) in seconds. Training figures need no unstated hyperparameters for these.
- **Always compute a closed-form floor.** Least-squares fit over the model's exact frequency set (B1) separates "cannot express" from "failed to train". A matched classical model with the same frequencies (B2) separates circuit effects from optimiser effects.
- **Expressible is not reliably trainable.** Fig. 4's r = 5 fit is exactly representable, yet under the paper's protocol (lr 0.3, mini-batch 25, 200 steps) only 25–35% of seeds reach it. A matched classical model is similarly protocol-sensitive.
- **The protocol, not the circuit, is the first suspect.** Full-batch Adam at lr 0.01–0.05, or L-BFGS, makes every model whose class contains the target reach it in nearly all seeds. Report several protocols, labelled, next to the paper's.
- **Report seed counts, not one typical run.** Published single runs can be favourable draws; 6 to 20 seeds change the picture.

## Photonic translation

- The encoding generator maps to the **photon number** on the encoded modes. Degree = photons × encoding layers; encoding more modes adds nothing (all modes encoded is a global phase). Per-mode phase scales give high degree from few photons.
- **Accessible spectrum is not controllable coefficients.** With a fixed single-mode threshold readout, n photons reach degree n but with tiny top coefficients (about 0.002 against 0.07 needed), because frequency n needs all photons bunched in one mode.
- **The observable decides expressivity.** A fixed ±1 observable stalls photonic models near 2e-3 even under L-BFGS; a trainable linear readout over click patterns, plus universal meshes, removes the stall. Give qubit and classical models the same readout class when comparing.
- **Universality is testable photonically in the single-photon setting:** one photon in m modes, phase k·x on mode k, universal mesh, trainable readout. It is classically simulable, so it tests the theorem, not quantum value. Multi-photon models reach only a subgroup of Fock-space unitaries.
- **Multi-photon, multi-scale encodings are parameter-efficient** (degree 9 from 3 photons in 4 modes, about 40 parameters, against 10 modes and 200 parameters for one photon), at slightly lower per-seed reliability.

## MerLin patterns and pitfalls (0.2.3)

- Use `FOCK` and group outcomes into click patterns to emulate threshold detection. `UNBUNCHED` is postselection with renormalisation, which makes outputs non-band-limited.
- One `add_entangling_layer()` is not a universal mesh (m(m-1) of m²-1 parameters). Put a trainable phase screen on modes 1..m-1 before each mesh that follows an encoding; the input mesh needs none.
- Several `add_angle_encoding` calls with non-unit scales are unreliable. Use one call at scale 1 and pre-scale the inputs.
- Initial state: single photons in the first n of n+1 modes; keep one empty mode.

## Baseline fairness

- Match baselines on the exact frequency set for an expressivity claim, not on parameter count.
- Check that initialisation conventions are consistent across compared models (zero-initialised readouts versus uniformly initialised baseline coefficients flipped one comparison at small learning rates). Report both variants.
- Check each claimed hyperparameter against the original figures: the Fig. 4 caption disagreed with the published panel.

## Negative results and caveats

- The paper's step-budget phrase ("at most 200 steps") hides a stopping rule that cannot be recovered.
- Fig. 3's loss axis lies below the floor of the model class; its plotted quantity is unknown.
- Eq. 28's universal Hamiltonian family needs a factor ½ to reach odd frequencies; Appendix A's sign is flipped; Eq. 12 is valid but loose (the distinct-sum bound is tight for a Golomb-ruler spectrum).
- With 6 seeds, report counts and medians; do not claim statistical superiority.

## Debugging heuristics

- When a model "cannot reach" a target with B1 = 0, test coefficient reachability (maximise |c_k|) before blaming the optimiser. If a fixed observable caps |c_k|, the limit is the model class.
- A model stuck at the same value across two different circuits points to the readout or encoding.
- Verify optimiser changes with a bit-identical regression run on a stored configuration.
- Shell variables holding multi-word flags are not word-split in zsh; split them explicitly.

## Future directions

- Noise, loss and finite shots for the photonic models.
- Multi-scale encodings beyond degree 9 and with more photons.
- A held-out-data task testing whether higher degree helps generalisation.
- Re-check the MerLin findings on 0.4.x.
