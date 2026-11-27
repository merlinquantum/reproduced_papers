"""Display the actual MZI topology of a compiled MerLin interferometer."""

import numpy as np


def draw_mzi_mesh(circuit, occupations):
    """Draw a full mesh overview and return its first MZI for detailed rendering.

    Parameters
    ----------
    circuit : perceval.Circuit
        Compiled BS/PS/BS/PS mesh; non-optical layout barriers are omitted.
    occupations : list[int]
        Input photon occupations in mode order.

    Returns
    -------
    tuple
        Matplotlib figure, actual first two-mode MZI circuit, and topology counts.
        Stages preserve component order on shared modes; disjoint MZIs can share
        a stage. This is a topology overview, not a physical timing diagram.
    """
    import matplotlib.pyplot as plt
    import perceval as pc
    from matplotlib.patches import Patch, Rectangle

    values = np.asarray(occupations)
    if (
        values.shape != (circuit.m,)
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values < 0)
    ):
        raise ValueError("Input occupations must be nonnegative integers in mode order")
    components = [
        (modes, gate) for modes, gate in circuit if not isinstance(gate, pc.Barrier)
    ]
    blocks, index = [], 0
    while index < len(components) and isinstance(components[index][1], pc.BS):
        block = components[index : index + 4]
        if len(block) != 4 or not all(
            isinstance(item[1], kind)
            for item, kind in zip(block, [pc.BS, pc.PS, pc.BS, pc.PS])
        ):
            raise ValueError("Expected BS/PS/BS/PS MZI blocks")
        modes = block[0][0]
        if len(modes) != 2 or modes != block[2][0] or modes[1] != modes[0] + 1:
            raise ValueError("Expected adjacent two-mode MZIs")
        if any(not set(item[0]).issubset(modes) for item in block):
            raise ValueError("MZI phase shifters are outside its mode pair")
        blocks.append(block)
        index += 4
    terminal = components[index:]
    if not blocks or any(
        not isinstance(gate, pc.PS) or len(modes) != 1 for modes, gate in terminal
    ):
        raise ValueError("Unexpected components after the MZI mesh")

    last_stage = np.full(circuit.m, -1, dtype=int)
    layout = []
    for block in blocks:
        a, b = block[0][0]
        column = int(max(last_stage[a], last_stage[b]) + 1)
        last_stage[[a, b]] = column
        layout.append((column, a, b))
    stages = int(last_stage.max() + 1)
    output_x = stages + 0.2
    fig, ax = plt.subplots(figsize=(13, 7.3), layout="constrained")
    for mode, photons in enumerate(values):
        ax.plot(
            [-0.75, output_x + 0.7], [mode, mode], color="#a4adb5", lw=0.65, zorder=0
        )
        ax.scatter(
            [-1.05],
            [mode],
            s=30,
            color="#167c91" if photons else "white",
            edgecolors="#167c91" if photons else "#a4adb5",
            linewidths=0.8,
        )
        ax.text(-1.65, mode, str(int(photons)), ha="center", va="center", fontsize=8)
    for column, a, b in layout:
        ax.add_patch(
            Rectangle(
                (column - 0.32, a - 0.16),
                0.64,
                b - a + 0.32,
                facecolor="#cee7eb",
                edgecolor="#167c91",
                linewidth=0.55,
            )
        )
    for modes, _ in terminal:
        ax.scatter([output_x], [modes[0]], marker="s", s=23, color="#df9d45", zorder=3)
    ax.set(
        xlim=(-2.1, output_x + 1),
        ylim=(circuit.m - 0.4, -0.85),
        xlabel="MZI coupling stage (shared-mode order preserved)",
        ylabel="Optical mode",
        title=f"Actual fixed MerLin mesh: {circuit.m} modes, {int(values.sum())} photons, {len(blocks)} MZIs",
    )
    ax.set_yticks(range(circuit.m))
    ticks = list(range(0, stages, 4))
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.text(-1.65, -0.7, "Input", ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    legend = [
        Patch(
            facecolor="#cee7eb",
            edgecolor="#167c91",
            label="One MZI (two beam splitters + two phase shifters)",
        )
    ]
    if terminal:
        legend.append(Patch(facecolor="#df9d45", label="Output phase shifter"))
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=9,
    )

    first_mzi = pc.Circuit(2, name="First MZI")
    offset = min(blocks[0][0][0])
    for modes, gate in blocks[0]:
        first_mzi.add(tuple(mode - offset for mode in modes), gate)
    summary = {
        "modes": circuit.m,
        "mzis": len(blocks),
        "coupling_stages": stages,
        "output_phase_shifters": len(terminal),
        "input_photons": int(values.sum()),
    }
    return fig, first_mzi, summary
