"""Mutant evaluation harness and mutation-score computation.

Implements Section IV-B of the paper: the original model predicts the sampled
test data, the correctly predicted samples become the test suite, and a mutant
is *killed* when it no longer predicts the reference label on at least one
suite sample. Mutants that cannot be executed at all are *incompetent* and are
excluded from the mutation score, as the paper prescribes for LS.

Two verdicts are produced for every mutant, each under its own internally
consistent protocol:

- **shot protocol** (the paper's): the suite is the set of candidate samples the
  original model classifies correctly with a 1,024-shot majority vote, and a
  mutant is killed when its 1,024-shot prediction differs from the true label
  on at least one of them;
- **analytic protocol**: the suite is the set of candidates the original model
  classifies correctly from the exact output distribution, and the verdict uses
  exact distributions throughout.

The protocols must not be mixed. A sample can be shot-correct but
analytically incorrect, so judging an analytic prediction against a
shot-selected suite lets sampling noise "kill" an unmutated circuit. Keeping
them separate makes the analytic false-kill rate exactly zero by construction,
which the null control verifies.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field

import torch
from lib.mutations import Mutant, apply_feature_transform
from lib.photonic_qnn import PhotonicQnnClassifier, PhotonicQnnSpec

logger = logging.getLogger(__name__)

STATUS_KILLED = "killed"
STATUS_SURVIVED = "survived"
STATUS_INCOMPETENT = "incompetent"


@dataclass
class MutantOutcome:
    """Result of evaluating one mutant.

    Attributes
    ----------
    mutant_id, operator, description : str
        Identity of the mutant.
    status : str
        ``killed``, ``survived``, or ``incompetent`` (shot-budget verdict).
    status_analytic : str
        Same verdict computed from the exact distribution.
    n_changed : int
        Number of suite samples whose predicted label changed (shot protocol).
    n_suite : int
        Size of the shot-protocol test suite.
    n_changed_analytic, n_suite_analytic : int
        The same two quantities under the analytic protocol.
    predictions : list[int]
        Predicted labels on the suite, used for behavioural-redundancy
        analysis.
    circuit_signature : str | None
        Hash of the mutant circuit, used for structural-redundancy analysis.
    generation_time_s, evaluation_time_s : float
        Wall-clock cost of building and evaluating the mutant.
    error : str | None
        Exception text for incompetent mutants.
    """

    mutant_id: str
    operator: str
    description: str
    status: str
    status_analytic: str
    n_changed: int
    n_suite: int
    n_changed_analytic: int = 0
    n_suite_analytic: int = 0
    predictions: list[int] = field(default_factory=list)
    circuit_signature: str | None = None
    generation_time_s: float = 0.0
    evaluation_time_s: float = 0.0
    error: str | None = None


def _signature_hash(mutant: Mutant) -> str:
    """Return a short unique hash of the mutant's circuit structure."""
    if mutant.ir is None:
        payload = repr(mutant.feature_transform)
    elif isinstance(mutant.ir, dict):
        signatures = {
            layer_idx: layer_ir.signature() for layer_idx, layer_ir in mutant.ir.items()
        }
        payload = repr(sorted(signatures.items()))
    else:
        payload = repr(mutant.ir.signature())

    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _deterministic_seed(mutant_id: str) -> int:
    """Return a stable per-mutant seed so that sampled verdicts reproduce."""

    digest = hashlib.sha256(mutant_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class SuiteSpec:
    """The two test suites of a trained original model.

    Attributes
    ----------
    x_candidates : torch.Tensor
        All candidate samples (20 in the paper).
    labels : torch.Tensor
        Their true labels.
    shot_positions : list[int]
        Candidates the original model classifies correctly with the shot
        budget; the paper's suite.
    analytic_positions : list[int]
        Candidates it classifies correctly from the exact distribution.
    """

    x_candidates: torch.Tensor
    labels: torch.Tensor
    shot_positions: list[int]
    analytic_positions: list[int]

    @classmethod
    def from_suite(cls, x_candidates, labels, suite: dict) -> SuiteSpec:
        """Build from the mapping returned by ``build_mutation_test_suite``."""

        return cls(
            x_candidates=x_candidates,
            labels=labels,
            shot_positions=list(suite["suite_positions"]),
            analytic_positions=list(suite["suite_positions_analytic"]),
        )

    def _verdict(self, predictions: torch.Tensor, positions: list[int]) -> int:
        """Return how many suite samples changed label."""

        if not positions:
            return 0
        index = torch.tensor(positions, dtype=torch.long)
        return int((predictions[index] != self.labels[index]).sum().item())


def evaluate_mutant(
    mutant: Mutant,
    spec: PhotonicQnnSpec,
    original_model: PhotonicQnnClassifier,
    suite: SuiteSpec,
    shots: int,
) -> MutantOutcome:
    """Evaluate one mutant under both protocols.

    Parameters
    ----------
    mutant : Mutant
        The mutant to evaluate.
    spec : PhotonicQnnSpec
        Spec used to realise circuit mutants.
    original_model : PhotonicQnnClassifier
        Original model, reused for data mutations (DFC).
    suite : SuiteSpec
        Candidate samples and the two suites.
    shots : int
        Shot budget of the paper's protocol.

    Returns
    -------
    MutantOutcome
        Verdicts and diagnostics. Construction or execution failures produce an
        ``incompetent`` outcome rather than propagating.
    """

    outcome = MutantOutcome(
        mutant_id=mutant.mutant_id,
        operator=mutant.operator,
        description=mutant.description,
        status=STATUS_INCOMPETENT,
        status_analytic=STATUS_INCOMPETENT,
        n_changed=0,
        n_suite=len(suite.shot_positions),
        n_suite_analytic=len(suite.analytic_positions),
        circuit_signature=_signature_hash(mutant),
    )

    build_started = time.perf_counter()
    try:
        if mutant.ir is not None:
            model = PhotonicQnnClassifier(spec, mutant.ir)
            inputs = suite.x_candidates
        else:
            model = original_model
            inputs = apply_feature_transform(
                suite.x_candidates, mutant.feature_transform
            )
        outcome.generation_time_s = time.perf_counter() - build_started
    except Exception as exc:  # noqa: BLE001 - an unbuildable mutant is incompetent
        outcome.generation_time_s = time.perf_counter() - build_started
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    eval_started = time.perf_counter()
    try:
        model.eval()
        torch.manual_seed(_deterministic_seed(mutant.mutant_id))
        with torch.no_grad():
            sampled = model(inputs, shots=shots).argmax(dim=1)
            analytic = model(inputs).argmax(dim=1)
    except Exception as exc:  # noqa: BLE001 - an unrunnable mutant is incompetent
        outcome.evaluation_time_s = time.perf_counter() - eval_started
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.evaluation_time_s = time.perf_counter() - eval_started
    index = torch.tensor(suite.shot_positions, dtype=torch.long)
    outcome.predictions = sampled[index].tolist() if len(suite.shot_positions) else []
    outcome.n_changed = suite._verdict(sampled, suite.shot_positions)
    outcome.n_changed_analytic = suite._verdict(analytic, suite.analytic_positions)
    outcome.status = STATUS_KILLED if outcome.n_changed > 0 else STATUS_SURVIVED
    outcome.status_analytic = (
        STATUS_KILLED if outcome.n_changed_analytic > 0 else STATUS_SURVIVED
    )
    return outcome


def evaluate_mutants(
    mutants: list[Mutant],
    spec: PhotonicQnnSpec,
    original_model: PhotonicQnnClassifier,
    suite: SuiteSpec,
    shots: int,
    *,
    family: str,
    log_every: int = 100,
) -> list[MutantOutcome]:
    """Evaluate a whole mutant family, logging progress.

    Parameters
    ----------
    mutants : list[Mutant]
        Mutants to evaluate.
    spec : PhotonicQnnSpec
        Spec used to realise circuit mutants.
    original_model : PhotonicQnnClassifier
        Original model.
    suite : SuiteSpec
        Candidate samples and the two suites.
    shots : int
        Shot budget.
    family : str
        ``"new"`` or ``"baseline"``, used in the log records.
    log_every : int
        Progress log interval. Default value is 100.

    Returns
    -------
    list[MutantOutcome]
        One outcome per mutant, in generation order.
    """

    logger.info(
        "MUTANT_EVALUATION_STARTED | family=%s | mutants=%d | suite_shots=%d "
        "| suite_analytic=%d | shots=%d",
        family,
        len(mutants),
        len(suite.shot_positions),
        len(suite.analytic_positions),
        shots,
    )
    outcomes: list[MutantOutcome] = []
    started = time.perf_counter()
    for index, mutant in enumerate(mutants, start=1):
        outcomes.append(evaluate_mutant(mutant, spec, original_model, suite, shots))
        if index % log_every == 0 or index == len(mutants):
            killed = sum(1 for o in outcomes if o.status == STATUS_KILLED)
            logger.info(
                "MUTANT_PROGRESS | family=%s | evaluated=%d/%d | killed=%d "
                "| elapsed_s=%.1f",
                family,
                index,
                len(mutants),
                killed,
                time.perf_counter() - started,
            )
    logger.info(
        "MUTANT_EVALUATION_COMPLETED | family=%s | mutants=%d | elapsed_s=%.1f",
        family,
        len(mutants),
        time.perf_counter() - started,
    )
    return outcomes


def summarise(
    outcomes: list[MutantOutcome],
    *,
    family: str,
    suppressed: dict[str, int] | None = None,
) -> dict:
    """Aggregate outcomes into per-operator and overall mutation scores.

    Mutation score is ``killed / (killed + survived)``; incompetent mutants are
    excluded from the denominator, as the paper prescribes.

    Parameters
    ----------
    outcomes : list[MutantOutcome]
        Evaluated mutants.
    family : str
        Family label recorded in the summary.
    suppressed : dict[str, int] | None
        Operator -> number of provably redundant variants the directed scheme
        declined to generate. Default value is None.

    Returns
    -------
    dict
        Structured summary suitable for a results table.
    """

    operators: dict[str, dict] = {}
    for outcome in outcomes:
        row = operators.setdefault(
            outcome.operator,
            {
                "generated": 0,
                "killed": 0,
                "survived": 0,
                "incompetent": 0,
                "killed_analytic": 0,
                "distinct_circuits": set(),
                "distinct_behaviours": set(),
                "generation_time_s": 0.0,
                "evaluation_time_s": 0.0,
            },
        )
        row["generated"] += 1
        row["generation_time_s"] += outcome.generation_time_s
        row["evaluation_time_s"] += outcome.evaluation_time_s
        if outcome.status == STATUS_KILLED:
            row["killed"] += 1
        elif outcome.status == STATUS_SURVIVED:
            row["survived"] += 1
        else:
            row["incompetent"] += 1
        if outcome.status_analytic == STATUS_KILLED:
            row["killed_analytic"] += 1
        if outcome.circuit_signature is not None:
            row["distinct_circuits"].add(outcome.circuit_signature)
        if outcome.predictions:
            row["distinct_behaviours"].add(tuple(outcome.predictions))

    summary_rows = {}
    for operator, row in operators.items():
        evaluated = row["killed"] + row["survived"]
        summary_rows[operator] = {
            "generated": row["generated"],
            "killed": row["killed"],
            "survived": row["survived"],
            "incompetent": row["incompetent"],
            "mutation_score": row["killed"] / evaluated if evaluated else None,
            "mutation_score_analytic": (
                row["killed_analytic"] / evaluated if evaluated else None
            ),
            "distinct_circuits": len(row["distinct_circuits"]),
            "distinct_behaviours": len(row["distinct_behaviours"]),
            "suppressed_redundant": (suppressed or {}).get(operator, 0),
            "generation_time_s": row["generation_time_s"],
            "evaluation_time_s": row["evaluation_time_s"],
            "generation_time_per_mutant_s": (
                row["generation_time_s"] / row["generated"]
                if row["generated"]
                else None
            ),
            "evaluation_time_per_mutant_s": (
                row["evaluation_time_s"] / row["generated"]
                if row["generated"]
                else None
            ),
        }

    total_killed = sum(row["killed"] for row in summary_rows.values())
    total_survived = sum(row["survived"] for row in summary_rows.values())
    total_incompetent = sum(row["incompetent"] for row in summary_rows.values())
    total_generated = sum(row["generated"] for row in summary_rows.values())
    total_evaluated = total_killed + total_survived
    all_behaviours = {
        tuple(outcome.predictions) for outcome in outcomes if outcome.predictions
    }

    summary = {
        "family": family,
        "per_operator": summary_rows,
        "total_generated": total_generated,
        "total_killed": total_killed,
        "total_survived": total_survived,
        "total_incompetent": total_incompetent,
        "mutation_score": total_killed / total_evaluated if total_evaluated else None,
        "mutation_score_analytic": (
            sum(1 for o in outcomes if o.status_analytic == STATUS_KILLED)
            / total_evaluated
            if total_evaluated
            else None
        ),
        "distinct_behaviours_total": len(all_behaviours),
        "behavioural_redundancy_rate": (
            1.0 - len(all_behaviours) / total_evaluated if total_evaluated else None
        ),
        "suppressed_redundant_total": sum((suppressed or {}).values()),
        "generation_time_total_s": sum(o.generation_time_s for o in outcomes),
        "evaluation_time_total_s": sum(o.evaluation_time_s for o in outcomes),
    }
    logger.info(
        "MUTATION_SCORE | family=%s | MS=%.4f | killed=%d | survived=%d "
        "| incompetent=%d | distinct_behaviours=%d",
        family,
        summary["mutation_score"]
        if summary["mutation_score"] is not None
        else float("nan"),
        total_killed,
        total_survived,
        total_incompetent,
        len(all_behaviours),
    )
    return summary


def outcomes_to_records(outcomes: list[MutantOutcome]) -> list[dict]:
    """Return JSON-serialisable outcome records without the prediction vectors."""

    records = []
    for outcome in outcomes:
        record = asdict(outcome)
        record.pop("predictions", None)
        records.append(record)
    return records
