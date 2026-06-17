"""Pool scoring — turn model probabilities into competition picks and points.

Your pool: predict the result (winner or draw) AND the exact score.
Points (NOT additive — the exact total supersedes the result point):
    RESULT_PTS = 2   # correct result (winner/draw), wrong scoreline
    EXACT_PTS  = 3   # exact scoreline — the TOTAL for that game (not 2+3)
A correct result earns 2; nailing the exact score earns 3 in total, not on top.

The model gives two ingredients:
    * a scoreline-probability matrix  P(i, j)   — from Dixon–Coles
    * outcome probabilities           P(result) — from the DC+Elo ensemble (stronger)

`recommend_pick` chooses the single scoreline that MAXIMISES your expected points
under the scoring rules — which is not always the most likely score. (Picking a
heavy favourite's modal 1-0 can beat a coin-flip exact score even when a draw is
individually likelier, because the result point tips it.)
"""
from __future__ import annotations

import numpy as np

RESULT_PTS = 2   # pool: correct result (winner/draw), wrong scoreline
EXACT_PTS = 3    # pool: exact scoreline — TOTAL for the game (supersedes, not additive)


def _result(i: int, j: int) -> str:
    return "home" if i > j else "away" if i < j else "draw"


def top_scorelines(matrix: np.ndarray, n: int = 4):
    """The n most probable exact scores as [( (i,j), prob ), ...]."""
    idx = np.dstack(np.unravel_index(np.argsort(matrix.ravel())[::-1], matrix.shape))[0]
    return [((int(i), int(j)), float(matrix[i, j])) for i, j in idx[:n]]


def recommend_pick(matrix: np.ndarray, outcome_probs: dict,
                   result_pts: float = RESULT_PTS, exact_pts: float = EXACT_PTS):
    """Return the expected-points-optimal scoreline pick.

    Scoring is non-additive (exact total supersedes the result point), so the
    expected points for submitting score (i, j) are:
        result_pts * P(result implied by i,j)  +  (exact_pts - result_pts) * P(score = i,j)
    P(score) comes from the DC matrix; P(result) from the (better) ensemble.
    """
    best = None
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ep = result_pts * outcome_probs[_result(i, j)] + (exact_pts - result_pts) * matrix[i, j]
            if best is None or ep > best[1]:
                best = ((i, j), ep)
    (pi, pj), ep = best
    return {"score": (pi, pj), "result": _result(pi, pj), "exp_pts": round(float(ep), 3)}


def score_pick(pick_score, actual_score,
               result_pts: float = RESULT_PTS, exact_pts: float = EXACT_PTS) -> dict:
    """Award pool points for one frozen pick vs the actual result.

    Non-additive: an exact-score match earns exact_pts as the TOTAL; a correct
    result with the wrong score earns result_pts; otherwise zero. (Exact is always
    also a correct result, so it supersedes rather than stacks.) Returns the
    breakdown so the dashboard can show ✓/✗.
    """
    pi, pj = int(pick_score[0]), int(pick_score[1])
    ai, aj = int(actual_score[0]), int(actual_score[1])
    result_hit = _result(pi, pj) == _result(ai, aj)
    exact_hit = (pi == ai) and (pj == aj)
    pts = exact_pts if exact_hit else (result_pts if result_hit else 0)
    return {"points": pts, "result_hit": result_hit, "exact_hit": exact_hit,
            "max_points": exact_pts}
