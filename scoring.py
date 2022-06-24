import numpy as np
from typing import List, Tuple


def per_example_scores(predicted_objective: np.ndarray, predicted_constraints: np.ndarray,
                       true_objective: np.ndarray, true_constraints: np.ndarray) -> Tuple[int, int, int]:
    """
    Computes score of a single problem
    :param predicted_objective:     vector of length k1,    k1 = number of predicted variables
    :param predicted_constraints:   d1 x k1 matrix,         d1 = number of predicted constraints
    :param true_objective:          vector of length k2,    k1 = number of ground truth variables
    :param true_constraints:        d2 x k2 matrix,         d2 = number of ground truth constraints
    :return: false positives, false negatives, number of declarations in ground truth
    """
    # remove duplicate constraints
    constraints = np.unique(predicted_constraints, axis=0)

    def obj_eq(a: np.ndarray, b: np.ndarray):
        return a.shape == b.shape and (predicted_objective == true_objective).all()

    def constraint_isin(a: np.ndarray, B: np.ndarray):
        # check if shape matches and if row a matches with any row in B
        return len(a) > 0 and len(B) > 0 and \
               a.shape[0] == B.shape[1] and (a == B).all(axis=1).any()

    d = 1 + len(true_constraints)
    fp = 0
    fn = max(len(true_constraints) - len(constraints), 0)

    if not obj_eq(predicted_objective, true_objective):
        fp += 1
    # check how many of the predicted constraints match the true constraints
    matches = np.sum([constraint_isin(item, true_constraints) for item in constraints])
    fp += len(constraints) - matches
    # cap number of false positives to d,as having way too many constraints could result in negative accuracy
    # todo: discuss further
    fp = min(fp, d)

    return fp, int(fn), d


def overall_score(predicted_objectives: List[np.ndarray], predicted_constraints: List[np.ndarray],
                  true_objectives: List[np.ndarray], true_constraints: List[np.ndarray]) -> float:
    """
    Computes overall score on multiple problems
    :param predicted_objectives:
    :param predicted_constraints:
    :param true_objectives:
    :param true_constraints:
    :return: overall score
    """
    numerator = 0
    denominator = 0
    for p_obj, p_const, t_obj, t_const in zip(predicted_objectives, predicted_constraints,
                                              true_objectives, true_constraints):
        fp, fn, d = per_example_scores(p_obj, p_const, t_obj, t_const)
        numerator += fp + fn
        denominator += d
    return 1 - numerator / denominator
