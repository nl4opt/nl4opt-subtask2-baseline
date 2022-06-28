LT = 'LESS_OR_EQUAL'
GT = 'GREATER_OR_EQUAL'

SUM_CONSTRAINT = '[SUM_CONSTRAINT]'
UPPER_BOUND = '[UPPER_BOUND]'
LOWER_BOUND = '[LOWER_BOUND]'
LINEAR_CONSTRAINT = '[LINEAR_CONSTRAINT]'
RATIO_CONTROL_CONSTRAINT = '[RATIO_CONSTRAINT]'
BALANCE_CONSTRAINT_1 = '[XBY_CONSTRAINT]'
BALANCE_CONSTRAINT_2 = '[XY_CONSTRAINT]'

TYPE_DICT = {
    'sum': SUM_CONSTRAINT,
    'upperbound': UPPER_BOUND,
    'lowerbound': LOWER_BOUND,
    'linear': LINEAR_CONSTRAINT,
    'ratio': RATIO_CONTROL_CONSTRAINT,
    'xby': BALANCE_CONSTRAINT_1,
    'xy': BALANCE_CONSTRAINT_2
}

# word to number mappings in ground truth
NUMS_DICT = {
    'a half': 1 / 2,
    'half': 1 / 2,
    'one half': 1 / 2,
    'one third': 1 / 3,
    'third': 1 / 3,
    'a third': 1 / 3,
    'one fourth': 1 / 4,
    'fourth': 1 / 4,
    'a fourth': 1 / 4,
    'one fifth': 1 / 5,
    'fifth': 1 / 5,
    'a fifth': 1 / 5,
    'twice': 2,
    'thrice': 3
}