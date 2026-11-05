def expected_binary_cost(y_true, decisions, false_positive_cost=1.0, false_negative_cost=10.0):
    fp = ((decisions == 1) & (y_true == 0)).sum()
    fn = ((decisions == 0) & (y_true == 1)).sum()
    return float(fp * false_positive_cost + fn * false_negative_cost)
