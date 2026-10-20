import numpy as np
from diagnostics.m5c_localization import matching, absolute_metrics


def test_height_and_each_gt_unmatched():
    shape = (4, 160, 120)
    p = np.zeros(shape, bool)
    g = p.copy()
    p[1, 10, 10] = True
    g[1, 10, 10] = True
    g[2, 10, 10] = True
    g[1, 20, 10] = True
    r = matching(p, g)["tolerances"]["1.0"]
    assert r["matched_gt"] == 1 and r["unmatched_gt"] == 2


def test_extra_prediction_and_known_unknown_separation():
    shape = (4, 160, 120)
    p = np.zeros(shape)
    g = np.zeros(shape, bool)
    v = g.copy()
    p[1, 10, 10] = 1
    p[1, 30, 10] = 1
    p[3, 120, 110] = 1
    g[1, 10, 10] = True
    v[1, 10, 10] = v[1, 30, 10] = True
    r = absolute_metrics(p, g, v, 0.5)
    assert r["unknown_predicted"] == 1 and r["exact"]["fp"] == 1
    assert r["spatial"]["tolerances"]["1.0"]["extra_pred"] == 1


def test_no_predictions_unmatched_all():
    p = np.zeros((4, 160, 120), bool)
    g = p.copy()
    g[2, 30, 40] = True
    assert matching(p, g)["tolerances"]["0.5"]["unmatched_gt"] == 1
