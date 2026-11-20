import numpy as np

from a3opt.backends.adaptive_phase import OPEN, RETRACT
from a3opt.placement_hybrid import fit as fit_combination
from a3opt.placement_pooled import fit as fit_shared_offset
from a3opt.placement_release import adjust_release
from a3opt.placement_sim import Correction
from a3opt.placement_tail import OFFSETS


def receipt(column, count, shifted, *, unsafe=False):
    options = {}
    for name in OFFSETS:
        observed = shifted if name == "out3" else count
        options[name] = {"wrong_source": 0, "physics": {"counted_ids": list(range(observed)), "filled_slots": observed,
                         "source_remaining": 70, "safety_reason": "stop" if unsafe and name == "out3" else None}}
    return {"column": column, "fork": {"outcomes": options}}


def test_shared_offset_requires_repeated_gain_across_columns_with_no_unsafe_tail():
    train = [{"reference": receipt(column, 5, 10)} for column in (1, 2)]
    expected = tuple(Correction(column, 1, OFFSETS["out3"]) for column in range(1, 5))
    assert fit_shared_offset(train)[0] == expected
    assert fit_shared_offset([train[0], train[0]])[0] == ()
    assert fit_shared_offset([*train, {"reference": receipt(3, 10, 10, unsafe=True)}])[0] == ()


def test_first_batch_admission_uses_the_same_second_batch_action_on_both_states():
    anchor = tuple(Correction(column, 0, (-.003, 0., .008), .03) for column in (1, 2, 3))
    cases = [(1, 5, 5, 10, 10), (2, 5, 5, 10, 10), (3, 5, 10, 10, 5)]
    pairs = [{"reference": receipt(column, base, shifted), "anchor": receipt(column, changed, combined)}
             for column, base, shifted, changed, combined in cases]
    pairs += [{"reference": receipt(column, 10, 10), "anchor": receipt(column, 10, 10)}
              for column in (1, 1, 2, 2, 3, 3)]
    selected, scores = fit_combination(pairs, anchor, "out3")
    assert selected == anchor[:2]
    assert scores["3"]["paired_deltas"] == [-5, 0, 0]


def test_faster_release_preserves_command_length_and_opening_endpoints():
    original = np.zeros((14, 16))
    original[:, 7] = np.linspace(.28, .48, 14)
    phases = [original.copy() for _ in range(RETRACT + 1)]
    policy = type("Policy", (), {"actions": [phases]})()
    adjust_release(policy, 0, "faster-opening")
    np.testing.assert_array_equal(phases[OPEN][[0, -1]], original[[0, -1]])
    np.testing.assert_array_equal(phases[OPEN][:, :7], original[:, :7])
    np.testing.assert_array_equal(phases[RETRACT], original)
    assert phases[OPEN][len(original) // 2, 7] == original[-1, 7]
