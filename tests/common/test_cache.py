from morloquio.common.cache import params_hash


def test_params_hash_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_length_and_sensitivity():
    assert len(params_hash({})) == 8
    assert params_hash({"a": 1}) != params_hash({"a": 2})
    assert params_hash({"a": {"x": [1, 2]}}) == params_hash({"a": {"x": [1, 2]}})
