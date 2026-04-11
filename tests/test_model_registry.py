from utils.model_registry import list_model_names


def test_registry_contains_exactly_eight_supported_models():
    assert list_model_names() == [
        "gmm_no_forgetting_no_lr",
        "gmm_no_forgetting_tied_lrs",
        "gmm_forgetting_no_lr",
        "gmm_forgetting_tied_lrs",
        "wm_exponential_decay",
        "wm_exponential_decay_dual",
        "wm_last_m",
        "wm_last_m_category_specific",
    ]
