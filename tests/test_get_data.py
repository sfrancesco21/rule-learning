import numpy as np

from utils.get_data import load_participant_data, read_mat_files


def test_single_participant_loader_matches_bulk_loader():
    participant_id = 1
    single = load_participant_data(participant_id)
    bulk = read_mat_files()[participant_id]

    for phase in ("training", "test"):
        assert single[phase].keys() == bulk[phase].keys()
        for key in single[phase]:
            left = single[phase][key]
            right = bulk[phase][key]
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right
