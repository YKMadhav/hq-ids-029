import numpy as np
import pandas as pd
import pytest

from scripts.run_multiclass_benchmark import map_attack_categories
from src.biflow_loader import _protocol_code
from src.error_mitigation import MatrixInversionReadoutMitigator


def test_matrix_inversion_uses_a_real_simplex_projection():
    projected = MatrixInversionReadoutMitigator._project_to_simplex(
        np.array([1.2, -0.1, -0.1])
    )
    np.testing.assert_allclose(projected, [1.0, 0.0, 0.0])
    assert np.isclose(projected.sum(), 1.0)
    assert (projected >= 0).all()


def test_phase4_protocol_codes_do_not_collide_for_arp_and_hopopt():
    encoded = _protocol_code(pd.Series(["arp", "hopopt"]))
    assert encoded[0] != encoded[1]


def test_unknown_multiclass_label_is_not_silently_mapped_to_normal():
    with pytest.raises(ValueError, match="Unknown attack categories"):
        map_attack_categories(pd.Series(["Normal", "FutureAttackFamily"]))
