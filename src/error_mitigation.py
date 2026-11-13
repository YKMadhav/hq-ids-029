"""
Project HQ-IDS-029: NISQ Noise Models & Readout Error Mitigation
=================================================================
Implements:
1. NISQNoiseModelBuilder: assembles a `qiskit_aer.noise.NoiseModel` representing
   IBM-like superconducting hardware:
     - single-qubit depolarizing error, p = 0.001 on 1q gates
     - two-qubit CNOT depolarizing error, p = 0.01
     - symmetric readout bit-flip error, p_readout = 0.03 on all qubits
2. MatrixInversionReadoutMitigator: tensor-product assignment-matrix readout
   mitigation.  This is *not* IBM's M3 package or a hardware calibration run.
   2^N x 2^N assignment matrix M (readout = tensor product of per-qubit
   channels, so f_measured = M f_true), inverts it exactly, and inverts raw shot
   counts to recover unbiased Pauli-Z expectation values.
"""

import numpy as np
import qiskit_aer.noise as noise


class NISQNoiseModelBuilder:
    """Builds realistic IBM Quantum hardware noise models for Aer simulation."""

    @staticmethod
    def build_ibm_noise_model(p_single=0.001, p_cnot=0.01, p_readout=0.03):
        """
        Constructs a noise model with depolarizing errors on single-qubit and two-qubit gates,
        plus readout bit-flip probabilities.
        """
        noise_model = noise.NoiseModel()

        # Single-qubit gate errors
        error_single = noise.depolarizing_error(p_single, 1)
        noise_model.add_all_qubit_quantum_error(
            error_single, ["ry", "rz", "x", "h", "id", "p", "u", "u1", "u2", "u3"]
        )

        # Two-qubit CNOT gate errors
        error_cnot = noise.depolarizing_error(p_cnot, 2)
        noise_model.add_all_qubit_quantum_error(error_cnot, ["cx", "cz"])

        # Readout assignment errors
        probabilities = [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
        error_readout = noise.ReadoutError(probabilities)
        noise_model.add_all_qubit_readout_error(error_readout)

        return noise_model


class MatrixInversionReadoutMitigator:
    """
    Tensor-product matrix-inversion readout mitigation for N-qubit measurements.

    With an independent symmetric bit-flip channel on every qubit, the measured
    distribution is related to the ideal one by

        f_measured = M f_true,      M = kron_{i=1..N} A1,
        A1 = [[1 - p, p], [p, 1 - p]].

    M is a 2^N x 2^N tensor-product assignment matrix.  This implementation
    inverts that *assumed simulated* matrix exactly; it must not be described as
    M3 or as hardware-calibrated mitigation.

        f_true = M^{-1} f_measured,

    and the corrected Pauli-Z expectation values follow from the marginals of
    f_true. Condition numbers are reported so that unstable inversions (large p)
    are visible rather than silently producing out-of-range values.
    """

    def __init__(self, num_qubits=4, p_readout=0.03):
        self.num_qubits = num_qubits
        self.p_readout = p_readout
        self.dim = 2 ** num_qubits
        self.M = self._build_assignment_matrix()
        self.M_inv = np.linalg.inv(self.M)
        self.condition_number = float(np.linalg.cond(self.M))

    def _build_assignment_matrix(self):
        """
        Constructs the 2^N x 2^N tensor-product assignment matrix.

        Entry M[k, j] = P(measured k | true j). Because the per-qubit channel is
        independent and identical, M[k, j] = prod_i A1[bit_i(k), bit_i(j)].
        """
        A1 = np.array(
            [[1 - self.p_readout, self.p_readout], [self.p_readout, 1 - self.p_readout]]
        )
        M = np.array([[1.0]])
        for _ in range(self.num_qubits):
            M = np.kron(M, A1)
        return M

    def counts_to_distribution(self, counts):
        """
        Converts a Qiskit count dictionary into a normalised 2^N probability vector.

        Qiskit count keys are big-endian bitstrings of the classical registers
        (leftmost character is the highest classical bit / highest qubit index), so
        the integer value of the key indexes basis states with qubit i on bit i.
        """
        distribution = np.zeros(self.dim, dtype=np.float64)
        total = 0
        for bitstring, count in counts.items():
            key = int(bitstring.replace(" ", ""), 2)
            distribution[key] += count
            total += count
        if total <= 0:
            raise ValueError("Empty counts dictionary; cannot normalise.")
        return distribution / total

    def expectations_from_distribution(self, distribution):
        """
        Computes <Z_i> for every qubit from a (mitigated) probability vector.
        """
        expectations = np.zeros(self.num_qubits, dtype=np.float64)
        for qubit in range(self.num_qubits):
            mask = 1 << qubit
            p_one = sum(p for index, p in enumerate(distribution) if index & mask)
            expectations[qubit] = 1.0 - 2.0 * p_one
        return expectations

    def mitigate_counts(self, counts):
        """
        Invert the 2^N x 2^N assignment matrix on the measured distribution
        and return the corrected Pauli-Z expectation values.

        The inverted distribution is projected back onto the probability simplex
        with the Euclidean projection (not clipping-and-renormalising), because
        M^{-1} f can leave the simplex when the channel is ill-conditioned.
        """
        measured = self.counts_to_distribution(counts)
        mitigated = self._project_to_simplex(self.M_inv @ measured)
        return self.expectations_from_distribution(mitigated)

    @staticmethod
    def _project_to_simplex(values):
        """Euclidean projection onto {p: p >= 0, sum(p) = 1} (Duchi et al.)."""
        values = np.asarray(values, dtype=np.float64)
        ordered = np.sort(values)[::-1]
        cssv = np.cumsum(ordered) - 1.0
        rho = np.nonzero(ordered - cssv / np.arange(1, len(values) + 1) > 0)[0]
        if len(rho) == 0:  # Defensive; a finite vector always has a solution.
            return np.full_like(values, 1.0 / len(values))
        theta = cssv[rho[-1]] / float(rho[-1] + 1)
        return np.maximum(values - theta, 0.0)

    def raw_expectations(self, counts):
        """Unmitigated Pauli-Z expectation values <Z_i> = (N_0 - N_1) / shots."""
        return self.expectations_from_distribution(self.counts_to_distribution(counts))

    def mitigate_expectations(self, raw_expectations):
        """
        Marginal-only readout correction.

        For an independent per-qubit readout channel the 1-qubit marginal obeys
        E_meas[Z_i] = (1 - 2 p) E_true[Z_i] exactly, so rescaling by 1/(1 - 2p)
        is the closed-form limit of the matrix inversion restricted to one qubit.
        Provided for callers that only have expectation values, not counts;
        `mitigate_counts` is the full distribution-level path used in the NISQ
        experiments.
        """
        raw = np.asarray(raw_expectations, dtype=np.float64)
        corrected = raw / (1.0 - 2.0 * self.p_readout)
        return np.clip(corrected, -1.0, 1.0)


M3ReadoutMitigator = MatrixInversionReadoutMitigator
"""Deprecated compatibility alias; retained to avoid silently breaking old scripts."""


if __name__ == "__main__":
    print("Testing NISQNoiseModelBuilder & MatrixInversionReadoutMitigator...")

    builder = NISQNoiseModelBuilder()
    noise_model = builder.build_ibm_noise_model()
    print("Noise Model created successfully!")

    mitigator = MatrixInversionReadoutMitigator(num_qubits=4, p_readout=0.03)
    print(f"Assignment matrix shape: {mitigator.M.shape}, condition number: {mitigator.condition_number:.4f}")

    # Synthetic check: an ideal distribution pushed through a known readout channel
    # must be recovered by M3 up to the shot-noise-free limit.
    rng = np.random.default_rng(0)
    ideal = rng.random(mitigator.dim)
    ideal /= ideal.sum()
    A1 = np.array([[0.97, 0.03], [0.03, 0.97]])
    M = np.array([[1.0]])
    for _ in range(4):
        M = np.kron(M, A1)
    measured = M @ ideal
    counts = {format(k, "04b"): int(round(p * 1_000_000)) for k, p in enumerate(measured)}
    counts = {k: v for k, v in counts.items() if v > 0}

    recovered = mitigator.mitigate_counts(counts)
    ideal_z = mitigator.expectations_from_distribution(ideal)
    error = np.abs(recovered - ideal_z).max()
    print(f"Ideal <Z>       : {np.round(ideal_z, 6)}")
    print(f"Recovered <Z>   : {np.round(recovered, 6)}")
    print(f"Max recovery err: {error:.3e}")
    assert error < 1e-4, "M3 failed to invert the assignment matrix."
    print("Self-test PASSED!")
