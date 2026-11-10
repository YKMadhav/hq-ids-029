"""
Project HQ-IDS-029: Parameterized Quantum Circuits (PennyLane & Qiskit)
=======================================================================
This module implements:
1. PennyLaneVQCModule: A PyTorch-compatible Variational Quantum Classifier (VQC)
   using PennyLane's default.qubit simulator and analytical backpropagation.
   - Feature Map: AngleEmbedding (Ry rotations) or ZZFeatureMap
   - Variational Ansatz: RealAmplitudes topology
     (Ry rotation layers + linear CNOT entangling rings, `reps` repetitions)
   - Measurement: Multi-qubit Pauli-Z expectation values <Z_i>
2. QiskitVQCBuilder: Qiskit circuit generator for transpilation, hardware realism,
   circuit depth / CNOT counting, and AerSimulator noise modeling.

The PennyLane and Qiskit implementations are verified to be numerically
equivalent (`verify_pennylane_qiskit_equivalence`) so that weights trained on the
analytical simulator can be replayed faithfully on the shot-based Aer backend.
"""

import os
import math
from typing import Dict, Any, Tuple, Optional, List

# Suppress font cache / matplotlib warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# Qiskit imports for circuit metrics, CNOT counting, and depth
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    # Support Qiskit 2.x function syntax and fallback to 1.x classes
    try:
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
        _USE_QISKIT_FUNCS = True
    except ImportError:
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        _USE_QISKIT_FUNCS = False
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


def _linear_pairs(num_qubits: int) -> List[Tuple[int, int]]:
    """Linear (nearest-neighbour) entanglement pairs, matching Qiskit's 'linear' topology."""
    return [(i, i + 1) for i in range(num_qubits - 1)]


def _apply_zz_feature_map(inputs, wires: List[int], reps: int = 1) -> None:
    """
    PennyLane transcription of Qiskit's `ZZFeatureMap(reps=reps, entanglement='linear')`.

    Per repetition:
        H^⊗n  ->  P(2*x_i) on each qubit
               ->  CX(i,j); P(2*(pi - x_i)*(pi - x_j)); CX(i,j) over linear pairs

    The data mapping matches Qiskit's `self_product`, whose multi-qubit coefficient is
    `prod_m(pi - x_m)`; the `pi - x` convention keeps the phase bounded for
    latent angles mapped from [-pi, pi].
    """
    pairs = _linear_pairs(len(wires))
    for _ in range(reps):
        for wire in wires:
            qml.Hadamard(wires=wire)
        # Index the trailing (latent) axis so that PennyLane broadcasting works
        # for both a single flow (4,) and a batch of flows (B, 4).
        for i in wires:
            qml.PhaseShift(2.0 * inputs[..., i], wires=i)
        for i, j in pairs:
            qml.CNOT(wires=[i, j])
            qml.PhaseShift(2.0 * (math.pi - inputs[..., i]) * (math.pi - inputs[..., j]), wires=j)
            qml.CNOT(wires=[i, j])


def _apply_real_amplitudes(weights, wires: List[int], reps: int = 2) -> None:
    """
    PennyLane transcription of Qiskit's `RealAmplitudes(reps=reps, entanglement='linear')`.

    `weights` has shape (reps + 1, num_qubits): an R_y rotation layer, a linear CNOT ring,
    repeated `reps` times, followed by one trailing rotation layer.
    """
    pairs = _linear_pairs(len(wires))
    for i in range(len(wires)):
        qml.RY(weights[0, i], wires=i)
    for r in range(reps):
        for i, j in pairs:
            qml.CNOT(wires=[i, j])
        for i in range(len(wires)):
            qml.RY(weights[r + 1, i], wires=i)


class PennyLaneVQCModule(nn.Module):
    """
    4-Qubit Variational Quantum Circuit implemented in PennyLane and wrapped
    as a PyTorch nn.Module for end-to-end backpropagation.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        ansatz_type: str = "RealAmplitudes",
        feature_map: str = "AngleEmbedding",
        feature_map_reps: int = 1,
        output_mode: str = "expectations",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_type = ansatz_type
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.output_mode = output_mode

        # Device initialization
        self.dev = qml.device(device_name, wires=num_qubits)

        if feature_map == "AngleEmbedding":
            encode = lambda inputs: qml.AngleEmbedding(
                inputs, wires=list(range(num_qubits)), rotation="Y"
            )
        elif feature_map == "ZZFeatureMap":
            encode = lambda inputs: _apply_zz_feature_map(
                inputs, list(range(num_qubits)), reps=feature_map_reps
            )
        else:
            raise ValueError(f"Unsupported feature_map: {feature_map}")

        def _measurement():
            if self.output_mode == "bitstrings":
                return qml.probs(wires=list(range(num_qubits)))
            else:
                return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        if ansatz_type == "RealAmplitudes":
            # RealAmplitudes: (num_layers + 1) Ry layers interleaved with linear CNOT rings
            self.weight_shapes = {"weights": (num_layers + 1, num_qubits)}

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                encode(inputs)
                _apply_real_amplitudes(weights, list(range(num_qubits)), reps=num_layers)
                return _measurement()

            self.circuit_fn = _circuit

        elif ansatz_type == "StronglyEntangling":
            # StronglyEntanglingLayers requires weight shape (num_layers, num_qubits, 3)
            self.weight_shapes = {"weights": (num_layers, num_qubits, 3)}

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                encode(inputs)
                qml.StronglyEntanglingLayers(weights, wires=list(range(num_qubits)))
                return _measurement()

            self.circuit_fn = _circuit

        elif ansatz_type == "BasicEntangler":
            # BasicEntanglerLayers requires weight shape (num_layers, num_qubits)
            self.weight_shapes = {"weights": (num_layers, num_qubits)}

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def _circuit(inputs, weights):
                encode(inputs)
                qml.BasicEntanglerLayers(weights, wires=list(range(num_qubits)))
                return _measurement()

            self.circuit_fn = _circuit

        else:
            raise ValueError(f"Unsupported ansatz_type: {ansatz_type}")

        # Wrap in native PennyLane TorchLayer
        self.qnn = qml.qnn.TorchLayer(self.circuit_fn, self.weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Batch of 4-dimensional continuous latent features x in [-pi, pi]^4.
        Output:
          - If output_mode == 'expectations': 4-dim Pauli-Z expectations in [-1, 1]^4.
          - If output_mode == 'bitstrings': 16-dim computational basis probabilities in Delta^15.
        """
        return self.qnn(x)


class QiskitVQCBuilder:
    """
    Builds parameterized quantum circuits in Qiskit for circuit complexity
    profiling (depth, 2-qubit CNOT count) and hardware noise simulation.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 2,
        entanglement: str = "linear",
        feature_map_type: str = "ZZFeatureMap"
    ):
        if not _QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is not available in the current environment.")

        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map_type = feature_map_type

        if feature_map_type == "ZZFeatureMap":
            if _USE_QISKIT_FUNCS:
                self.feature_map = zz_feature_map(
                    feature_dimension=num_qubits,
                    reps=1,
                    entanglement=entanglement,
                )
            else:
                self.feature_map = ZZFeatureMap(
                    feature_dimension=num_qubits,
                    reps=1,
                    entanglement=entanglement,
                )
        elif feature_map_type == "AngleEmbedding":
            # Qiskit transcription of PennyLane's AngleEmbedding(rotation='Y'):
            # one data-encoding R_y rotation per qubit.
            x = ParameterVector("x", num_qubits)
            self.feature_map = QuantumCircuit(num_qubits, name="AngleEmbedding")
            for qubit in range(num_qubits):
                self.feature_map.ry(x[qubit], qubit)
        else:
            raise ValueError(f"Unsupported feature_map_type: {feature_map_type}")

        if _USE_QISKIT_FUNCS:
            self.ansatz = real_amplitudes(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement,
            )
        else:
            self.ansatz = RealAmplitudes(
                num_qubits=num_qubits,
                reps=reps,
                entanglement=entanglement,
            )

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """
        Computes circuit depth, decomposed depth, CNOT count, parameter counts,
        and gate distributions.
        """
        qc = self.build_circuit(measure=False)
        decomposed = qc.decompose()
        # Remove measurement barrier / measure for pure gate depth
        pure_gate_circuit = decomposed.remove_final_measurements(inplace=False)
        ops = dict(pure_gate_circuit.count_ops())

        cnot_count = ops.get("cx", 0) + ops.get("cz", 0)
        total_gates = sum(ops.values())

        return {
            "num_qubits": self.num_qubits,
            "feature_map_type": self.feature_map_type,
            "depth": pure_gate_circuit.depth(),
            "raw_depth": qc.depth(),
            "cnot_count": cnot_count,
            "total_gates": total_gates,
            "num_parameters": qc.num_parameters,
            "feature_map_parameters": self.feature_map.num_parameters,
            "ansatz_parameters": self.ansatz.num_parameters,
            "ops": ops,
        }

    def build_circuit(self, measure: bool = True) -> "QuantumCircuit":
        """
        Constructs the composite circuit: feature map + RealAmplitudes ansatz.
        """
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)
        if measure:
            qc.measure_all()
        return qc

    def bind_circuit(self, latent: np.ndarray, weights: np.ndarray, measure: bool = True) -> "QuantumCircuit":
        """
        Returns a fully bound, decomposed circuit ready for Aer execution.

        Aer executes primitive gates, not the library's composite blueprint
        instructions, so the circuit is decomposed to expose `ry` / `cx` / `u` /
        `p` for the noise model to attach to.
        """
        latent = np.asarray(latent, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if latent.size != self.feature_map.num_parameters:
            raise ValueError(
                f"Expected {self.feature_map.num_parameters} latent angles, got {latent.size}."
            )
        if weights.size != self.ansatz.num_parameters:
            raise ValueError(
                f"Expected {self.ansatz.num_parameters} variational weights, got {weights.size}."
            )
        circuit = self.build_circuit(measure=measure)
        bound = circuit.assign_parameters(
            {
                **dict(zip(self.feature_map.parameters, latent)),
                **dict(zip(self.ansatz.parameters, weights)),
            },
            inplace=False,
        )
        return bound.decompose(reps=2)

    def draw_ascii(self) -> str:
        """Returns text-based ASCII representation of the quantum circuit."""
        qc = self.build_circuit()
        return qc.draw(output="text").__str__()


def verify_pennylane_qiskit_equivalence(
    num_qubits: int = 4,
    reps: int = 2,
    n_samples: int = 8,
    seed: int = 42,
    feature_map: str = "AngleEmbedding",
) -> float:
    """
    Confirms that the PennyLane VQC and the Qiskit circuit produce identical
    Pauli-Z expectation values for the same latent inputs and variational weights.

    This is what licenses transferring weights trained on the analytical
    default.qubit backend onto the shot-based Aer NISQ backend.

    Returns the maximum absolute deviation across all sampled circuits.
    """
    if not _QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required to verify circuit equivalence.")

    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    latents = rng.uniform(-np.pi, np.pi, size=(n_samples, num_qubits))
    weights = rng.uniform(-np.pi, np.pi, size=((reps + 1) * num_qubits,))

    # PennyLane (analytical)
    module = PennyLaneVQCModule(
        num_qubits=num_qubits,
        num_layers=reps,
        ansatz_type="RealAmplitudes",
        feature_map=feature_map,
        feature_map_reps=1,
    )
    # Inject identical variational weights so the two backends are directly comparable
    # (TorchLayer parameter order must therefore match Qiskit's `ansatz.parameters` order).
    with torch.no_grad():
        module.qnn.weights.copy_(
            torch.tensor(weights, dtype=torch.float32).reshape(module.qnn.weights.shape)
        )
    with torch.no_grad():
        pl_out = module(torch.tensor(latents, dtype=torch.float32)).numpy()

    # Qiskit (statevector)
    builder = QiskitVQCBuilder(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear",
        feature_map_type=feature_map,
    )
    deviations = []
    for index, latent in enumerate(latents):
        bound = builder.bind_circuit(latent, weights, measure=False)
        state = Statevector.from_instruction(bound).data
        probabilities = np.abs(state) ** 2
        marginals = []
        for qubit in range(num_qubits):
            # Qiskit indexes basis states little-endian: qubit 0 is bit 0.
            mask = 1 << qubit
            p_one = sum(p for k, p in enumerate(probabilities) if k & mask)
            marginals.append(1.0 - 2.0 * p_one)
        deviations.append(np.abs(np.array(marginals) - pl_out[index]).max())

    return float(max(deviations))


if __name__ == "__main__":
    print("=" * 70)
    print("Project HQ-IDS-029: Quantum Circuit Modules Self-Test")
    print("=" * 70)

    # 0. Cross-backend circuit equivalence for both supported feature maps
    for feature_map in ("AngleEmbedding", "ZZFeatureMap"):
        deviation = verify_pennylane_qiskit_equivalence(
            num_qubits=4, reps=2, n_samples=8, seed=42, feature_map=feature_map
        )
        print(f"\n[PennyLane vs Qiskit] feature_map={feature_map:15s} "
              f"max |<Z_i> deviation| = {deviation:.3e}")
        assert deviation < 1e-5, f"PennyLane and Qiskit {feature_map} circuits are NOT equivalent!"

    # 1. Test PennyLane PyTorch VQC Module
    print("\n[PennyLane] Initializing 4-qubit PennyLaneVQCModule (RealAmplitudes + AngleEmbedding)...")
    vqc = PennyLaneVQCModule(num_qubits=4, num_layers=2)

    dummy_latent = torch.tensor([
        [0.5, -1.2, 2.1, -0.8],
        [-2.0, 1.5, -0.4, 0.9]
    ], dtype=torch.float32)

    q_out = vqc(dummy_latent)
    print(f"[PennyLane] Input shape: {dummy_latent.shape} -> Expectation values shape: {q_out.shape}")
    print(f"[PennyLane] Expectation values sample:\n{q_out.detach().numpy()}")

    # Test backpropagation
    loss = q_out.sum()
    loss.backward()
    print("[PennyLane] Backward pass successful. Gradient computed for quantum weights!")
    assert q_out.shape == (2, 4), "PennyLane output shape mismatch!"

    # 2. Circuit complexity profiles for both feature maps
    print("\n[Qiskit] Circuit complexity profiles (depth / CNOT / parameters):")
    profiles = {}
    for feature_map in ("AngleEmbedding", "ZZFeatureMap"):
        builder = QiskitVQCBuilder(
            num_qubits=4, reps=2, entanglement="linear", feature_map_type=feature_map
        )
        metrics = builder.get_circuit_metrics()
        profiles[feature_map] = metrics
        print(f"[Qiskit] {feature_map:15s} -> depth {metrics['depth']:2d} | CNOTs {metrics['cnot_count']:2d} | "
              f"params {metrics['num_parameters']:2d} "
              f"({metrics['feature_map_parameters']} data + {metrics['ansatz_parameters']} variational) "
              f"| gates {metrics['ops']}")

    # The hardware-reference ZZ configuration must match the specified budget.
    reference = profiles["ZZFeatureMap"]
    assert reference["depth"] == 18, f"Expected depth 18, got {reference['depth']}"
    assert reference["cnot_count"] == 12, f"Expected 12 CNOTs, got {reference['cnot_count']}"
    assert reference["num_parameters"] == 16, f"Expected 16 parameters, got {reference['num_parameters']}"

    print("\n[Qiskit] Trained-circuit schematic (AngleEmbedding + RealAmplitudes):")
    print(QiskitVQCBuilder(
        num_qubits=4, reps=2, entanglement="linear", feature_map_type="AngleEmbedding"
    ).draw_ascii())

    print("\nAll quantum circuit modules verified successfully!")
