"""Mixed spiking/graded simulation of the MaleCNS connectome.

The original fly.ai model treats every cell as a leaky integrate-and-fire unit.
That is useful for whole-network experiments, but it breaks the earliest visual
relay because Drosophila photoreceptors and first-order lamina monopolar cells
carry graded membrane signals rather than communicating only with spikes.

Flybit keeps the original model as the default. With graded_visual=True,
R1-6/R7/R8 photoreceptors and L1/L2/L3 lamina cells keep continuous membrane
state and release a signed analog signal through their measured MaleCNS
connections. All other neurons retain the upstream LIF dynamics.

The graded parameters are computational approximations, not measured membrane
models. The wiring, transmitter-derived synapse sign and cell identities remain
those of MaleCNS v1.0.
"""
from __future__ import annotations

import os
from pathlib import Path

import numba
import numpy as np
from scipy import sparse

from .data import DATA, ensure_data


GRADED_VISUAL_TYPES = ("R1-6", "R7", "R8", "L1", "L2", "L3")


@numba.njit(nogil=True, parallel=True)
def _propagate(indptr, indices, weights, fired, n):
    """Sum outgoing weights (CSC columns) from spiking neurons."""
    threads = numba.get_num_threads()
    partial = np.zeros((threads, n), np.float32)
    chunk = (len(fired) + threads - 1) // threads
    for t in numba.prange(threads):
        acc = partial[t]
        for k in range(t * chunk, min(len(fired), (t + 1) * chunk)):
            j = fired[k]
            for e in range(indptr[j], indptr[j + 1]):
                acc[indices[e]] += weights[e]
    current = np.zeros(n, np.float32)
    for i in numba.prange(n):
        s = np.float32(0.0)
        for t in range(threads):
            s += partial[t, i]
        current[i] = s
    return current


def cuda_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class FlyBrain:
    """MaleCNS neural dynamics on CPU or NVIDIA CUDA.

    graded_visual=False preserves the upstream all-LIF model.

    graded_visual=True creates a mixed-signal network:
      * R1-6, R7, R8, L1, L2 and L3 carry continuous membrane state.
      * Their synaptic output is tanh(membrane / graded_scale), so
        hyperpolarization and depolarization both propagate.
      * The sign and strength of every connection still come from the MaleCNS
        matrix built by flybrain/build.py.
      * All other cells remain LIF neurons and emit discrete spikes.
    """

    dt = 0.020
    tau = 0.100
    gain = 3.0
    tonic = 0.14
    noise_hz = 1.2
    noise_amp = 0.22
    eye_gain = 0.62

    # Conservative continuous dynamics for cells with direct physiological
    # evidence of graded visual signalling. These values set numerical
    # timescale/range only; they are not claimed as fitted biophysical values.
    graded_tau = 0.050
    graded_gain = 1.0
    graded_scale = 0.50
    graded_clip = 2.0

    def __init__(
        self,
        data: Path | str | None = None,
        seed: int = 64,
        device: str | None = None,
        batch: int = 1,
        dt: float | None = None,
        sensory_input: bool = True,
        refractory: float = 0.0,
        graded_visual: bool = False,
    ):
        device = device or os.environ.get("FLY_DEVICE", "cpu")
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be cpu, cuda or auto, not {device!r}")

        self.device = device
        self.batch = int(batch)
        self.graded_visual = bool(graded_visual)
        if dt is not None:
            self.dt = float(dt)

        self.tonic = type(self).tonic * (
            (1 - np.exp(-self.dt / self.tau))
            / (1 - np.exp(-0.020 / self.tau))
        )
        self.refractory_steps = int(round(refractory / self.dt))
        self.sensory_input = sensory_input

        data = ensure_data(data)
        meta = np.load(data / "brain.npz")
        W = sparse.load_npz(data / "weights.npz").tocsr().astype(np.float32)

        self.visual = meta["visual"]
        self.azimuth = meta["azimuth"]
        self.cell_type = meta["cell_type"]
        self.side = meta["side"]
        self.positions = meta["positions"] if "positions" in meta.files else None
        self.superclass = meta["superclass"] if "superclass" in meta.files else None
        self.groups = {
            k.removeprefix("group_"): meta[k]
            for k in meta.files
            if k.startswith("group_")
        }

        if not sensory_input:
            if self.superclass is None:
                raise RuntimeError("brain.npz has no superclass; run `flybrain build`")
            sensory = np.char.find(self.superclass.astype(str), "sensory") >= 0
            W = sparse.diags((~sensory).astype(np.float32)) @ W
            W = W.tocsr().astype(np.float32)

        self.n = W.shape[0]

        if self.graded_visual:
            graded_mask = np.isin(self.cell_type.astype(str), GRADED_VISUAL_TYPES)
            self.graded = np.flatnonzero(graded_mask).astype(np.int32)
            if len(self.graded) == 0:
                raise RuntimeError(
                    "graded_visual=True but no R1-8/L1-3 cell types were found"
                )
        else:
            self.graded = np.empty(0, np.int32)

        spiking_mask = np.ones(self.n, dtype=np.bool_)
        spiking_mask[self.graded] = False
        self.spiking = np.flatnonzero(spiking_mask).astype(np.int32)

        # Sparse matrices for analog and spike propagation.
        if self.graded_visual:
            W_graded = W[:, self.graded].tocsr()
        else:
            W_graded = None

        if device == "cuda":
            import cupy
            from cupyx.scipy import sparse as cusparse

            self.xp = cupy
            self._W = cusparse.csr_matrix(W)
            self._W_graded = (
                cusparse.csr_matrix(W_graded) if W_graded is not None else None
            )
        else:
            self.xp = np
            self._W_graded = W_graded

        Wc = W.tocsc()
        self.indptr = Wc.indptr
        self.indices = Wc.indices
        self.weights = Wc.data

        self._visual = self.xp.asarray(self.visual)
        self._graded = self.xp.asarray(self.graded)
        self._spiking = self.xp.asarray(self.spiking)

        self.decay = np.float32(np.exp(-self.dt / self.tau))
        self.graded_decay = np.float32(np.exp(-self.dt / self.graded_tau))
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset dynamic state and restart stochastic spike noise."""
        xp = self.xp
        self.rng = xp.random.default_rng(seed)
        self.v = xp.zeros((self.n, self.batch), xp.float32)
        self.fired = xp.empty(0, xp.int64)
        self.steps = 0
        self.graded_output = xp.zeros(
            (len(self.graded), self.batch), xp.float32
        )
        self.last_spike = (
            xp.full((self.n, self.batch), -10**6, xp.int32)
            if self.refractory_steps
            else None
        )

    def cells(self, types: list[str], side: str | None = None) -> np.ndarray:
        """Indices matching cell types or superclass names."""
        mask = np.isin(self.cell_type, types)
        if self.superclass is not None:
            mask |= np.isin(self.superclass, types)
        if side:
            mask &= self.side == side
        return np.flatnonzero(mask)

    def membrane_values(self, types: list[str]) -> np.ndarray:
        """Copy current membrane state for named cells to host NumPy."""
        idx = self.cells(types)
        if not len(idx):
            return np.empty((0, self.batch), np.float32)
        arr = self.v[self.xp.asarray(idx)]
        if self.xp is not np:
            arr = arr.get()
        return np.asarray(arr)

    def _amount(self, amount):
        a = self.xp.asarray(amount, dtype=self.xp.float32)
        return a if a.ndim == 0 else a.reshape(1, -1)

    def stimulate(self, idx: np.ndarray, amount) -> None:
        self.v[self.xp.asarray(idx)] += self._amount(amount)

    def synaptic_input(self, fired):
        """Current from discrete spikes from the previous timestep."""
        xp, B = self.xp, self.batch
        if self.device == "cuda":
            spikes = xp.zeros((self.n, B), xp.float32)
            spikes.ravel()[fired] = 1.0
            if B == 1:
                return (self._W @ spikes[:, 0])[:, None]
            return self._W @ spikes

        rows, cols = np.divmod(fired, B)
        return np.column_stack(
            [
                _propagate(
                    self.indptr,
                    self.indices,
                    self.weights,
                    rows[cols == b],
                    self.n,
                )
                for b in range(B)
            ]
        )

    def graded_synaptic_input(self):
        """Continuous synaptic current from graded visual neurons."""
        xp = self.xp
        if not self.graded_visual or not len(self.graded):
            return xp.zeros((self.n, self.batch), xp.float32)
        current = self._W_graded @ self.graded_output
        if current.ndim == 1:
            current = current[:, None]
        return current

    def step(self, eye_drive: np.ndarray | None = None, inject=()):
        """Advance one timestep.

        In the legacy model, eye_drive is the upstream positive 0..1 drive.

        With graded_visual=True, eye_drive is expected to be a signed
        photoreceptor contrast signal in [-1, 1]. Positive values depolarize
        photoreceptors; negative values hyperpolarize them. No feature neuron
        is stimulated directly.
        """
        xp, B = self.xp, self.batch

        current = self.synaptic_input(self.fired)
        if self.graded_visual:
            current = current + self.graded_synaptic_input()

            self.v[self._spiking] *= self.decay
            self.v[self._graded] *= self.graded_decay

            self.v[self._spiking] += (
                current[self._spiking] * np.float32(self.gain) + self.tonic
            )
            self.v[self._spiking] += (
                self.rng.random((len(self.spiking), B))
                < self.noise_hz * self.dt
            ) * np.float32(self.noise_amp)

            self.v[self._graded] += (
                current[self._graded] * np.float32(self.graded_gain)
            )
        else:
            self.v *= self.decay
            self.v += current * self.gain + self.tonic
            self.v += (
                self.rng.random((self.n, B)) < self.noise_hz * self.dt
            ) * np.float32(self.noise_amp)

        if eye_drive is not None:
            drive = xp.asarray(eye_drive, dtype=xp.float32)
            self.v[self._visual] += (
                drive[:, None] if drive.ndim == 1 else drive
            ) * self.eye_gain

        for idx, amount in inject:
            self.v[xp.asarray(idx)] += self._amount(amount)

        if self.refractory_steps:
            blocked = (self.steps - self.last_spike) <= self.refractory_steps
            if self.graded_visual and len(self.graded):
                blocked[self._graded] = False
            self.v[blocked] = 0.0

        if self.graded_visual and len(self.graded):
            clipped = xp.clip(
                self.v[self._graded],
                -np.float32(self.graded_clip),
                np.float32(self.graded_clip),
            )
            self.v[self._graded] = clipped
            self.graded_output = xp.tanh(
                clipped / np.float32(self.graded_scale)
            )

            eligible = self.v >= 1.0
            eligible[self._graded] = False
            fired = xp.flatnonzero(eligible)
        else:
            fired = xp.flatnonzero(self.v >= 1.0)

        self.v.ravel()[fired] = 0.0
        if self.refractory_steps:
            self.last_spike.ravel()[fired] = self.steps

        self.fired = fired
        self.steps += 1

        flat = fired if xp is np else fired.get()
        if B == 1:
            return flat

        rows, cols = np.divmod(flat, B)
        order = np.argsort(cols, kind="stable")
        return np.split(
            rows[order],
            np.cumsum(np.bincount(cols, minlength=B))[:-1],
        )
