from __future__ import annotations
from datetime import datetime, timezone
import unittest

from flybit.controller import TimedNeuralOutput
from flybit.motion import FlyBodyState, MotorActivity
from flybit.simulation import FlybitSimulation
from flybit.state import FlybitState


class SimulationTest(unittest.TestCase):
    def make(self):
        state = FlybitState(created_at=datetime.now(timezone.utc).isoformat())
        return FlybitSimulation(state, FlyBodyState(300, 200), seed=4)

    def test_monotonic_steps_and_stale_telemetry(self):
        sim = self.make()
        sim.set_neural_output(TimedNeuralOutput(MotorActivity(forward_left=.5, forward_right=.5), 10.0, 3))
        a = sim.tick(.02, timestamp=10.02)
        b = sim.tick(.02, timestamp=10.30)
        self.assertEqual((a.step, b.step), (1, 2))
        self.assertFalse(a.stale_neural_output)
        self.assertTrue(b.stale_neural_output)

    def test_older_neural_snapshot_is_rejected(self):
        sim = self.make()
        sim.set_neural_output(TimedNeuralOutput(MotorActivity(forward_left=1), 5.0, 5))
        sim.set_neural_output(TimedNeuralOutput(MotorActivity(), 4.0, 4))
        snap = sim.tick(.02, timestamp=5.02)
        self.assertEqual(snap.neural_step, 5)

if __name__ == "__main__": unittest.main()
