
import unittest

from services.monitoring.drift import (
    calculate_psi,
    classify_psi,
    check_sample_size,
)


class TestPSICalculation(unittest.TestCase):

    def test_identical_distributions_have_zero_psi(self):
        result = calculate_psi([25, 25, 50], [25, 25, 50])
        self.assertAlmostEqual(result, 0.0)

    def test_proportional_counts_have_zero_psi(self):
        result = calculate_psi([10, 20, 30], [100, 200, 300])
        self.assertAlmostEqual(result, 0.0)

    def test_different_distributions_have_positive_psi(self):
        result = calculate_psi([80, 20], [20, 80])
        self.assertGreater(result, 0.0)

    def test_mismatched_distribution_lengths_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_psi([10, 20], [10, 20, 30])

    def test_empty_distributions_raise_error(self):
        with self.assertRaises(ValueError):
            calculate_psi([], [])

    def test_zero_total_distribution_raises_error(self):
        with self.assertRaises(ValueError):
            calculate_psi([0, 0], [10, 20])


class TestPSIClassification(unittest.TestCase):

    def test_stable_boundary(self):
        self.assertEqual(classify_psi(0.099), "STABLE")

    def test_warning_boundary(self):
        self.assertEqual(classify_psi(0.10), "WARNING")

    def test_drift_boundary(self):
        self.assertEqual(classify_psi(0.25), "DRIFT")


class TestSampleSizeGuard(unittest.TestCase):

    def test_below_minimum_is_insufficient(self):
        self.assertEqual(check_sample_size(99), "INSUFFICIENT_DATA")

    def test_exact_minimum_is_ready(self):
        self.assertEqual(check_sample_size(100), "READY")

    def test_custom_minimum(self):
        self.assertEqual(check_sample_size(49, minimum_samples=50), "INSUFFICIENT_DATA")
        self.assertEqual(check_sample_size(50, minimum_samples=50), "READY")


if __name__ == "__main__":
    unittest.main()
