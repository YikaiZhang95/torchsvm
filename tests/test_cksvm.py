import unittest
import numpy as np
from torchsvm.cksvm import CKSVM
from torchsvm.kernels import linear_kernel

class TestCKSVM(unittest.TestCase):
    def test_fit_predict(self):
        # Simple training set: 2D
        X_train = np.array([[0, 0],
                            [1, 1]])
        y_train = np.array([-1, 1])

        model = CKSVM(kernel=linear_kernel)
        model.fit(X_train, y_train)

        # Check that alpha is set
        self.assertIsNotNone(model.alpha, "Alpha should not be None after fit")

        X_test = np.array([[0.5, 0.5]])
        y_pred = model.predict(X_test)
        
        # This is a naive test checking the shape of predictions
        self.assertEqual(len(y_pred), 1, "Prediction shape mismatch")

        # Optionally, check sign or magnitude
        # For a real test, you'd compare against known correct labels
        self.assertIn(y_pred[0], [-1, 1], "Predicted label should be -1 or 1")

if __name__ == '__main__':
    unittest.main()
