import unittest
import torch
# add path to sys.path
import sys
sys.path.append('/home/behrendt/projects/MONAI/MONAI')
from monai.inferers import SliceInferer

# Todo: run this tests with spatial and dense conditioning data, with different roi sizes, spatial dims, and sw batch sizes
def model(x, condition):
    
    if condition is not None:
        print(x.shape, condition.shape)
        if condition.dim() > 3:
            return x + condition
        else:
            return x 
    else:
        print(x.shape)
        return x
class TestSliceInferer(unittest.TestCase):
    def test_inference_with_condition(self):
        # model = lambda x, condition: x + condition if condition is not None else x
        inferer = SliceInferer(roi_size=(128, 128), spatial_dim=0)
        
        input_data = torch.randn(1, 1, 32, 128, 128)
        condition = torch.ones(1, 1, 32, 128)
        
        result = inferer(input_data, model, condition)
        if condition.dim() > 4:
            self.assertEqual(result.shape, input_data.shape)
            self.assertEqual(result.sum(), (input_data + condition).sum())
        else:
            self.assertEqual(result.shape, input_data.shape)
            self.assertEqual(result.sum(), input_data.sum())

if __name__ == "__main__":
    unittest.main()