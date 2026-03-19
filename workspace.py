import torch
import torch.nn as nn

print(torch.cuda.is_available())        # should be True
print(torch.cuda.get_device_name(0))    # should show RTX name
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # should show ~8GB

