import torch
from conformer_model import Conformer8Mic
from evidential_model import EvidentialLocalization

model = torch.load("best_evidential_model.pt", map_location="cpu", weights_only=False)
print(model)