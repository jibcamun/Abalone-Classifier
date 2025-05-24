import torch
import torch.nn as nn

class Abalone_Classification_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 30)
        )

    def forward(self, x):
        return self.model(x)

model = Abalone_Classification_Model(8, 30).to("cpu")
model.load_state_dict(torch.load("model.pt"))
model.eval()

sample = [3,0.425,0.3,0.095,0.3515,0.141,0.0775,0.12]
X_single = torch.FloatTensor(sample).unsqueeze(0)
with torch.no_grad():
    prediction = model(X_single)
    predicted_class = torch.argmax(prediction, dim=1).item()

print(f"Predicted Class: {predicted_class + 1}")