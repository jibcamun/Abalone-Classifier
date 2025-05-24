import torch
from torch import nn, save
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("data/abalone.csv")
data.dropna()
data["Sex"] = pd.factorize(data["Sex"])[0]

X = data.drop("Rings", axis=1).values

y = data["Rings"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
y_train, y_test = torch.LongTensor(y_train), torch.LongTensor(y_test)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Abalone_Classification_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, y.max() + 1)
        )

    def forward(self, x):
        return self.model(x)

model = Abalone_Classification_Model(8, y.max() + 1).to("cpu")
optimizer = SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    for epoch in range(1000):
        for batch in dataloader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to("cpu"), y_batch.to("cpu")

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    with torch.no_grad():
        y_eval = model(X_test)
        eval_loss = criterion(y_eval, y_test)

    print(f"Final Loss: {eval_loss.item()}")

    with open("model.pt", 'wb') as f:
        save(model.state_dict(), f)