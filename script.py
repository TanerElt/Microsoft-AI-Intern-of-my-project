import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

df = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
df = df[["Close"]]


scaler = MinMaxScaler()
df["Close"] = scaler.fit_transform(df[["Close"]])


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

SEQ_LEN = 60
data = df.values
X, y = create_sequences(data, SEQ_LEN)

X = torch.tensor(X).float()
y = torch.tensor(y).float()


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(20):
    model.train()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.6f}")


model.eval()
predicted = model(X_test).detach().numpy()
actual = y_test.numpy()


predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(actual)

test_dates = df.index[SEQ_LEN + train_size:]

start_date_str = input("Başlangıç tarihi (YYYY-MM-DD): ")
end_date_str = input("Bitiş tarihi (YYYY-MM-DD): ")
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")


user_pred = float(input("Kendi tahmininizi girin (örnek: 180.0): "))


mask = (test_dates >= start_date) & (test_dates <= end_date)
filtered_dates = test_dates[mask]
filtered_actual = actual[mask]
filtered_predicted = predicted[mask]


plt.figure(figsize=(12,6))
plt.plot(filtered_dates, filtered_actual, label="Gerçek", color="blue")
plt.plot(filtered_dates, filtered_predicted, label="Model Tahmini", color="orange")
plt.axhline(y=user_pred, color='green', linestyle='--', label=f"Kendi Tahminin ({user_pred})")
plt.legend()
plt.title(f"AAPL Tahmini: {start_date_str} - {end_date_str}")
plt.xlabel("Tarih")
plt.ylabel("Fiyat")
plt.grid(True)
plt.show()
