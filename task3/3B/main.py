import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import sleep

#shamelessly stolen from the internet
def fit(num_epochs, model, loss_fn, opt, inputs, targets):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in zip(inputs, targets):
            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()


#get data
sample_data = pd.read_csv("C:\\Users\\Casper\\PycharmProjects\\DM\\task3\\3B\\sample.csv")
sample_data = sample_data.drop(["Gender"], axis=1)
sample_np = sample_data.to_numpy(dtype='float32')
height = sample_np[:,0]
weight = sample_np[:,1]

#linear regression using torch
inputs = torch.from_numpy(weight)
outputs = torch.from_numpy(height)

inputs = inputs.reshape(-1, 1)
outputs = outputs.reshape(-1, 1)

model1 = nn.Linear(1, 1)
model2 = nn.Linear(1, 1)

loss_fn1 = F.mse_loss
loss_fn2 = F.l1_loss

opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)


fit(300, model1, loss_fn1, opt1, inputs/100, outputs/100)
fit(300, model2, loss_fn2, opt2, inputs/100, outputs/100)

preds1 = model1(inputs/100)*100
preds2 = model2(inputs/100)*100

#show plot
plt.scatter(inputs, outputs, label="Data", c="Blue")
plt.xlabel("Weight (lb.)")
plt.ylabel("Height (inch)")

plt.plot(inputs, preds1.detach().numpy(), c="red", label="MSE")
plt.plot(inputs, preds2.detach().numpy(), c="orange", label="MAE")
plt.title("MSE vs MAE")
plt.legend()
#plt.show()



errors_MSE = np.absolute(preds1.detach().numpy()-outputs.detach().numpy())
errors_MAE = np.absolute(preds2.detach().numpy()-outputs.detach().numpy())

print("MAE measure, MSE loss: ", np.mean(errors_MSE))
print("MAE measure, MAE loss: ", np.mean(errors_MAE))

print("errors_MSE", errors_MSE.round().tolist())
print("errors_MAE", errors_MAE.round().tolist())

print("MSE measure, MSE loss: ", np.mean(np.square(errors_MSE)))
print("MSE measure, MAE loss: ", np.mean(np.square(errors_MAE)))