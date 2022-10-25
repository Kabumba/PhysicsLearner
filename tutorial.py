import math
from torch.utils.data import DataLoader
from kickoff_dataset import KickoffDataset

dataset = KickoffDataset("[SmoothSteps]edf30761-0557-481b-90fe-d46df739f8ee_10.pbz2")
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1)%5 ==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
