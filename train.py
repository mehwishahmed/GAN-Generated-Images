import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import Generator, Discriminator
import time
import os
from torchvision.utils import save_image

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Initialize networks and move them to the appropriate device
netG = Generator().to(device)
netD = Discriminator().to(device)

# Define loss function and optimizers
criterion = nn.BCELoss().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

def main():
    num_epochs = 50  # Start with 50 epochs
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)  # For monitoring progress
    print("Starting training...")

    for epoch in range(num_epochs):
        start_time = time.time()
        for i, (data, _) in enumerate(trainloader):
            real = data.to(device)
            b_size = real.size(0)

            # Update Discriminator
            netD.zero_grad()
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Print batch progress
            if i % 100 == 0:
                print(f"Processing batch {i+1} of epoch {epoch+1}")

        epoch_duration = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds - Loss D: {errD_real.item() + errD_fake.item()}, Loss G: {errG.item()}')

        # Save intermediate images for monitoring progress
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
                save_image(fake_images, f"generated_images/epoch_{epoch+1}.png", normalize=True)
                print(f"Saved generated images for epoch {epoch+1}")

    # Save models
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')
    print("Training finished and models saved.")

if __name__ == '__main__':
    main()
