import torch
import torchvision.utils as vutils
from models import Generator

print("Loading Generator model...")
# Load the trained Generator
netG = Generator()
netG.load_state_dict(torch.load('generator.pth'))
netG.eval()
print("Generator model loaded.")

print("Generating images...")
# Generate images
with torch.no_grad():
    fixed_noise = torch.randn(64, 100, 1, 1)  # Correct shape for Conv-based Generator
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(), 'generated_images.png', normalize=True)
print("Images generated and saved as 'generated_images.png'.")
