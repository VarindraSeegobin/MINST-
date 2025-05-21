import pygame
import numpy as np
import torch
import torch.nn.functional as F
from model import MinstNet


pygame.init()


WIDTH, HEIGHT = 280, 280
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a Digit (0–9)")


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
screen.fill(BLACK)
drawing = False
radius = 8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinstNet().to(device)
model.load_state_dict(torch.load("minstnet_final.pth", map_location=device))
model.eval()


def predict_digit(surface):
    data = pygame.surfarray.array3d(surface)  # shape (width, height, 3)
    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    data = np.transpose(data)  # Flip axes to get (H, W)
    data = pygame.transform.scale(pygame.surfarray.make_surface(data), (28, 28))
    data = pygame.surfarray.array3d(data)
    data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])
    data = data.astype(np.float32) / 255.0
    tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, 28, 28]

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()
    return prediction


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pred = predict_digit(screen)
                print(f"Predicted Digit: {pred}")
            elif event.key == pygame.K_c:
                screen.fill(BLACK)

    if drawing:
        pos = pygame.mouse.get_pos()
        pygame.draw.circle(screen, WHITE, pos, radius)

    pygame.display.flip()

pygame.quit()
