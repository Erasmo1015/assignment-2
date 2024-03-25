import numpy as np
import cvxpy as cp
import pygame
import random

# Initialize Pygame
pygame.init()

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define classes
class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.size = 20

    def flap(self):
        self.velocity = -8  # Increase the velocity when flapping

    def update(self):
        self.velocity += 0.5  # Add gravity
        self.y += self.velocity

class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap = 200  # Size of the gap between pipes
        self.width = 70
        self.speed = 5
        self.top_height = random.randint(50, SCREEN_HEIGHT - self.gap - 50)

    def move(self):
        self.x -= self.speed

    def off_screen(self):
        return self.x < -self.width

    def collision(self, bird):
        if bird.y < 0 or bird.y > SCREEN_HEIGHT:
            return True
        if self.x < bird.x < self.x + self.width:
            if not (self.top_height < bird.y < self.top_height + self.gap):
                return True
        return False

# Create Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Initialize game variables
bird = Bird()
pipes = [Pipe(SCREEN_WIDTH + i * 300) for i in range(3)]
clock = pygame.time.Clock()
score = 0

# Define model predictive control
horizon = 10
u = cp.Variable(horizon)  # Control signal
obj = cp.Minimize(cp.sum_squares(u))  # Minimize the sum of squared control signal
constraints = [u >= -1, u <= 1]  # Control signal limits

# Main game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            bird.flap()

    # Game logic
    bird.update()

    # Check for collisions
    for pipe in pipes:
        if pipe.collision(bird):
            running = False

    # Move pipes
    for pipe in pipes:
        pipe.move()

    # Add new pipe when one goes off screen
    if pipes[0].off_screen():
        pipes.pop(0)
        pipes.append(Pipe(SCREEN_WIDTH))

    # Check for score
    if pipes[0].x < bird.x and not pipes[0].off_screen():
        score += 1

    # Solve the optimization problem for control signal
    problem = cp.Problem(obj, constraints)
    try:
        problem.solve()
        u_jump = u[0].value
    except cp.error.SolverError:
        u_jump = 0

    # Rendering
    screen.fill(WHITE)

    # Draw bird
    pygame.draw.rect(screen, BLACK, (bird.x, bird.y, bird.size, bird.size))

    # Draw pipes
    for pipe in pipes:
        pygame.draw.rect(screen, BLACK, (pipe.x, 0, pipe.width, pipe.top_height))
        pygame.draw.rect(screen, BLACK, (pipe.x, pipe.top_height + pipe.gap, pipe.width, SCREEN_HEIGHT - pipe.top_height - pipe.gap))

    # Display score
    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f'Score: {score}', True, BLACK)
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
