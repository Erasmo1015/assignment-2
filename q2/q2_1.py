from dataclasses import dataclass
import copy
import random

@dataclass
class Bird:
    x: float
    y: float
    vx: float
    vy: float
    w: float = 20
    h: float = 20

@dataclass
class Pipe:
    x: float
    h: float
    w: float = 70
    gap: float = 200

SCREEN_WIDTH = 800  # Define your screen width here

# Define PID controller
@dataclass
class PIDController:
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.1
    error_accumulator: float = 0
    prev_error: float = 0

    def calc_input(self, sp: float, pv: float, umin: float = -100, umax: float = 100) -> float:
        """Calculate the control signal.
        sp: Set point
        pv: Process variable
        """
        e = sp - pv
        P = self.Kp * e

        self.error_accumulator += e
        I = self.Ki * self.error_accumulator

        D = self.Kd * (e - self.prev_error)
        self.prev_error = e
        
        pid = P + I + D
        if pid < umin:
            return umin
        elif pid > umax:
            return umax
        else:
            return pid 

# Initialize bird and pipe
bird = Bird(50, 300, 30, 0)
pipe_height = random.randint(200, 300)
pipe = Pipe(SCREEN_WIDTH - 50, pipe_height)

# Define bird motion function
def bird_motion(bird: Bird, u: float, dt: float, gravity: float = -50) -> Bird:
    """Updates the bird's y position and velocity."""
    new_bird = copy.deepcopy(bird)
    new_bird.y = bird.y + bird.vy * dt
    new_bird.vy = bird.vy + (u + gravity) * dt
    return new_bird

# Define pipe motion function
def pipe_motion(pipe: Pipe, vx: float, dt: float) -> (Pipe, int):
    """Updates the pipe"""
    new_pipe = copy.deepcopy(pipe)
    new_pipe.x -= vx * dt

    d_score = 0
    if new_pipe.x < -pipe.w:
        new_pipe.x = SCREEN_WIDTH
        new_pipe.h = random.randint(200, 300)
        d_score = 1
    return new_pipe, d_score

# Implement the control signal calculation
def calculate_the_control_signal(bird: Bird, pipe: Pipe, k: int):
    """Calculate the control signal for the bird using PID controller."""
    sp = pipe.h + pipe.gap / 2
    pv = bird.y + bird.h / 2

    # Initialize PID controller
    controller = PIDController(Kp=0.5, Ki=0.05, Kd=0.1)
    
    # Calculate control signal
    u_jump = controller.calc_input(sp, pv)
    
    return u_jump
