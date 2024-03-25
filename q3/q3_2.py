import numpy as np
import pandas as pd
import cv2

# Load fingertip tracking data
data = pd.read_csv('txys_missingdata.csv')

# Number of particles
num_particles = 100

# Initialize particles randomly
particles = np.random.rand(num_particles, 4)  # Each particle is [x, vx, y, vy]

# Initialize variables for estimated trajectory
trajectory = []

# Particle filtering loop
for i in range(len(data)):
    # Motion model: Assume constant velocity
    particles[:, 0] += particles[:, 1]  # Update x position
    particles[:, 2] += particles[:, 3]  # Update y position
    
    # Weight update: Calculate particle weights based on distance from measurement
    particle_distances = np.sqrt((particles[:, 0] - data.iloc[i]['x_px'])**2 + (particles[:, 2] - data.iloc[i]['y_px'])**2)
    weights = 1 / (1 + particle_distances)  # Use inverse distance as weights
    
    # Normalize weights
    weights /= np.sum(weights)
    
    # Resampling: Sample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, replace=True, p=weights)
    particles = particles[indices]
    
    # Save estimated position for visualization (use particle with highest weight)
    estimated_position = particles[np.argmax(weights), :2]
    trajectory.append((int(estimated_position[0]), int(estimated_position[1])))

# Initialize video writer
out = cv2.VideoWriter('output_particle_filter.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

# Visualize estimated trajectory on video
cap = cv2.VideoCapture('video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Overlay estimated trajectory on frame
    for point in trajectory:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)
    
    # Resize frame if necessary
    if frame.shape[0] != frame_height or frame.shape[1] != frame_width:
        frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
