import numpy as np
import pandas as pd
import cv2

# Load fingertip tracking data
data = pd.read_csv('txys_missingdata.csv')

# Initialize Kalman filter parameters
dt = 1.0  # Time step between measurements (seconds)
A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # State transition matrix
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])  # Measurement matrix
Q = np.eye(4) * 0.01  # Process noise covariance matrix
R = np.eye(2) * 10.0  # Measurement noise covariance matrix

# Initialize state estimate and covariance matrix
x_hat = np.array([[data.iloc[0]['x_px']],
                  [0],
                  [data.iloc[0]['y_px']],
                  [0]])  # Initial state estimate (position and velocity)
P = np.eye(4) * 10.0  # Initial state covariance matrix

# Initialize variables for estimated trajectory
trajectory = []

# Kalman filter loop
for i in range(len(data)):
    # Update measurement
    z = np.array([[data.iloc[i]['x_px']],
                  [data.iloc[i]['y_px']]])
    
    # Predict step
    x_hat_minus = np.dot(A, x_hat)
    P_minus = np.dot(np.dot(A, P), A.T) + Q
    
    # Kalman gain
    K = np.dot(np.dot(P_minus, H.T), np.linalg.inv(np.dot(np.dot(H, P_minus), H.T) + R))
    
    # Update step
    x_hat = x_hat_minus + np.dot(K, (z - np.dot(H, x_hat_minus)))
    P = np.dot((np.eye(4) - np.dot(K, H)), P_minus)
    
    # Save estimated position for visualization
    trajectory.append((int(x_hat[0][0]), int(x_hat[2][0])))

# Initialize video writer
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

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
