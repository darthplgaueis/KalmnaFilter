import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# Load data
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    initial = np.fromstring(lines[0], sep=',')
    data = np.array([np.fromstring(line.strip(), sep=',') for line in lines[1:]])
    return initial, data

# Basic linear Kalman Filter
def run_kf(initial, data):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State: [x, y, vx, vy]
    dt = 1  # Assuming fixed timestep
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.x = np.array([initial[0], initial[1], 0, 0])

    kf.P *= 100  # Large initial uncertainty
    kf.R = np.eye(2) * 5  # Measurement noise
    kf.Q = np.eye(4) * 0.1  # Process noise

    predictions = []
    for z in data:
        gps_x, gps_y, vel_x, vel_y = z
        kf.predict()
        kf.update([gps_x, gps_y])
        predictions.append((kf.x[:2].copy(), kf.P[:2, :2].copy()))
        # optionally update velocity
        kf.x[2] = vel_x
        kf.x[3] = vel_y

    return predictions

# Extended Kalman Filter
def hx(x):
    return np.array([x[0], x[1]])

def H_jacobian(x):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

def run_ekf(initial, data):
    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    dt = 1
    ekf.F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    ekf.x = np.array([initial[0], initial[1], 0, 0])
    ekf.P *= 100
    ekf.R = np.eye(2) * 5
    ekf.Q = np.eye(4) * 0.1

    predictions = []
    for z in data:
        gps_x, gps_y, vel_x, vel_y = z
        ekf.predict()
        ekf.update([gps_x, gps_y], H_jacobian, hx)
        predictions.append((ekf.x[:2].copy(), ekf.P[:2, :2].copy()))
        ekf.x[2] = vel_x
        ekf.x[3] = vel_y
    return predictions

# Unscented Kalman Filter
def fx(x, dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    return np.dot(F, x)

def run_ukf(initial, data):
    dt = 1
    points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=1)
    ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)

    ukf.x = np.array([initial[0], initial[1], 0, 0])
    ukf.P *= 100
    ukf.R = np.eye(2) * 5
    ukf.Q = np.eye(4) * 0.1

    predictions = []
    for z in data:
        gps_x, gps_y, vel_x, vel_y = z
        ukf.predict()
        ukf.update([gps_x, gps_y])
        predictions.append((ukf.x[:2].copy(), ukf.P[:2, :2].copy()))
        ukf.x[2] = vel_x
        ukf.x[3] = vel_y
    return predictions

# Visualization
def visualize(data, predictions, label):
    measured = data[:, :2]
    pred = np.array([p[0] for p in predictions])
    uncertainties = np.array([np.trace(p[1]) for p in predictions])

    plt.plot(measured[:, 0], measured[:, 1], 'r.', label='GPS')
    plt.plot(pred[:, 0], pred[:, 1], '-', label=label)
    plt.title(label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(uncertainties, label='Uncertainty')
    plt.title(f"Uncertainty - {label}")
    plt.xlabel("Time step")
    plt.ylabel("Trace of Covariance")
    plt.legend()
    plt.grid()
    plt.show()

# Error analysis: RMSE and MAE
def compute_errors(predictions, data):
    pred_pos = np.array([p[0] for p in predictions])
    gps_pos = data[:, :2]

    errors = pred_pos - gps_pos
    mse = np.mean(errors**2, axis=0)
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(mse)

    uncertainty_trace = [np.trace(p[1]) for p in predictions]

    return rmse, mae, errors, uncertainty_trace

def plot_errors(errors_dict):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    for method, values in errors_dict.items():
        rmse, mae, errors, trace = values
        axs[0].plot([e[0] for e in errors], label=f"{method}")
        axs[1].plot([e[1] for e in errors], label=f"{method}")
        axs[2].plot(trace, label=f"{method}")

    axs[0].set_title("Prediction Error in X")
    axs[1].set_title("Prediction Error in Y")
    axs[2].set_title("Trace of Covariance (Uncertainty)")

    for ax in axs:
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Error / Trace")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

# Use this after all filters have run
def evaluate_all_filters(initial, data):
    errors_dict = {}

    for name, runner in zip(["KF", "EKF", "UKF"], [run_kf, run_ekf, run_ukf]):
        preds = runner(initial, data)
        rmse, mae, errors, trace = compute_errors(preds, data)
        errors_dict[name] = (rmse, mae, errors, trace)
        print(f"\n{name} RMSE: {rmse}, MAE: {mae}")

    plot_errors(errors_dict)

# Main
if __name__ == "__main__":
    initial, data = load_data("input.txt")
    for method, runner in zip(["KF", "EKF", "UKF"], [run_kf, run_ekf, run_ukf]):
        preds = runner(initial, data)
        visualize(data, preds, method)
    evaluate_all_filters(initial, data)
