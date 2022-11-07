    import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# Simulation time in seconds
SIM_TIME = 60

from scipy.spatial.transform import Rotation as Rot
import math
def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle
    Parameters
    ----------
    angle :
    Returns
    -------
    A 2D rotation matrix
    Examples
    --------
    >>> angle_mod(-4.0)
    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def plot_covariance_ellipse(xEst, PEst):
    """Function to plot covariance ellipse"""
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    fx = rot_mat_2d(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0]).flatten()
    py = np.array(fx[1, :] + xEst[1]).flatten()
    plt.plot(px, py, "--r")


class Robot:
    def __init__(self,dT_pred,W,V) -> None:
        """ State[4]
            x,
            y,
            yaw,
            vel,
        """
        self.state_mean = np.zeros(4) # predicted state
        self.state_true = np.zeros(4) # true state for reference
        self.state_deadreck = np.zeros(4) # dead reckoing for reference
        
        # Time between two consecutive predictions
        self.dT_pred = dT_pred

        """ Model and EKF Matrices """
        self.cov = np.eye(4)

        # df/dv (motion,noise)
        self.Q = np.eye(4)

        # noise params for model
        self.W = W
        self.Q_ = self.Q @ self.W @ self.Q.T
        # self.Q_ = np.zeros(shape=(4,4))

        # dh/dx (observation,state)
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # dh/dv (observation,noise)
        self.R = np.eye(2)

        # noise params for gps measurements
        self.V = V
        self.R_ = self.R @ self.V @ self.R.T


    def motion_model(self,state,acceleration,yaw_rate,dT) -> np.array:
        """ Describes the model of motion for the robot
            x = f(x,u,w)
        """
        # TODO: add noise
        x,y,yaw,vel = state
        
        # Equations of motion
        x = x + np.cos(yaw) * vel * dT
        y = y + np.sin(yaw) * vel * dT
        yaw = yaw + yaw_rate * dT
        vel = vel + acceleration * dT
        
        return np.array([x,y,yaw,vel])
    
    def observation_model(self,state,noise) -> np.array:
        """ Outputs the expected measurement for a given state of the robot
            z = h(x,v)
        """
        # TODO: add noise
        return self.C @ state #+noise

    def jac_F(self,state):
        """ returns the jacobian F of the motion model, evaluated at x
            df/dx(x) (motion,state)
        """
        x,y,yaw,vel = state
        F = np.array([
            [ 1, 0, -np.sin(yaw)*vel*self.dT_pred,  np.cos(yaw) * self.dT_pred],
            [ 0, 1,  np.cos(yaw)*vel*self.dT_pred,  np.sin(yaw) * self.dT_pred],
            [ 0, 0,                     1,                  0                 ],
            [ 0, 0,                     0,                  1                 ]
        ])
        return F

    def EKF_predict(self,state,acceleration,yaw_rate):
        x = self.motion_model(state,acceleration,yaw_rate,self.dT_pred)
        F = self.jac_F(x)

        cov = F @ self.cov @ F.T + self.Q_
        return x, cov


    def EKF_update(self,x_prior,cov_prior,z_measure):
        mt = self.observation_model(x_prior,noise=None)
        S = self.C @ self.cov @ self.C.T + self.R_
        D = self.cov @ self.C.T
        
        # Kalman gain
        K = D @ np.linalg.inv(S)

        # Kalman update
        x = x_prior + K @ (z_measure - mt)
        cov = cov_prior - K @ S @ K.T
        return x,cov



if __name__=="__main__":
    imu_update_freq = 100 #Hz
    gps_update_freq = 1   #Hz

    # Generate true input trajectories
    acceleration_traj = 0.1*np.sin(0.1*np.arange(start=0,stop=SIM_TIME,step=1/imu_update_freq)) + 0.1
    yaw_rate_traj = np.ones(SIM_TIME*imu_update_freq)*2*np.pi/SIM_TIME

    # Simulate noise
    gps_variance = 3 #standardavvik
    acc_variance = 0.2 #m/s^2
    gyro_variance_degrees = 10
    gyro_variance_rad = gyro_variance_degrees / 180 * np.pi
    noise_acc_signal = np.random.normal(loc=0,scale=np.sqrt(acc_variance),size=SIM_TIME*imu_update_freq)
    noise_yaw_rate_signal = np.random.normal(loc=0,scale=np.sqrt(gyro_variance_rad),size=SIM_TIME*imu_update_freq)

    V = np.array([
            [gps_variance,         0],
            [0,         gps_variance]
        ]) * 1/gps_update_freq
    W = np.array([
            [0, 0,          0, 0],
            [0, 0,          0, 0],
            [0, 0, gyro_variance_rad, 0],
            [0, 0,          0, acc_variance],
        ]) *1/imu_update_freq

    
    robot = Robot(dT_pred=1/imu_update_freq,W=W,V=V)
    states_est = robot.state_mean
    states_true = robot.state_mean
    states_deadreck = robot.state_deadreck

    time = 0
    while time < SIM_TIME:

        # Simulating EKF predictions on IMU data
        # 100Hz
        for i in range(imu_update_freq):
            # Input
            acc = acceleration_traj[time*imu_update_freq + i]
            yaw_rate = yaw_rate_traj[time*imu_update_freq + i]

            # Noise
            noise_acc = noise_acc_signal[time*imu_update_freq + i]
            noise_yaw_rate = noise_yaw_rate_signal[time*imu_update_freq + i]

            # Run EKF prediction with input and noise
            state,cov = robot.EKF_predict(robot.state_mean,acceleration=acc+noise_acc,yaw_rate=yaw_rate+noise_yaw_rate)
            robot.state_mean = state
            robot.cov = cov
            print(cov)

            # Also run for true state and dead reckoning for references
            robot.state_true = robot.motion_model(robot.state_true,acc,yaw_rate,dT=robot.dT_pred)
            robot.state_deadreck = robot.motion_model(robot.state_deadreck,acc+noise_acc,yaw_rate+noise_yaw_rate,dT=robot.dT_pred)
            states_est = np.vstack((states_est,robot.state_mean))


        # Simulate GPS 1Hz measurement by adding gaussian noise to true position
        gps_measurement = robot.state_true[:2] + np.random.normal(loc=0,scale=np.sqrt(gps_variance),size=2)

        # Executing EKF Update, updating upon GPS measurement
        # 1 Hz
        state,cov = robot.EKF_update(x_prior=robot.state_mean,cov_prior=cov,z_measure=gps_measurement)
        robot.state_mean = state
        robot.cov = cov
        
        # Saving true and predicted states
        states_true = np.vstack((states_true,robot.state_true))
        states_est = np.vstack((states_est,robot.state_mean))
        states_deadreck = np.vstack((states_deadreck,robot.state_deadreck))


        time += 1

        # Stopping simulation with esc key
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        plt.plot(states_est[:,0],states_est[:,1],"-r")
        plt.plot(states_true[:,0],states_true[:,1],"-b")
        plt.plot(states_deadreck[:,0],states_deadreck[:,1],"-k")
        plt.plot(gps_measurement[0],gps_measurement[1],"dg")
        plot_covariance_ellipse(robot.state_mean,robot.cov)
        # plt.pause(0.1)


    plt.show()






