
import gym
import time
import numpy as np
import tensorflow as tf
import time
from gym import error, utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
import CoDrone
drone = CoDrone.CoDrone()

#register the training environment in the gym as an available one
reg = register(
    id='Codrone-v0',
    entry_point='gym.envs.codrone_env:CodroneEnv',
    timestep_limit=100,
    )

X = 0
Y = 0
Z = 1500


class CoDroneEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.drone = CoDrone.CoDrone()
        self.drone_pair = drone.pair()
        #self.dronestate = drone.getState()
        #self.takeoff = drone.takeoff()
        #self.hover = drone.hover(3)
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        gyro_angles = drone.getGyroAngles()
        
        self.horizontal_position = drone.getOptFlowPosition()
        self.horizontal_position_X = horizontal_position.X
        self.horizontal_position_Y = horizontal_position.Y
        print(horizontal_position_X, horizontal_position_Y, height)
        #X = 0
        #Y = 0
        #Z = 2000
        # gets training parameters from sensors
        self.throttle = drone.getThrottle()
        #self.horizontal_position = drone.getOptFlowPosition()
        #self.horizontal_position_X = horizontal_position.X
        #self.horizontal_position_Y = print(horizontal_position_Y)
        self.height = drone.getHeight()
        self.desired_position_X = X
        self.desired_position_Y = Y
        self.desired_height = Z
        self.gyro_angles = drone.getGyroAngles()
        self.angular_speed = drone.getAngularSpeed()
        self.accelerometer = drone.getAccelerometer()
        self.pressure = drone.getPressure()
        self.trim = drone.getTrim()
        self.battery_state = drone.getBatteryPercentage()
        self.drone_temp = drone.getDroneTemp()
        self.observation_space = [horizontal_position_X, horizontal_position_Y, height]
        self.max_altitude = 3000
        self.max_incl = 25
        self.running_step = 1
        
        self.action_space = spaces.Discrete(6) 
        #FORWARD, BACKWARD, LEFT, RIGHT, UP, and DOWN
        self.reward_range = (-np.inf, np.inf)

        self.seed()
        self.state = None
        self.steps_beyond_done = None

    # A function to initialize the random generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        
        reset_position = drone.emergencyStop()
                
        # 4th: takes an observation of the initial condition of the robot
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        
        #gyro_angles = drone.getGyroAngles()
        #observation = [self.take_observation]
        #return observation

        self.state = (horizontal_position_X, horizontal_position_Y, height)
        self.steps_beyond_done = None
        return np.array(self.state)
        print("drone reset OK")
    
    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        horizontal_position_X, horizontal_position_Y, height = state
        
        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        
        # 1st, we decide if the drone is flying or not FORWARD, BACKWARD, LEFT, RIGHT, UP, and DOWN
        if drone.getState()!= 'FLIGHT':
            drone.takeoff()
            drone.hover()
        else:
            if action == 0: #FORWARD
                drone.go(1, 3)
            elif action == 1: #BACKWARD
                   drone.go(2, 3)
            elif action == 2: #LEFT
                drone.go(3, 3)
            elif action == 3: #RIGHT
                drone.go(4, 3)
            elif action == 4: #UP
                drone.go(5, 3)
            elif action == 5: #DOWN
                drone.go(6, 3)
        
        #go(vel_cmd)
        time.sleep(self.running_step)
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        #horizontal_position_X, horizontal_position_Y, height = self.take_observation()
        reward, done = self.process_data(horizontal_position_X, horizontal_position_Y, height)
        
        max_dist_allowed = 300
        test_upside = drone.isUpsideDown()
        test_accident = drone.getState()
                
        done = horizontal_position_X > max_dist_allowed \
            or horizontal_position_X < -max_dist_allowed \
            or horizontal_position_Y > max_dist_allowed \
            or horizontal_position_Y < -max_dist_allowed \
            or height < (height - max_dist_allowed)
            # or test_upside == True \
            # or test_accident == 'ACCIDENT':
        done = bool(done)
        
        if not done:
            # Promote going upwards instead of turning
            if action == 4:
                reward += 1
            elif action == 1 or action == 5:
                reward -= 5
            elif action == 0 or action == 2 or action == 3:
                reward -= 1
        
        self.state = (horizontal_position_X, horizontal_position_Y, height)
        #state = [horizontal_position_X, horizontal_position_Y, height, gyro_angles]
        return np.array(self.state), reward, done, {}
        #self.horizontal_position = drone.getOptFlowPosition()
        
       
    def take_observation (self):
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        return horizontal_position_X, horizontal_position_Y, height
    
    
    def process_data(self, horizontal_position_X, horizontal_position_Y, height):

        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        done = False
        roll = drone.getRoll()
        pitch = drone.getPitch()
        yaw = drone.getYaw()
        
        
        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = height > self.max_altitude

        if altitude_bad or pitch_bad or roll_bad:
            print("Drone flight status is wrong")
            done = True
            reward = -20
        else:
            reward = self.improved_X_reward(horizontal_position_X) + self.improved_Y_reward(horizontal_position_Y) + self.improved_Z_reward(height)

        print("Reward calculated = "+str(reward))
        return reward,done

    
    def init_desired_pose(self):
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        horizontal_position_X, horizontal_position_Y, height = self.take_observation()
        
        self.best_dist_X = self.calculate_dist_X(horizontal_position_X)
        self.best_dist_Y = self.calculate_dist_Y(horizontal_position_Y)
        self.best_dist_Z = self.calculate_dist_Z(height)
 

    def improved_X_reward(self, horizontal_position_X):
        
        self.best_dist_X = self.calculate_dist_X(horizontal_position_X)
                
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        current_dist_X = self.calculate_dist_X(horizontal_position_X)
        print("Calculated Distance X = "+str(current_dist_X))
        
        if horizontal_position_X < self.desired_position_X:
            reward = 10
            self.best_dist_X = current_dist_X
        elif current_dist_X == self.best_dist_X:
            reward = 0
        else:
            reward = -10
            print("Made X Distance bigger= "+str(self.best_dist_X))
        
        return reward        
     
    def improved_Y_reward(self, horizontal_position_Y):
        
        self.best_dist_Y = self.calculate_dist_Y(horizontal_position_Y)
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        current_dist_Y = self.calculate_dist_Y(horizontal_position_Y)
        print("Calculated Distance Y = "+str(current_dist_Y))
        
        if horizontal_position_Y < self.desired_position_Y:
            reward = 10
            self.best_dist_Y = current_dist_Y
        elif current_dist_Y == self.best_dist_Y:
            reward = 0
        else:
            reward = -10
            print("Made Y Distance bigger= "+str(self.best_dist_Y))
        
        return reward 
    
    def improved_Z_reward(self, height):
        
        self.best_dist_Z = self.calculate_dist_Z(height)
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
        horizontal_position_Y = horizontal_position.Y
        height = drone.getHeight()
        #gyro_angles = drone.getGyroAngles()
        
        current_dist_Z = self.calculate_dist_Z(height)
        print("Calculated Distance Z = "+str(current_dist_Z))
        
        if height < self.desired_height:
            reward = 100
            self.best_dist_Z = current_dist_Z
        elif current_dist_Z == self.best_dist_Z:
            reward = 0
        else:
            reward = -100
            print("Made Z Distance bigger= "+str(self.best_dist_Z))
        
        return reward 
 
    
    #def calculate_dist_between_two_Points(self,p_init,p_end):
    def calculate_dist_X(self, horizontal_position_X):
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_X = horizontal_position.X
       
        desired_position_X = X
        
        a = horizontal_position_X
        b = desired_position_X
        
        dist_X = np.linalg.norm(a-b)
        
        return dist_X
    
    def calculate_dist_Y(self, horizontal_position_Y):
        
        horizontal_position = drone.getOptFlowPosition()
        horizontal_position_Y = horizontal_position.Y
       
        desired_position_Y = Y
        
        c = horizontal_position_Y
        d = desired_position_Y
        
        dist_Y = np.linalg.norm(c-d)
        
        return dist_Y

    def calculate_dist_Z(self, height):
        
        height = drone.getHeight()

        desired_height = Z
        
        e = height
        f = desired_height
        
        dist_Z = np.linalg.norm(e-f)
        
        return dist_Z
    
    def takeoff_sequence(self):
        # not sure if usefull
        drone.takeoff()
        drone.hover(3)
        print("Taking-Off sequence completed")
