
import gym
import time
import numpy as np
import tensorflow as tf
import time
from gym import error, utils, spaces
from gym.utils import seeding
from gym.envs.registration import register
import CoDrone


#register the training environment in the gym as an available one
reg = register(
    id='CodroneEnv-v0',
    entry_point='codrone_env:CodroneEnv',
    timestep_limit=100,
    )

class CodroneEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.drone = CoDrone.CoDrone()
        self.drone_pair = drone.pair()
        self.state = drone.getState()
        self.takeoff = drone.takeoff()
        self.hover = drone.hover(3)
        
        # gets training parameters from sensors
        self.throttle = drone.getThrottle()
        self.horizontal_position = drone.getOptFlowPosition()
        self.horizontal_position.X = drone.getOptFlowPosition(position.X)
        self.horizontal_position.Y = drone.getOptFlowPosition(position.Y)
        self.height = drone.getHeight()
        self.desired_position.X = X
        self.desired_position.Y = Y
        self.desired_height = Z
        self.gyro_angles = drone.getGyroAngles()
        self.angular_speed = drone.getAngularSpeed()
        self.accelerometer = drone.getAccelerometer()
        self.pressure = drone.getPressure()
        self.trim = drone.getTrim()
        self.battery_state = drone.getBatteryPercentage()
        self.drone_temp = drone.getDroneTemp()
        self.observation_space = [drone.getOptFlowPosition(),drone.getHeight()]
        self.max_altitude = 2000
        
        self.action_space = spaces.Discrete(6) 
        #FORWARD, BACKWARD, LEFT, RIGHT, UP, and DOWN
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        
        if self.drone.isReadyToFly() == True and self.drone.isFlying() == False:
            self.throttle = None
            self.horizontal_position = None
            self.horizontal_position.X = None
            self.horizontal_position.Y = None
            self.height = None
            self.gyro_angles = None
            self.angular_speed = None
            self.accelerometer = None

        # 4th: takes an observation of the initial condition of the robot
        observation = [self.take_observation]
        return observation

    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        
        # 1st, we decide if the drone is flying or not FORWARD, BACKWARD, LEFT, RIGHT, UP, and DOWN
        if self.drone.connect == True:
            drone.takeoff()
            drone.hover()
        else:
            if action == 0: #FORWARD
                drone.go(Direction.FORWARD)
            elif action == 1: #BACKWARD
                drone.go(Direction.BACKWARD)
            elif action == 2: #LEFT
                drone.go(Direction.LEFT)
            elif action == 3: #RIGHT
                drone.go(Direction.RIGHT)
            elif action == 4: #UP
                drone.go(Direction.UP)
            elif action == 5: #DOWN
                drone.go(Direction.DOWN)
            

        # Then we send the command to the robot and let it go
        # for running_step seconds
        #horizontal_position, height, gyro_angles = self.take_observation()
        
        # finally we get an evaluation based on what happened in the sim
        #reward,done = self.process_data(horizontal_position, height, gyro_angles)

        #state = [height]
        #return state, reward, done, {}

    def take_observation (self):
        return horizontal_position, height, gyro_angles
    
    
    def process_data(self, horizontal_position, height, gyro_angles):

        done = False
        roll = GyroAngles.ROLL
        pitch = GyroAngles.PITCH
        yaw = GyroAngles.YAW
        
        
        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = height > max_altitude

        if altitude_bad or pitch_bad or roll_bad:
            print("Drone flight status is wrong")
            done = True
            reward = -200
        else:
            reward = self.improved_Z_reward(height)

        return reward,done

    
    def init_desired_pose(self):
        
        horizontal_position, height = self.take_observation()
        
        self.best_dist = self.calculate_dist_between_two_Points([horizontal_position.X, horizontal_position.Y, height],
                                                                [self.desired_position.X,self.desired_position.Y, self.desired_height])
 

    def improved_X_reward(self, horizontal_position.X):
        current_dist_X = self.calculate_dist_between_two_Points(horizontal_position.X, self.desired_horizontal_position.X)
        print("Calculated Distance = "+str(current_dist))
        
        if height < self.desired_horizontal_position:
            reward = 10
            self.best_dist = current_dist_X
        elif current_dist_X == self.best_dist:
            reward = 0
        else:
            reward = -10
            print("Made X Distance bigger= "+str(self.best_dist))
        
        return reward        
     
    def improved_Y_reward(self, horizontal_position):
        current_dist_Y = self.calculate_dist_between_two_Points(horizontal_position.Y, self.desired_horizontal_position.Y)
        print("Calculated Distance = "+str(current_dist))
        
        if height < self.desired_horizontal_position.Y:
            reward = 10
            self.best_dist = current_dist_Y
        elif current_dist_Y == self.best_dist:
            reward = 0
        else:
            reward = -10
            print("Made Y Distance bigger= "+str(self.best_dist))
        
        return reward 
    
    def improved_Z_reward(self, height):
        current_dist_Z = self.calculate_dist_between_two_Points(height, self.desired_height)
        print("Calculated Distance = "+str(current_dist))
        
        if height < self.desired_height:
            reward = 100
            self.best_dist = current_dist_Z
        elif current_dist_Z == self.best_dist:
            reward = 0
        else:
            reward = -100
            print("Made Z Distance bigger= "+str(self.best_dist))
        
        return reward 
 
    
    def calculate_dist_between_two_Points(self,p_init,p_end):
        a = np.array((p_init.horizontal_position.X ,p_init.horizontal_position.Y, p_init.height))
        b = np.array((p_end.desired_position.X ,p_end.desired_position.Y, p_end.desired_height))
        
        dist = np.linalg.norm(a-b)
        
        return dist


    def takeoff_sequence(self):
        # not sure if usefull
        drone.takeoff()
        drone.hover(3)
        print("Taking-Off sequence completed")

