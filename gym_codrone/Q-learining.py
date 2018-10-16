
# coding: utf-8

# In[ ]:


import time
import numpy
import random
import time
from gym import wrappers



if __name__ == '__main__':
    
    #drone = CoDrone.CoDrone(True, False, False, False, False)
    #drone.pair()
    #time.sleep(30)
    #if drone.getState() = True:
    #    break
    #else:
    #    time.sleep(30)
    # Create the Gym environment
    #env = gym.make('CodroneEnv-v0')
    #rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    #rospack = rospkg.RosPack()
    #pkg_path = 'c:'
    #outdir = pkg_path + '\training_results'
    #env = wrappers.Monitor(env, outdir, force=True) 
    #print( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    #Alpha = rospy.get_param("/alpha")
    #epsilon = rospy.get_param("/epsilon")
    #Gamma = rospy.get_param("/gamma")
    #epsilon_discount = rospy.get_param("/epsilon_discount")
    #nepisodes = rospy.get_param("/nepisodes")
    #nsteps = rospy.get_param("/nsteps")
    #import param_list
    
    # Initialises the algorithm that we are going to use for learning
    qlearn = QLearn(actions=range(env.action_space.n), epsilon = epsilon, alpha = alpha, gamma = gamma)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        print("STARTING Episode #"+str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))
        
        # Show on screen the actual situation of the robot
        #env.render()
        
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                print("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break 

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print(("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    
    print( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

