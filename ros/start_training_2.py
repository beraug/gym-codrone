
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import gym
import time
import random
import Ros_Policy
from gym import wrappers
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt

# ROS packages required
import rospy
import rospkg

# import our training environment
import myquadcopter_env


if __name__ == '__main__':
    
    rospy.init_node('drone_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('QuadcopterLiveShow-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/alpha")
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    nsteps = rospy.get_param("/nsteps")

    # Initialises the algorithm that we are going to use for learning
    #Training the Agent
    tf.reset_default_graph() #Clear the Tensorflow graph.
    myAgent = agent(lr=1e-2,s_size=3,a_size=6,h_size=8) #Load the agent.
    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 1
        total_reward = []
        total_lenght = []
        
        for i in range(nepisodes):
        rospy.loginfo ("STARTING Episode #"+str(i))
        
        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        while i < total_episodes:
            s = env.reset()
            running_reward = 0
            ep_history = []
            for j in range(max_ep):
                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)

                s1,r,d,_ = env.step(a) #Get our reward for taking an action given.
                ep_history.append([s,a,r,s1])
                s = s1
                running_reward += r
                print("Total Learning Reward ="+str(r))
                print("Episode number ="+str(i))
                print("Step number ="+str(j))
                print("Done? "+str(d))
                print("Running_reward "+str(running_reward))
                print("Done? "+str(d))
        
      
                if d == True:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    feed_dict={myAgent.reward_holder:ep_history[:,2],
                            myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad
                    print("Discounted_rewards ="+str(discount_rewards))

                    if i < total_episodes:
                        feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                
                    total_reward.append(running_reward)
                    total_lenght.append(j)
                    print("Total reward sum = "+str(np.sum(running_reward)))
                                
                    break

        
            #Update our running tally of scores.
            if i % 10 == 0:
                print("Mean Total Reward = "+str(np.mean(total_reward)))
            i += 1
            s = env.reset()
            d = False

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo ( ("EP: "+str(i+1)+" - [alpha: "+str(round(Ros_Policy.alpha,2))+" - gamma: "+str(round(Ros_Policy.gamma,2))+" - epsilon: "+str(round(Ros_Policy.epsilon,2))+"] - Reward: "+str(running_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    
    rospy.loginfo ( ("\n|"+str(i)+"|"+str(Ros_Policy.alpha)+"|"+str(Ros_Policy.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(running_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

