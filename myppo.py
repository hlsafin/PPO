import torch
import numpy as np
import gym
import torch.nn as nn
import time 
import torch.optim as optim
from torch.distributions import Categorical
## HyperParameters
learning_rate = 0.0003
gamma = 0.99
lmbda = 0.95
entr_coef = 0.001
max_timeSteps = 400000
update_timestep = 200
ep_clip = 0.2
n_episodes = 1000
class ActorCritic(nn.Module):
    
    def __init__ (self,inputLayer,actionSpace):
        super(ActorCritic,self).__init__()
        self.actor = nn.Sequential(nn.Linear(inputLayer,64),
                                   nn.Tanh(),
                                   nn.Linear(64,32),
                                   nn.Tanh(),
                                   nn.Linear(32,actionSpace),
                                   nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(inputLayer,64),
                                    nn.Tanh(),
                                    nn.Linear(64,32),
                                    nn.Tanh(),
                                    nn.Linear(32,1))
#        
#        self.actor = nn.Sequential(nn.Linear(inputLayer,64),
#                                   nn.Tanh(),
#                                   nn.Linear(64,32),
#                                   nn.Tanh(),
#                                   nn.Linear(32,actionSpace),
#                                   nn.Softmax(dim=-1))
#        self.critic = nn.Sequential(nn.Linear(inputLayer,64),
#                                    nn.Tanh(),
#                                    nn.Linear(64,32),
#                                    nn.Tanh(),
#                                    nn.Linear(32,1))
        
        
        self.reward =[]
        self.counter =0
        self.done = []
        self.value = []
        self.action_prob= []
        self.action=[]
        self.log_prob=[]
        self.oldlog_prob = []
        self.MSE_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        
                                    
        
    def forward(self,x):
        actor = self.actor(x)        
        critic = self.critic(x)

        return actor,critic
    
    def clean_data(self):
        self.reward =[]
        self.done = []
        self.value = []
        self.action_prob= []
        self.action=[]
        self.log_prob=[]
        
    def update(self,next_value):
        
        
        self.done =[0 if x==True else 1 for x in self.done]
        self.value = self.value+[next_value.item()]
        self.reward =torch.FloatTensor(self.reward)
        self.done = torch.FloatTensor(self.done)
        self.value = torch.FloatTensor(self.value)
        self.action_prob= torch.FloatTensor(self.action_prob)
        self.action=torch.FloatTensor(self.action)
        self.log_prob=torch.FloatTensor(self.log_prob)
        
#        self.value = self.value+[next_value.item()]
#        self.reward =torch.FloatTensor(self.reward)
#        self.done = torch.FloatTensor(self.done)
#        self.value = torch.FloatTensor(self.value)
#        self.action_prob= torch.FloatTensor(self.action_prob)
#        self.action=torch.FloatTensor(self.action)
#        self.log_prob=torch.FloatTensor(self.log_prob)
        
        
        gae = 0
        returns =[]        
        
        for step in reversed(range(len(self.reward))):
            delta= self.reward[step] + gamma*self.value[step+1]*self.done[step] - self.value[step]
            #advantage.append(delta)
            
            gae = delta+ gamma * lmbda * self.done[step] * gae
            returns.insert(0,gae.item()+self.value[step].item())
        returns = torch.FloatTensor(returns)
        returns.detach()
        advantage = np.asarray(returns) - np.asarray(self.value[:-1])   
        
        #advantage = (advantage - np.mean(advantage)/np.std(advantage) + 1e-10)
        advantage = torch.FloatTensor(advantage)
        
        if self.counter==0:
            self.oldlog_prob = self.log_prob
            self.counter+=1
            return 0
        else:
            
            ratio = torch.exp(self.log_prob - self.oldlog_prob)
            surr1 = ratio * advantage
            
            surr2 = torch.clamp(ratio,1-ep_clip,1+ep_clip)*advantage
            
            #actor_loss = - torch.min(surr1,surr2)
            #actor_loss=torch.tensor(actor_loss, dtype=torch.float, requires_grad=True)
            
            #critic_loss = 0.5 * self.MSE_loss(returns, self.reward) 
            #critic_loss = torch.tensor(critic_loss, dtype=torch.float, requires_grad=True)
            loss = - torch.min(surr1,surr2) +  0.5 * self.MSE_loss(returns, self.value[:-1]) 
            loss = torch.tensor(loss, dtype=torch.float,requires_grad=True)
            
            
            return loss
            
           
#
#            self.optimizer.zero_grad()            
#            loss.mean().backward()
#            self.optimizer.step()
#            self.oldlog_prob = self.log_prob
#        
        
        
        #surr1 =
            
            
            
            
            
            
        
        
        
#for param in model.parameters():
#    print(param)
        
        
        
        
    


model = ActorCritic(4,2)   
model.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
env = gym.make('CartPole-v0')
done = False
time_step = 0
avg = 0
counter = 0
#for ep in range(n_episodes):    
state= env.reset()
state = torch.FloatTensor(state)   
while time_step<=max_timeSteps:
    time_step+=1                     
    action_prob,value = model(state)            
    sample_item = Categorical(action_prob)
    action = sample_item.sample().item()
    log_prob = sample_item.log_prob(torch.FloatTensor([action]))
    model.log_prob.append(log_prob.item())
    model.value.append(value.item())
    model.action_prob.append((action_prob[action]).item())
    model.action.append(action)              
    state, reward, done,_ = env.step(action)
    avg =avg+reward
    model.done.append(done)
    model.reward.append(reward)                     
    state = torch.FloatTensor(state)
    if done:
        state= env.reset()
        state = torch.FloatTensor(state)  
        #print(avg)
        avg=0
        
    if time_step%update_timestep==0:
        _,next_value = model(state)
        #next_value = model.critic(state)
        loss=model.update(next_value)
        if counter==0:
            counter = 1
            model.clean_data()
            continue
        else:
            
            model.optimizer.zero_grad()            
            loss.mean().backward()
            print(loss.mean())
            model.optimizer.step()
            model.oldlog_prob = model.log_prob
            model.clean_data()
        
#        if time_step==max_timeSteps:  
#            model.clean_data()
#            time_step=0

            
            
                
   


        
        
        
        

