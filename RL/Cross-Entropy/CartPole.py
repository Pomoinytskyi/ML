import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" #torch.device("cuda" if args.cuda else "cpu")

HiddenLayerSize = 128
TrainingBatchSize = 16
Percentile = 90

class Network(nn.Module):
    def __init__(self, observationSize, hiddenLayersSize, actionsNumber):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observationSize, hiddenLayersSize),
            nn.ReLU(),
            nn.Linear(hiddenLayersSize, actionsNumber)
        )
    
    def forward(self, x):
        return self.net(x)
    
Episode = namedtuple('Episode', field_names = ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names = ['observation', 'action'])

def iterate_batches(environment, network, batchSize):
    batch = []
    episodeReward = 0.0
    episodeSteps = []
    observation = environment.reset()
    softmaxLayer = nn.Softmax(dim = 1)

    while True:
        observationVector = torch.FloatTensor([observation]).to(device)
        actionsScoresVector = network(observationVector)
        actionProbabilitiesTensor = softmaxLayer(actionsScoresVector)
        actionProbabilitiesVector = actionProbabilitiesTensor.to("cpu").data.numpy()[0] #unpack actionprobabilities from torch vector

        action = np.random.choice(len(actionProbabilitiesVector), p=actionProbabilitiesVector)
        nextObservation, revard, isDone, _ = environment.step(action)

        episodeReward += revard
        step = EpisodeStep(observation = observation, action = action)
        episodeSteps.append(step)

        if isDone:
            e = Episode(reward = episodeReward, steps = episodeSteps)
            batch.append(e)
            episodeReward = 0.0
            episodeSteps = []
            nextObservation = environment.reset()

            if len(batch) == batchSize :
                yield batch
                batch = []

        observation = nextObservation        

def filterBatch(batch, percentile):
    rewards = list( map(lambda s: s.reward, batch))
    rewardBoundary = np.percentile(rewards, percentile)
    rewardMean = float(np.mean(rewards))
    
    trainObservations = []
    trainActions = []
    
    for reward, steps in batch:
        if reward < rewardBoundary:
            continue
        trainObservations.extend( map(lambda step: step.observation, steps) )
        trainActions.extend( map( lambda step: step.action, steps))
    
    trainObservationTensor = torch.FloatTensor(trainObservations).to(device)
    trainActionTensor = torch.LongTensor(trainActions).to(device)
    return trainObservationTensor, trainActionTensor, rewardBoundary, rewardMean
    
if __name__ == "__main__":
    environment = gym.make("CartPole-v0")
    environment = gym.wrappers.Monitor(environment, directory="mon", force=True)
    
    observationSize = environment.observation_space.shape[0]
    numberOfActions = environment.action_space.n

    network = Network(observationSize, HiddenLayerSize, numberOfActions).to(device)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = network.parameters(), lr = 0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iterationNumber, batch in enumerate(iterate_batches( environment, network, TrainingBatchSize )):
        observation, actsVector, rewardBoundary, rewardMean = filterBatch(batch, Percentile)
        
        optimizer.zero_grad()

        actionsScoresVector = network(observation)
        lossVector = objective(actionsScoresVector, actsVector)
        lossVector.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iterationNumber, lossVector.item(), rewardMean, rewardBoundary))
        writer.add_scalar("loss", lossVector.item(), iterationNumber)
        writer.add_scalar("reward_bound", rewardBoundary, iterationNumber)
        writer.add_scalar("reward_mean", rewardMean, iterationNumber)
        if rewardMean > 199:
            print("Solved!")
            break

    
    writer.close()

    print("Testing!")
    iterate_batches(environment, network, 200)
    print("Done!")



