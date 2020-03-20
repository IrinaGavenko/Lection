import numpy
from math import *
from random import Random
random = Random()

from snn2 import Network, InputNeuron, OutputNeuron, HiddenNeuron, Connection

def randomWeight():
    # return (random.random() * 2 - 1) / 8
    return (pow(random.random(), 0.5)) / 8 * (random.randint(1, 2) * 2 - 3)

class BrainFactory(object):
    def __init__(self, net_config):
        self.params = None
        self.net_config = net_config
        self.epoch = 1
        self.round = 1
        self.max_score = [-100.0, 0]

    def createNet(self):
        """
        1) initialize net
        2) make neurons
        3) make connections
        """

        # 1) initialize net
        net = Network(self.net_config)

        # 2) make neurons
        inputLayer = net.addLayer()
        for _ in range(self.net_config['inputs_num']):
            neorun_id = net.neuron_num
            net.neuron_ids.add(neorun_id)
            net.neuron_num += 1
            inputNeuron = InputNeuron(net.neuron_num)
            inputLayer.append(inputNeuron)
            net.neurons.append(inputNeuron)

        hiddenLayer = net.addLayer()
        for _ in range(self.net_config.get('hidden_num', 0)):
            neorun_id = net.neuron_num
            net.neuron_ids.add(neorun_id)
            net.neuron_num += 1
            hiddenNeuron = HiddenNeuron(net.neuron_num)
            hiddenLayer.append(hiddenNeuron)
            net.neurons.append(hiddenNeuron)

        outputLayer = net.addLayer()
        for _ in range(self.net_config['outputs_num']):
            neorun_id = net.neuron_num
            net.neuron_ids.add(neorun_id)
            net.neuron_num += 1
            outputNeuron = OutputNeuron(net.neuron_num)
            outputLayer.append(outputNeuron)
            net.neurons.append(outputNeuron)

        # 3) make connections
        synapsesPerConnection = self.net_config['synapsesPerConnection']
        for hiddenNeuron in hiddenLayer:
            for inputNeuron in inputLayer:
                c_innov_num = net.innov_num 
                net.innov_nums.add(c_innov_num)
                net.innov_num += 1
                net.connections.append(Connection( [(i * 3 + 1, randomWeight()) for i in range(synapsesPerConnection)], inputNeuron, hiddenNeuron, c_innov_num))        

        for outputNeuron in outputLayer:
            for hiddenNeuron in hiddenLayer:
                c_innov_num = net.innov_num 
                net.innov_nums.add(c_innov_num)
                net.innov_num += 1
                net.connections.append(Connection( [(i * 3 + 1, randomWeight()) for i in range(synapsesPerConnection)], hiddenNeuron, outputNeuron, c_innov_num))        

        return net

    def score(self, score, params):
        print("Score: {}; max_score: {}".format(score, self.max_score))
        if score > self.max_score:
            self.max_score = score
            self.params = params