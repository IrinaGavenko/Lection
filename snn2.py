import random

from math import exp
from math import fabs

THRESHOLD = 0.07
TAU1 = 4.0
TAU2 = 2.0
TAU3 = 20.0

def randomWeight():
    return (pow(random.random(), 0.5)) / 8 * (random.randint(1, 2) * 2 - 3)

class Neuron:
    synapses = []
    potential = 0.0

    @staticmethod
    def potentialFunction(deltaIncludeDelay):
        return exp(-deltaIncludeDelay / TAU1) - exp(-deltaIncludeDelay / TAU2)

    @staticmethod
    def potentialFunctionDerivative(deltaIncludeDelay):
        return -exp(-deltaIncludeDelay / TAU1) / TAU1 + exp(-deltaIncludeDelay / TAU2) / TAU2

    @staticmethod
    def refractoryFunction(t):
        return THRESHOLD * exp(1 - t) / TAU3

    def add(self, to, delay, weight):
        self.synapses.append((to, delay, weight))

    def step(self):
        for synapse in self.synapses:
            synapse[0].potential += synapse

class InputNeuron: # just input
    def __init__(self, id_, type='input'):
        self.id = id_
        self.type = 'input'
        self.outputs = []
        self.inputSpikeTrain = []

    def setSpikeTrain(self, inputSpikeTrain):
        self.inputSpikeTrain = inputSpikeTrain[::]

    def addOutput(self, connection):
        self.outputs.append(connection)

    def clear(self):
        self.inputSpikeTrain.clear()
        for connection in self.outputs:
            for synapse in connection.synapses:
                synapse[2].clear()

    def propagate(self, t):
        if t in self.inputSpikeTrain:
            for connection in self.outputs:
                connection.putSpike(t) # push to synapses of the connection

class HiddenNeuron: # real neuron
    def __init__(self, id_, type_='hidden'):
        self.id = id_
        self.type = type_
        self.inputs = []
        self.outputs = []
        self.lastSpikeTime = -1

    def addInput(self, connection):
        self.inputs.append(connection)

    def addOutput(self, connection):
        self.outputs.append(connection)

    def clear(self):
        self.lastSpikeTime = -1
        for connection in self.outputs:
            for synapse in connection.synapses:
                synapse[2].clear()

    def propagate(self, t):
        sum = -THRESHOLD

        for connection in self.inputs:
            for synapse in connection.synapses:
                delay = synapse[0]
                weight = synapse[1]
                for spikeTime in synapse[2]: # are there more than one? yes, after the 0,1 time simulations during the network propagation. 
                    if spikeTime + delay <= t:
                        sum += Neuron.potentialFunction(t - spikeTime - delay) * weight

        if self.lastSpikeTime > 0:
            sum -= Neuron.refractoryFunction(t - self.lastSpikeTime)
        if sum > 0:
            self.lastSpikeTime = t
            for connection in self.outputs:
                connection.putSpike(t) # push to synapses of the connection

class OutputNeuron: # real neuron
    def __init__(self, id_, type_='output'):
        self.id = id_
        self.type = type_
        self.inputs = []
        self.outputSpikeTrain = []
        self.lastSpikeTime = -1

    def addInput(self, connection):
        self.inputs.append(connection)

    def clear(self):
        self.lastSpikeTime = -1
        self.outputSpikeTrain.clear()

    def propagate(self, t):
        sum = -THRESHOLD # down for a level - smart way, I like it

        for connection in self.inputs:
            for synapse in connection.synapses:
                delay = synapse[0] # this is integer...
                weight = synapse[1]
                for spikeTime in synapse[2]:
                    if spikeTime + delay <= t:
                        sum += Neuron.potentialFunction(t - spikeTime - delay) * weight

        if self.lastSpikeTime > 0:
            sum -= Neuron.refractoryFunction(t - self.lastSpikeTime)
        if sum > 0:
            self.lastSpikeTime = t
            self.outputSpikeTrain.append(t)

class Connection:
    def __init__(self, synapses, presynapticNeuron, postsynapticNeuron, innov_num):
        # synapse is order_number, weight, load array (initially empty)
        self.innov_num = innov_num
        self.synapses = [[synapse[0], synapse[1], []] for synapse in synapses]
        # binding
        presynapticNeuron.addOutput(self)
        postsynapticNeuron.addInput(self)
        self.input = presynapticNeuron
        self.output = postsynapticNeuron
        self.enable = True # for NEAT mutations and crossover

    def putSpike(self, t):
        for synapse in self.synapses: # do it for all equally? what for should we keep more than one synapse? - ah, they have doferent weights...
            synapse[2].append(t) # load to array of signals

class Network:
    def __init__(self, net_config=None):
        self.score = None
        self.net_config = net_config
        self.innov_nums = set()
        self.innov_num = 0
        self.neuron_ids = set()
        self.neuron_num = 0
        self.default_tMax = 30
        self.neurons = []
        self.connections = []
        self.layers = [] # three layers default
 
    def addLayer(self, newLayer=None):
        if newLayer is None:
            newLayer = []
        self.layers.append(newLayer)
        return newLayer

    def clear(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.clear()
 
    def add_neuron(self, type_, id_=None):
        if type_ == 'input':
            new_neuron = InputNeuron(id_ if id_ else self.neuron_num)
            self.layers[0].append(new_neuron)
        elif type_ == 'output':
            new_neuron = OutputNeuron(id_ if id_ else self.neuron_num)
            self.layers[2].append(new_neuron)
        else:
            new_neuron = HiddenNeuron(id_ if id_ else self.neuron_num)
            self.layers[1].append(new_neuron)

        self.neurons.append(new_neuron)
        if id_ is None: # else - reconstructing
            self.neuron_ids.add(self.neuron_num)
            self.neuron_num += 1
        return new_neuron

    def get_neuron(self, neuron_id):
        for neuron in self.neurons:
            if neuron.id == neuron_id:
                return neuron
        return None 

    def add_connection(self, input_neuron, output_neuron, synapses=None):
        innov_num = self.innov_num 
        self.innov_nums.add(innov_num)
        self.innov_num += 1
        if synapses is None:
            synapses = [(i * 3 + 1, randomWeight()) for i in range(self.net_config['synapsesPerConnection'])]
        self.connections.append(Connection(synapses, input_neuron, output_neuron, innov_num))
    
    def get_connection(self, innov_num):
        for connection in self.connections:
            if connection.innov_num == innov_num:
                return connection
        return None
    
    def serialize(self):
        dictForm = {
            'connections': [],
            'neurons': [],
            'score': self.score,
            'net_config': self.net_config,
            'innov_nums': self.innov_nums,
            'innov_num': self.innov_num,
            'neuron_ids': self.neuron_ids,
            'neuron_num': self.neuron_num,
        }

        for conn in self.connections:
            dictForm['connections'].append({
                'input_id': conn.input.id,
                'output_id': conn.output.id,
                'synapses': conn.synapses,
                'innov_num': conn.innov_num,
                'enable': conn.enable
            })

        for neur in self.neurons:
            dictForm['neurons'].append({
                'type': neur.type,
                'id': neur.id,
            })

        return dictForm


    def reConstruct(self, dictForm):
        self.score = dictForm['score']
        self.net_config = dictForm['net_config']
        self.innov_nums = dictForm['innov_nums']
        self.innov_num = dictForm['innov_num']
        self.neuron_ids = dictForm['neuron_ids']
        self.neuron_num = dictForm['neuron_num']

        self.addLayer()
        self.addLayer()
        self.addLayer()
        for neur in dictForm['neurons']:
            self.add_neuron(neur['type'], neur['id'])

        for conn in dictForm['connections']:
            new_conn = Connection(conn['synapses'], self.get_neuron(conn['input_id']), self.get_neuron(conn['output_id']), conn['innov_num'])
            new_conn.enable = conn['enable']
            self.connections.append(Connection)


    def getParams(self):
        params = []
        for layer in self.layers[:-1:]: # do not look into output neurons, it doesn't have out connections
            for presynapticNeuron in layer: # for just a neuron
                for connection in presynapticNeuron.outputs: # out connection
                    for synapse in connection.synapses: # out weight
                        params.append(synapse[1]) # weight
        return params



    def add_node_mutation(self):
        """
        This method adds a node by modifying connection genes.
        In the add node mutation an existing connection is split and the new node placed where the old
        connection used to be. The old connection is disabled and two new connections are added to the genotype.
        The new connection leading into the new node receives a weight of 1, and the new connection leading
        out receives the same weight as the old connection.
        """
        # Get new_node id
        ### YOUR CODE ###

        # Get a random existing connection
        ### YOUR CODE ###



    def creates_cycle(self, node_in_id, node_out_id):
        """
        Checks if the addition of a connection gene will create a cycle in the computation graph
        :param node_in_id: In node of the connection gene
        :param node_out_id: Out node of the connection gene
        :return: Boolean value
        """
        if node_in_id == node_out_id:
            return True

        visited = {node_out_id}
        while True:
            num_added = 0

            for connection in self.connections:
                if connection.input.id in visited and connection.output.id not in visited:
                    if connection.output.id == node_in_id:
                        return True
                    else:
                        visited.add(connection.output.id)
                        num_added += 1

            if num_added == 0:
                return False


    def _is_valid_connection(self, neuron_in_id, neuron_out_id):
        does_creates_cycle = self.creates_cycle(neuron_in_id, neuron_out_id)
        does_connection_exist = self._does_connection_exist(neuron_in_id, neuron_out_id)
        return (not does_creates_cycle) and (not does_connection_exist)


    def _does_connection_exist(self, node_1_id, node_2_id):
        for connection in self.connections:
            if (connection.input.id == node_1_id) and (connection.output.id == node_2_id):
                return True
            elif (connection.input.id == node_2_id) and (connection.output.id == node_1_id):
                return True
        return False


    def add_connection_mutation(self):
        """
        In the add connection mutation, a single new connection gene is added
        connecting two previously unconnected nodes.
        """

        ### YOUR CODE ###

    def get_num_excess_genes(self, other):
        num_excess = 0

        for connection in self.connections:
            if connection.innov_num > other.innov_num:
                num_excess += 1

        for n in self.neurons:
            if n.id > other.neuron_num:
                num_excess += 1

        return num_excess

    def get_num_disjoint_genes(self, other):
        num_disjoint = 0

        for connection in self.connections:
            if connection.innov_num <= other.innov_num:
                if other.get_connection(connection.innov_num) is None:
                    num_disjoint += 1

        for n in self.neurons:
            if n.id <= other.neuron_num:
                if other.get_neuron(n.id) is None:
                    num_disjoint += 1

        return num_disjoint

    def get_avg_weight_difference(self, other):
        weight_difference = 0.0
        num_weights = 0.0

        for connection in self.connections:
            matching_connection = other.get_connection(connection.innov_num)
            if matching_connection is not None:
                for s_i in range(len(connection.synapses)):
                    weight_difference += float(connection.synapses[s_i][1]) - float(matching_connection.synapses[s_i][1])
                    num_weights += 1

        if num_weights == 0.0:
            num_weights = 1.0
        return weight_difference / num_weights



    def propagateFull(self, x, tMax=None):
        # init
        self.clear()
        for i, spikeTrain in enumerate(x): # slice input spikeTrain to inputs
            self.layers[0][i].setSpikeTrain(spikeTrain) # number of input must be the shape number as trains 
        lastLayer = self.layers[len(self.layers) - 1] # he output layer

        # simulate
        if tMax is None: tMax = self.default_tMax
        for t in range(tMax):
            for layer in self.layers: # from input to last. but if there is some complicated connection sheme thing in hidden (the layers[1])?
                for neuron in layer: # should I make more layers dynamically? or should I chage propagarion strategy? second
                    neuron.propagate(t)

            for outputNeuron in lastLayer:
                if len(outputNeuron.outputSpikeTrain) > 0: # return just when some of them get their fire!
                    return [neuron.outputSpikeTrain for neuron in lastLayer] # the output array

        return [neuron.outputSpikeTrain for neuron in lastLayer] # return output array anyway
