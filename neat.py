import sys
import random
import numpy as np
from copy import deepcopy


from snn2 import Network, Connection, InputNeuron, HiddenNeuron, OutputNeuron
from net_creator import BrainFactory


class Species:
    def __init__(self, id, model_net, generation):
        self.id = id
        self.model_net = model_net
        self.members = []
        self.fitness_history = []
        self.fitness = None
        self.adjusted_fitness = None
        self.last_improved = generation

    @staticmethod
    def species_distance(net_1, net_2):
        C1 = 1.0
        C2 = 1.0
        C3 = 0.5
        N = 1

        num_excess = net_1.get_num_excess_genes(net_2)
        num_disjoint = net_1.get_num_disjoint_genes(net_2)
        avg_weight_difference = net_1.get_avg_weight_difference(net_2)

        distance = (C1 * num_excess) / N
        distance += (C2 * num_disjoint) / N
        distance += C3 * avg_weight_difference

        return distance

    @staticmethod
    def stagnation(species, generation):
        """
        From https://github.com/CodeReclaimers/neat-python/neat/stagnation.py
        :param species: List of Species
        :param generation: generation number
        :return:
        """
        species_data = []
        for s in species:
            if len(s.fitness_history) > 0:
                prev_fitness = max(s.fitness_history)
            else:
                # super small number
                prev_fitness = -sys.float_info.max

            s.fitness = max([net.score for net in s.members])
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None

            if prev_fitness is None or s.fitness>prev_fitness:
                s.last_improved = generation

            species_data.append(s)

        # sort in ascending fitness order
        species_data.sort(key=lambda g: g.fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for i, s in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > 1:
                is_stagnant = stagnant_time >= 10

            if (len(species_data) - i) <= 1:
                is_stagnant = False

            if (len(species_data) - i) <= 1:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result



class PopulationPool:
    def __init__(self, population_size, net_config, env_generator, net_evaluator, nCPU=1):
        self.nCPU = nCPU
        self.best_net = None
        self.population = []
        self.species = []
        self.net_config = net_config
        self.env_generator = env_generator
        self.net_evaluator = net_evaluator
        self.population_size = population_size
        self.initilize_populaion()


    def initilize_populaion(self):
        factory = BrainFactory(self.net_config)
        for _ in range(self.population_size):
            new_net = factory.createNet()
            new_net.score = self.score_net(new_net)
            self.population.append(new_net)
            self.speciate(new_net, 0)

        self.update_best()


    def speciate(self, net, generation_i):
        """
        Places Genome into proper species - index
        :param genome: Genome be speciated
        :param generation: Number of generation this speciation is occuring at
        :return: None
        """
        for species in self.species:
            if Species.species_distance(net, species.model_net) <= self.net_config['speciation_threshold']:
                net.species = species.id
                species.members.append(net)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(self.species), net, generation_i) # len(self.species) is a new auto-incrementing id
        net.species = new_species.id
        new_species.members.append(net)
        self.species.append(new_species)


    def update_best(self):
        best_i = max(enumerate(self.population), key=lambda x: self.population[x[0]].score)[0]
        self.best_net = self.population[best_i]


    def score_net(self, net):
        if net.score is None:
            with self.env_generator() as env:
                net.score = self.net_evaluator(env, net)
        return net.score


    def crossover(self, net_1, net_2):
        """
        Crossovers two Genome instances as described in the original NEAT implementation
        :param genome_1: First Genome Instance
        :param genome_2: Second Genome Instance
        :param config: Experiment's configuration class
        :return: A child Genome Instance
        """

        child = Network(self.net_config)
        inputLayer = []
        hiddenLayer = []
        outputLayer = []
        all_neurons = []
        best_parent, other_parent = self.order_parents(net_1, net_2)


        # Crossover Nodes
        # Randomly add matching genes from both parents
        for neuron in best_parent.neurons:
            matching_neuron = other_parent.get_neuron(neuron.id)

            if matching_neuron is not None:
                # Randomly choose where to inherit gene from
                if random.choice([True, False]):
                    child_neuron = neuron
                else:
                    child_neuron = matching_neuron

            # No matching gene - is disjoint or excess
            # Inherit disjoint and excess genes from best parent
            else:
                child_neuron = neuron

            # if there we some state in neuron to inherit here transmission should have been done.
            # now spike neuron doesn't have any specific INDIVIDUAL fixed parameters.
            if child_neuron.type == 'input':
                child_neuron = InputNeuron(child_neuron.id)
                inputLayer.append(child_neuron)
            elif child_neuron.type == 'output':
                child_neuron = OutputNeuron(child_neuron.id)
                outputLayer.append(child_neuron)
            else: # hidden
                child_neuron = HiddenNeuron(child_neuron.id)
                hiddenLayer.append(child_neuron)
            all_neurons.append(child_neuron)

        child.addLayer(inputLayer)
        child.addLayer(hiddenLayer)
        child.addLayer(outputLayer)
        child.neurons = all_neurons
        child.neuron_ids = {n.id for n in all_neurons}
        child.neuron_num = len(all_neurons)


        # Crossover connections
        # Randomly add matching genes from both parents - remember we may have multiple synapses in each connection
        for connection in best_parent.connections:
            matching_connection = other_parent.get_connection(connection.innov_num)

            if matching_connection is not None:
                # Randomly choose where to inherit gene from
                if(random.choice([True, False]) and
                    child.get_neuron(matching_connection.input.id) is not None and
                    child.get_neuron(matching_connection.output.id) is not None):
                    child_gene = matching_connection
                else:
                    child_gene = connection # deepcopy is useless because of inner structure of connection - it has links

            # No matching gene - is disjoint or excess
            # Inherit disjoint and excess genes from best parent
            else:
                child_gene = connection

            # clone  connection object and install it
            # here will be some checks on existing of needed neurons with ids for binding a new connection
            # there may be a situation, when another parent have neurons neurons than best and few connections,
            # so there may exist matching by id connection, but it may connect very distant neurons, that's ids
            # doesn't exist in child (child inherits not more and not less number of neurons best parent has).
            # how to behace here - who know, may be just ignore or take the best parent connection - yes, ok, this way look upstairs - solved.

            # check-get input-neuron.
            child_input_neuron = child.get_neuron(connection.input.id)
            # if child_input_neuron is None:
            #     raise Exception('No neuron found in child with id: {}'.format(connection.input.id))

            # check-get output-neuron
            child_output_neuron = child.get_neuron(connection.output.id)
            # if child_output_neuron is None:
            #     raise Exception('No neuron found in child with id: {}'.format(connection.output.id))

            # clone connection to child
            new_connection = Connection(deepcopy(connection.synapses), child_input_neuron,  child_output_neuron, connection.innov_num)
            child.connections.append(new_connection)
            child.innov_nums.add(new_connection.innov_num)
            child.innov_num += 1

            # Apply rate of disabled gene being re-enabled
            if not new_connection.enable:
                is_reenabeled = random.uniform(0, 1) <= self.net_config['crossover_reenable_connection_rate']
                enabled_in_best_parent = best_parent.get_connection(new_connection.innov_num).enable

                if is_reenabeled or enabled_in_best_parent:
                    new_connection.enabled = True


        return child


    def order_parents(self, net_1, net_2):
        """
        Orders parents with respect to fitness
        :param net_1: First Parent Genome
        :param net_2: Secont Parent Genome
        :return: Two Genome Instances
        """

        # Figure out which genotype is better
        # The worse genotype should not be allowed to add excess or disjoint genes
        # If they are the same, use the smaller one's disjoint and excess genes

        # initial setup
        best_parent = net_1
        other_parent = net_2

        len_net_1 = len(net_1.connections)
        len_net_2 = len(net_2.connections)

        if net_1.score == net_2.score:
            if len_net_1 == len_net_2:
                # Fitness and Length equal - randomly choose best parent
                if random.choice([True, False]):
                    best_parent = net_2
                    other_parent = net_1
            # Choose minimal parent
            elif len_net_2 < len_net_1:
                best_parent = net_2
                other_parent = net_1

        elif net_1.score < net_2.score:
            best_parent = net_2
            other_parent = net_1

        return best_parent, other_parent


    def mutate(self, net):
        """
        Applies connection and structural mutations at proper rate.
        Connection Mutations: Uniform Weight Perturbation or Replace Weight Value with Random Value
        Structural Mutations: Add Connection and Add Node
        :param genome: Genome to be mutated
        :param config: Experiments' configuration file
        :return: None
        """

        if random.uniform(0, 1) < self.net_config['connection_mutation_rate']:
            for connection in net.connections:
                if random.uniform(0, 1) < self.net_config['connection_perturbation_rate']:
                    for s_i in range(len(connection.synapses)):
                        perturb = random.uniform(-1, 1)
                        connection.synapses[s_i][1] += perturb
                else:
                    for s_i in range(len(connection.synapses)):
                        connection.synapses[s_i][1] = np.random.normal(0, 1, 1).tolist()[0]

        if random.uniform(0, 1) < self.net_config['add_node_mutation_rate']:
            net.add_node_mutation()

        if random.uniform(0, 1) < self.net_config['add_connection_mutation_rate']:
            net.add_connection_mutation()



    def run(self, generations_number):
        for generation_i in range(generations_number): # new experiment

            # Reproduce - make scpecies and mere the fitnesses-scores
            all_fitnesses = []
            remaining_species = []
            for species, is_stagnant in Species.stagnation(self.species, generation_i):
                if is_stagnant:
                    self.species.remove(species)
                else:
                    all_fitnesses.extend(net.score for net in species.members)
                    remaining_species.append(species)


            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)

            fit_range = max(1.0, (max_fitness-min_fitness))
            for species in remaining_species:
                # Set adjusted fitness
                avg_species_fitness = np.mean([g.score for g in species.members])
                species.adjusted_fitness = (avg_species_fitness - min_fitness) / fit_range

            adj_fitnesses = [s.adjusted_fitness for s in remaining_species]
            adj_fitness_sum = sum(adj_fitnesses)

            # Get the number of offspring for each species
            new_population = []
            for species in remaining_species: # let them all have children - in proportion to species fitnesses
                if species.adjusted_fitness > 0:
                    size = max(2, int((species.adjusted_fitness/adj_fitness_sum) * self.population_size))
                else:
                    size = 2

                # sort current members in order of descending fitness
                cur_members = species.members
                cur_members.sort(key=lambda net: net.score, reverse=True)
                species.members = []  # reset

                # save top individual in species
                new_population.append(cur_members[0]) # this is the net obj
                size -= 1

                # Only allow top x% to reproduce
                purge_index = int(self.net_config['percentage_to_save'] * len(cur_members))
                purge_index = max(2, purge_index)
                cur_members = cur_members[:purge_index]

                for _ in range(size): # among best from species
                    parent_1 = random.choice(cur_members)
                    parent_2 = random.choice(cur_members)

                    child_net = self.crossover(parent_1, parent_2)
                    self.mutate(child_net)
                    new_population.append(child_net)

            # Set new population
            self.population = new_population
            for net in self.population:
                self.score_net(net)

            # Speciate
            for net in self.population:
                self.speciate(net, generation_i)

            if self.best_net.score >= self.net_config['fitness_threshold']:
                return self.best_net, generation_i