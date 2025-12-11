import torch
import math
import numpy as np

class MinMaxStats :
    def __init__ (self , known_bounds=None):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        self.known_bounds = known_bounds
    def update(self , value ) :
        self.maximum = max(self.maximum , value)
        self.minimum = min(self.minimum ,value)
    def normalize(self , value ) :
        # adding +1 to avoid weird divisions
        if self.known_bounds :
            return (value - self.known_bounds.min) / (self.known_bounds.max - self.known_bounds.min)
        if self.maximum > self.minimum :
            return(value - self.minimum) / (self.maximum - self.minimum)
        # fixed return for cases when equal for initial states
        # return (value +1 ) / (self.maximum + 1)
        return 0.5

class Node :
    def __init__ (self , prior ) :
            self.visit_count = 0
            self.to_play = -1
            self.prior = prior
            self.value_sum = 0
            self.children = {}
            self.hidden_state = None
            self.reward = 0
            self.action = None
            self.observation = None
    def expanded(self) :
        return len(self.children) > 0
    def value(self) :
        if self.visit_count == 0 :
            return 0
        return self.value_sum / self.visit_count
class MCTS :
    def __init__ (self , config) :
        self.config = config

    def run( self , root,  network ,  min_max_stats) :
        """
        Core MCTS loop :
        1. Select a leaf node
        2. Expand the node using the Network
        3. Backpropagate the value up the tree
        """

        if root.hidden_state is None :
            if root.observation is not None :
                root.hidden_state = network.representation(root.observation)
                policy_logits , value = network.prediction(root.hidden_state)
                self.expand_node(root , policy_logits)
                self.backpropagate([root] , value , min_max_stats , self.config.discount)
            else :
                return # cant run without state >_< give me a state lil bro

        for _ in range(self.config.num_simulations) :
            node = root
            search_path = [node]
            while node.expanded() :
                action , node = self.select_child(node , min_max_stats)
                search_path.append(node)
            parent = search_path[-2] if len(search_path) > 1 else None # The parent of the leaf
            action = search_path[-1].action # action leadin to this leaf
            # EXPANSION
            # if in root we have hidden state from representaion net
            # if deeper in mcts we use dynamic net to generate new hidden state
            if node.hidden_state is None and parent is not None :
               # redundant condition now
                # using hidden state from dynamics network
                #we use parent hidden state and action we took
                network_output = network.dynamics(parent.hidden_state , [action] )
                node.hidden_state = network_output[0]
                # node.reward = network_output[1].item()
                #cinvert tensor to float 
                node.reward = network_output[1].item()
                #now asking prediction netework if this is a good state
                #policylogits what action  / mode
                #value how likely to win
                policy_logits , value = network.prediction(node.hidden_state)
                #we use policy to set prior prob for each child
                self.expand_node(node , policy_logits )
            elif node.hidden_state is not None and not node.expanded():
                # If we hit a node that has state but no children (rare edge case), evaluate it
                # gpt says to add this idk how this is possible tbh just in case adding
                policy_logits , value = network.prediction(node.hidden_state)
                self.expand_node(node , policy_logits )
            else :
                #redundant fallback
                continue
            # Backpropagation part final
            #update the tree
            self.backpropagate(search_path , value , min_max_stats , self.config.discount)


    def select_child (self , node , min_max_stats ) :
        """
        selecting child with highest UCB score
        UCB = q(s,a) + u(s ,a )
        similiar to bellman ford eqn
        """
        max_score= - float('inf')
        best_action = None
        best_child = None
        for action , child in node.children.items():
            score = self.ucb_score(node , child , min_max_stats)
            if score > max_score :
                max_score = score
                best_action = action
                best_child = child

        # --- FIX: Fallback if no child was selected (e.g., due to NaNs) ---
        if best_child is None and node.expanded():
            # Just pick the first child so we don't crash
            best_action, best_child = list(node.children.items())[0]

        return best_action , best_child

    def ucb_score (self , parent , child , min_max_stats) :
        pb_c = math.log((parent.visit_count + 19652)/19652) + 1.25
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        #normalization
        value_score = 0
        if child.visit_count > 0 :
            value_score = min_max_stats.normalize(child.reward + self.config.discount*child.value())
        return prior_score + value_score

    def expand_node(self , node , policy_logits):
        # to prabablities softmax
        policy = torch.softmax(policy_logits , dim =1 ).squeeze(0).tolist()
        #child for each action
        for action , prob in enumerate(policy) :
            child = Node(prior=prob)
            child.action = action
            node.children[action] = child
    def backpropagate (self , search_path , value , min_max_stats , discount ) :
        """
        update the value estimates of all nodes along the path
        curr val is leaf value
        """
        if isinstance(value , torch.Tensor) :
            current_value = value.item()
        else :
            current_value = value
        #reversing path
        for node in reversed(search_path) :
            node.value_sum += current_value
            node.visit_count += 1
            #updating min max here for future normalization
            min_max_stats.update(node.value())

            current_value = node.reward + discount * current_value

