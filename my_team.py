# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def register_initial_state(self,game_state):
        #Register initial state of the agent
        ReflexCaptureAgent.register_initial_state(self,game_state)

        #Get the walls of the map and find its middle column
        walls = game_state.get_walls()
        middle= walls.width // 2

        #Create a list to store all possible positions on our home border
        self.home_positions = []

        #find the column that belongs to our team's home side
        if self.red:
            home_x = middle - 1
        else:
            home_x = middle

        #Check every position along the home border
        for y in range(walls.height):
            #Only add positions that are not walls
            if not walls[home_x][y]:
                self.home_positions.append((home_x, y))

    def get_features(self, game_state, action):
        features = util.Counter() #features of this action
        successor = self.get_successor(game_state, action) #game state after performing an action
        food_list = self.get_food(successor).as_list() #food remaining on opponent side
        my_state = successor.get_agent_state(self.index) #info abount agent after the action
        my_pos = my_state.get_position() #get our agent's position after the action
        features['successor_score'] = -len(food_list)  # Feature 1: reward actions that reduce the amount of food remaining

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        #Feature 3: Calculate the distance to the nearest position on our home border
        
        #Check that there are available positions on the home border
        if len(self.home_positions) > 0:
            #calculate the distance from Pacman to every possible home position
            distances = [
                self.get_maze_distance(my_pos, home_pos)
                for home_pos in self.home_positions
            ]
            #Store the shortest distance as our distance to home
            features['distance_to_home'] = min(distances)


        #Feature 4: avoid dangerous enemies
        enemies = [
            successor.get_agent_state(i)
            for i in self.get_opponents(successor)
        ]
        #only consider visible enemies that are ghosts and are not scared
        dangerous_enemies = [
            enemy for enemy in enemies
            if enemy.get_position() is not None
            and not enemy.is_pacman
            and enemy.scared_timer == 0
        ]
        #Find the distance to the closest dangerous enemy
        if len(dangerous_enemies) > 0:
            enemy_distances = [
                self.get_maze_distance(my_pos, enemy.get_position())
                for enemy in dangerous_enemies
            ]
            min_enemy_distance = min(enemy_distances)

            #give a higher danger value when an anemy is closer
            if my_state.is_pacman:
                features['danger'] = max(0, 4 - min_enemy_distance)
            else:
                features['danger'] = 0
        else:
            features['danger'] = 0
        return features

    def get_weights(self, game_state, action):
        #Get the number of food pellets we are currently carrying
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying

        #If we are carrying at least 3 pellets, prioritize returning home
        if carrying >= 3:
            return {
                'successor_score': 0,
                'distance_to_food' : 0, #food no longer a priority
                'distance_to_home': -5, #returning home becomes more important
                'danger': -1000
            }
        #Otherwise, prioritize collecting food
        return {
            'successor_score': 100,
            'distance_to_food' : -1,
            'distance_to_home': -0.2,
            'danger': -1000
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
