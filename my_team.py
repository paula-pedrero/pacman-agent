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

        chosen_action = random.choice(best_actions)
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
        
        #Rememeber different border entrace to use, when the usual route is guarded by an enemy ghost
        self.alternative_target = None

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
                
    def choose_action(self, game_state):
        # Choose a safer border entrance if a dangerous ghost is nearby
        my_state = game_state.get_agent_state(self.index) # Get the current state of our offensive agent
        my_pos = my_state.get_position() # Get our current position on the map

        # Reset the alternative entrance after crossing into enemy territory
        if my_state.is_pacman:
            self.alternative_target = None # Resume the normal food-collecting strategy

        # Get the states of the ghost enemies
        enemies = [
            game_state.get_agent_state(i) # Retrieve the state of an opponent using its index
            for i in self.get_opponents(game_state) # Go through all opponents on the other team
        ]
        
        # Keep only visible ghosts that can capture us.
        ghosts = [
            enemy.get_position() # Store ghost's position
            for enemy in enemies  # Examine each opponent found above
            if enemy.get_position() is not None # Ignore enemies whose exact position is unknown
            and not enemy.is_pacman # Only consider ghosts, not enemy Pacman agents
            and enemy.scared_timer == 0  # Ignore scared ghosts
        ]

        # Consider a new entrance only if we are home and a ghost is close
        if (not my_state.is_pacman # We are on our side of the map
                and self.alternative_target is None # We are not already following another entrance
                and ghosts # At least one visible dangerous ghost exists
                and min(self.get_maze_distance(my_pos, g) for g in ghosts) <= 4): # Nearest ghost is within four maze steps

            walls = game_state.get_walls()  # Get the grid of blocked and free squares
            enemy_x = walls.width // 2 if self.red else walls.width // 2 - 1 # Find the enemy-side border column

            # Build a list of border squares we could enter from our side
            entrances = [
                (enemy_x, y) # Pair the enemy border column with a possible row
                for x, y in self.home_positions # Examine each previously calculated home-side border square
                if not walls[enemy_x][y]  # Keep it only if the enemy-side square has no wall
            ]

            # Filter out entrances too close to a visible dangerous ghost
            safe_entrances = [
                pos for pos in entrances # Keep coordinates and check each accessible enemy-side border square
                if min(self.get_maze_distance(pos, g) for g in ghosts) >= 5 # Require at least five maze steps from every visible ghost
            ]

            # Remember the nearest safe entrance so the target persists next turn
            if safe_entrances:
                self.alternative_target = min(
                    safe_entrances,  # Compare only entrances that passed the safety check
                    key=lambda pos: self.get_maze_distance(my_pos, pos) # Rank candidates by maze distance from our position
                )

        return ReflexCaptureAgent.choose_action(self, game_state) # Select and return the highest-scoring action

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
            
        if action == Directions.STOP:  # Test whether this proposed action would stop movin
            features['stop'] = 1 # Mark stopping so it can penalize
        
        if self.alternative_target is not None: # Check whether an alternative entrance target is active
            features['alternative_distance'] = self.get_maze_distance( my_pos, self.alternative_target) # Measure maze distance from the proposed next position to that entrance
        
        return features

    def get_weights(self, game_state, action):
        #Get the number of food pellets we are currently carrying
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        
        if self.alternative_target is not None and carrying < 3: # Prefer the alternative entrance only when carrying fewer than 3
            return {
                'successor_score': 0,
                'distance_to_food': 0,
                'distance_to_home': 0,
                'danger': -1000,
                'stop': -20,
                'alternative_distance': -10
            }

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
            'danger': -1000,
            'stop': -20
        }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    
    def register_initial_state(self, game_state):
        ReflexCaptureAgent.register_initial_state(self, game_state)
        
        # Save the food we must defend
        self.previous_food = set(self.get_food_you_are_defending(game_state).as_list())

        # No food has been stolen yet
        self.investigation_target = None
        
        #No previus position is remembered
        self.previous_position = None

        # Find the positions where enemies can enter our territory
        walls = game_state.get_walls()
        middle = walls.width // 2

        if self.red:
            home_x = middle - 1
            enemy_x = middle
        else:
            home_x = middle
            enemy_x = middle - 1

        self.border_positions = []

        for y in range(walls.height): # Inspect every row on the separating boundary
            if not walls[home_x][y] and not walls[enemy_x][y]:
                self.border_positions.append((home_x, y))

        # Choose a central position for patrolling
        if self.border_positions:
            center_y = walls.height // 2
            self.patrol_target = min(
                self.border_positions,
                key=lambda pos: abs(pos[1] - center_y)
            )
        else:
            self.patrol_target = self.start
        
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index) # List every move our defender can legally make now

        if not actions:
            return Directions.STOP # Stay in place if there are no legal actions

        my_state = game_state.get_agent_state(self.index) # Get our defender's current state
        my_pos = my_state.get_position() # Get our defender's current coordinates

        # 1. Detect food that has disappeared
        current_food = set( # Use a set so we can compare food positions easily
            self.get_food_you_are_defending(game_state).as_list() # Read the food still present on our side
        )

        missing_food = self.previous_food - current_food # Find food positions that were present before but are gone now
        self.previous_food = current_food # Save the current food set for the next defender turn

        if missing_food:
            self.investigation_target = min( # Select one missing-food square as our new target
                missing_food,
                key=lambda pos: self.get_maze_distance(my_pos, pos) # Prefer the missing-food position nearest to us
            )

        # Stop investigating once we reach the position
        if self.investigation_target == my_pos:
            self.investigation_target = None # Clear it so we can return to patrol, unless an invader is visible

        # 2. Find visible invaders
        enemies = [
            game_state.get_agent_state(i)  # Get the current state of an opposing agent
            for i in self.get_opponents(game_state) # Repeat for both opponents
        ]

        invaders = [
            enemy for enemy in enemies # Inspect each opposing agent
            if enemy.is_pacman # An opponent is an invader
            and enemy.get_position() is not None # Only chase invaders whose position is known
        ]

        # Pick a target in priority order: visible invader, missing food, patrol
        if invaders:
            target = min( # Choose the closest visible invader
                [enemy.get_position() for enemy in invaders], # Collect the positions of visible invaders
                key=lambda pos: self.get_maze_distance(my_pos, pos) # Compare their maze distances from us
            )

        elif self.investigation_target is not None: # No invader is visible, but a theft left a clue
            target = self.investigation_target # Head toward the last selected missing-food square

        else:
            if my_pos == self.patrol_target and len(self.border_positions) > 1: # Change goal after reaching it if another border entrance exists
                self.patrol_target = max(self.border_positions, key=lambda pos: self.get_maze_distance(my_pos, pos) ) # Choose the farthest border entrance
            target = self.patrol_target

        is_patrolling = (not invaders) and (self.investigation_target is None) # True only while neither chasing nor investigating.
        
        # 4. Evaluate possible actions
        values = [] # Keep one final score for each action in 'actions'

        for action in actions: # Evaluate every possible action
            # Keep the original evaluation function
            value = self.evaluate(game_state, action)

            successor = self.get_successor(game_state, action) # Simulate the game state after this action
            next_state = successor.get_agent_state(self.index) # Get our defender's simulated state
            next_pos = next_state.get_position() # Get the square reached by that simulated move
            
            if is_patrolling and self.previous_position is not None: # Apply this rule only in patrol mode and when a previous position exists
                if next_pos == self.previous_position: # Avoid repeating a loop between two positions
                    value -= 20

            # Stay in our own territory
            if next_state.is_pacman:
                value -= 1000

            # Move towards the selected target when there is no invader
            if not invaders:
                distance = self.get_maze_distance(next_pos, target)
                value -= 10 * distance

            # Avoid enemy Pacman when scared
            if my_state.scared_timer > 0 and invaders: # Apply this only if our ghost is scared and invaders are visibl
                min_distance = min( # Find the closest invader from the candidate next square
                    self.get_maze_distance(next_pos, enemy.get_position())  # Measure maze distance to one visible invader
                    for enemy in invaders # Compare all visible invaders
                )

                if min_distance <= 1:
                    value -= 1000
                elif min_distance == 2:
                    value -= 100

            values.append(value) # Store this action's final score in the same order as 'actions'

        # 5. Select the best action
        max_value = max(values)

        best_actions = [
            action for action, value in zip(actions, values) # Keep the legal action and pair it with their value
            if value == max_value
        ]
        
        self.previous_position = my_pos # Remember today's position to detect an immediate return next turn
        
        return random.choice(best_actions) # Break ties randomly

        
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
