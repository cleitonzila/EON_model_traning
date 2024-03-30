import gym
from gym import spaces
import numpy as np
import networkx as nx
from itertools import islice
import torch
import simpy
from stable_baselines3 import PPO

TOPOLOGY = 'JPN12'
N_PATH = 1
SLOTS = 16
MAX_DEMAND = 6
MAX_SLICE = 5

class OpticalNetworkEnv(gym.Env):
    def __init__(self, traffic):
        super(OpticalNetworkEnv, self).__init__()
        self.env = simpy.Environment()
        self.requests = self.read_requests(traffic)
        self.current_request_index = 0
        self.action_space = spaces.MultiDiscrete([MAX_SLICE] + [SLOTS] + [MAX_DEMAND - 2, SLOTS] * MAX_SLICE)
        self.total_requests = 0
        self.total_success = 0
        self.total_slots_fails = 0
        self.total_slots = 0
        self.total_duration = 0
        self.total_holding_time = 0
        self.allocations = {}  # Dictionary to track allocations (request ID to path, start slot, and number of slots)
        self.topology = nx.read_weighted_edgelist('topology/' + TOPOLOGY, nodetype=int)
        self.link_num = len(self.topology.edges)
        self.node_num = len(self.topology.nodes)

        # Limits for order details
        self.request_limits = [
            self.node_num,      # source
            self.node_num,      # destination
            MAX_DEMAND + 1         # num_slots
        ]

        self.network_space = spaces.MultiBinary([self.link_num, SLOTS])
        self.request_space = spaces.MultiDiscrete(self.request_limits)
        # Combining both in a single observation space
        self.observation_space = spaces.Dict({
            "network_state": self.network_space,
            "current_request": self.request_space
        })
        for u, v in self.topology.edges:
            self.topology[u][v]['capacity'] = [0] * SLOTS 

    def reset(self):
        self.current_request_index = 0
        self.total_requests = 0
        self.total_success = 0
        self.total_slots_fails = 0
        self.total_holding_time = 0
        self.total_slots = 0
        self.total_duration = 0
        self.env = simpy.Environment()
        for u, v in self.topology.edges:
            self.topology[u][v]['capacity'] = [0] * SLOTS
        initial_state = self.get_observation()
        return initial_state
    
    def step(self, action):
        # Geting request information
        request = self.requests[self.current_request_index]
        request_id, source, destination, num_slots, start_time, duration = request
        self.total_holding_time += duration
        # Finds the best path(s) for the request
        path = self.k_shortest_paths(self.topology, source, destination, N_PATH)[0]
        
        state = self.get_observation()
        done = (self.current_request_index >= len(self.requests)) 

        num_slices = action[0]
        # Creating arrays with the slicing possition and allocation possition


        if self.current_request_index >= len(self.requests):
            return self.get_observation(), 0, True, {}  

        if num_slices > 0:
            allocation_positions = [action[1]]
            cut_positions = []
            for i in range(1, num_slices + 1):
                cut_positions.append(action[i*2] + 1) 
                allocation_positions.append(action[i*2 + 1])

            cut_positions.sort()
            allocation_positions.sort()

            # Creating slices to try to allocate:
            allocation_slices = self.divide_slots(num_slots, cut_positions)

            sets = []
            for i in range(len(allocation_slices)):
                sets.append(set(range(allocation_positions[i], allocation_positions[i] + allocation_slices[i])))

            # Validating the position and cut arrays:
            if((len(cut_positions) > len(set(cut_positions)) or (max(cut_positions) > num_slots)) or (self.overlaps(sets))):
                return state, 0, done, {}
        
            for i in range(len(allocation_slices)):
                if(not (self.allocation_posible(path, allocation_slices[i], allocation_positions[i]))):
                    return state, 0, done, {}
            
            for i in range(len(allocation_slices)):
                self.env.process(self.allocate_path(str(request_id) + "_"+ str(i), path, allocation_slices[i], allocation_positions[i], start_time))
                self.env.process(self.release_path(str(request_id) + "_" + str(i), start_time, duration))
                
            reward = (1 - num_slices*(1/(MAX_SLICE+1)))
        else:
            if(self.allocation_posible(path, num_slots, action[1])):
                self.env.process(self.allocate_path(str(request_id), path, num_slots, action[1], start_time))
                self.env.process(self.release_path(str(request_id), start_time, duration))
                reward = 1
            else:
                return state, 0, done, {}

        self.total_success += 1
        self.env.run(until=start_time+1)
        self.current_request_index += 1  
        self.total_requests += 1
        done = (self.current_request_index >= len(self.requests)) 
        state = self.get_observation()
        return state, reward, done, {}

    def k_shortest_paths(self, G, source, target, k, weight='weight'):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    def get_observation(self):
        # Get network status
        network_state = self.get_network_state()
    
        # Get the details of the current order
        if self.current_request_index < len(self.requests):
            request = self.requests[self.current_request_index]
            request_id, source, destination, num_slots, start_time, duration = request
            request_details = [source, destination, num_slots]
        else:
            request_details = [0, 0, 0, 0, 0]  

        # Create a dictionary to represent the complete state
        state = {
            "network_state": network_state,
            "current_request": request_details
        }
        return state

    def get_network_state(self):
        network_state = []
        for u, v in self.topology.edges:
            network_state.append(self.topology[u][v]['capacity'])
        return(network_state)
    
    def allocation_posible(self, path, num_slots, start_slot):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if start_slot + num_slots > len(self.topology[u][v]['capacity']):
                return False 

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if any(self.topology[u][v]['capacity'][start_slot:start_slot + num_slots]):
                return False  

        return True  

    def allocate_path(self, request_id, path, num_slots, start_slot, start_time):
        yield self.env.timeout(start_time)

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for slot in range(start_slot, start_slot + num_slots):
                self.topology[u][v]['capacity'][slot] = 1

        self.allocations[request_id] = (path, start_slot, num_slots) 

    def release_path(self, request_id, start_time, duration):
        yield self.env.timeout(start_time + duration)

        if request_id in self.allocations:
            path, start_slot, num_slots = self.allocations[request_id]

            # Iterate through each link in the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]  # Get the nodes representing the link
                # Free up the allocated slots on the link
                for slot in range(start_slot, start_slot + num_slots):
                    self.topology[u][v]['capacity'][slot] = 0

            # Remove the allocation record from the dictionary
            del self.allocations[request_id]
        else:
            # Print an error message if the request ID is not found in the allocations
            print(f"Error releasing request: Request {request_id} or slots not found.")
    
    def read_requests(self, filename):
        # Open and read the file
        request = []
        with open(filename, 'r') as file:
            for line in file.readlines():
                # Split each line into parts
                parts = line.strip().split()
                # Check if the line has the correct number of parts (6 parts including the ':')
                if len(parts) == 6:  
                    # Extract and convert the parts of the line to appropriate types
                    request_id = int(parts[0].replace(":", ""))  # Request ID
                    source = int(parts[1])                       # Source node
                    destination = int(parts[2])                  # Destination node
                    num_slots = int(parts[3])                    # Number of slots required
                    start_time = int(parts[4])                   # Start time of the request
                    duration = int(parts[5])                     # Duration of the request

                    # Process the request with extracted information
                    request.append([request_id, source, destination, num_slots, start_time, duration])
                else:
                    # Print an error message for incorrectly formatted lines
                    print(f"Malformatted line: {line.strip()}")
        return (request)

    def divide_slots(self, slot_size, cut_positions):
        # Add the slot_size to the end of the cut_positions to make it easier to calculate the last part
        cut_positions_adjusted = cut_positions + [slot_size]
        
        # Start with 0 to calculate the first part correctly
        previous_cut = 0
        
        # Array to store the size of the parts after the cuts
        parts = []
        
        # Iterates through the set cutting positions
        for cut in cut_positions_adjusted:
            # Calculates the size of the current part and adds it to the parts array
            # The difference between the current cutting position and the previous one gives the size of the part
            parts.append(cut - previous_cut)
            
            # Updates the previous cut position for the next iteration
            previous_cut = cut
        
        return parts

    def overlaps(self, sets):
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if sets[i].intersection(sets[j]):
                    return True
        return False

    def print_statistics(self):
        bbr = (self.total_slots_fails/self.total_slots)  
        elang = (self.total_holding_time/self.total_duration)  
        return([bbr, elang])

#Traning
traffic = 10
"""
try:
    simulator = OpticalNetworkEnv("traffic_jpn12_traning_slicing" + str(traffic) + ".txt")
    model = PPO("MultiInputPolicy", simulator, verbose=2)
    total_timesteps = 10000 
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_optical_network_with_slicing"+ str(traffic))
except Exception as e:
    torch.cuda.empty_cache()
    print(f"Ocorreu um erro: {e}")
finally:
    torch.cuda.empty_cache()


#Testing
"""
"""
bbr_values = {}
for i in range(10, 110, 10):
    bbr = []
    elang = []
    model = PPO.load("ppo_optical_network_with_slicing"+ str(i))
    try:
        
        for trafic in range(1, 10):
            simulator = OpticalNetworkEnv("traffic_jpn12_testing_slicing" + str(trafic) + ".txt")
            obs = simulator.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic = True)
                obs, reward, done, info = simulator.step(action)
            bbr.append(simulator.print_statistics()[0])
            elang.append(simulator.print_statistics()[1])
            simulator.close()
            torch.cuda.empty_cache()
        
        for trafic in range(10, 110, 10):
            simulator = OpticalNetworkEnv("traffic_jpn12_testing_slicing" + str(trafic) + ".txt")
            obs = simulator.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic = True)
                obs, reward, done, info = simulator.step(action)
            bbr.append(simulator.print_statistics()[0])
            elang.append(simulator.print_statistics()[1])
            simulator.close()
            torch.cuda.empty_cache()

    except Exception as e:
        torch.cuda.empty_cache()
        print(f"Ocorreu um erro: {e}")
    finally:
        torch.cuda.empty_cache()
    bbr_values[i] = bbr
    print("Model " +str(i)+ " done")
print(elang)
print(bbr_values)
"""


bbr_values = {}

bbr = []
elang = []
model = PPO.load("ppo_optical_network_with_slicing"+ str(traffic))
try:
        
    for trafic in range(1, 10):
        simulator = OpticalNetworkEnv("traffic_jpn12_testing_slicing" + str(trafic) + ".txt")
        obs = simulator.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, info = simulator.step(action)
        bbr.append(simulator.print_statistics()[0])
        elang.append(simulator.print_statistics()[1])
        simulator.close()
        print(f"{traffic} done")
        torch.cuda.empty_cache()
    
    for trafic in range(10, 110, 10):
        simulator = OpticalNetworkEnv("traffic_jpn12_testing_slicing" + str(trafic) + ".txt")
        obs = simulator.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, info = simulator.step(action)
        bbr.append(simulator.print_statistics()[0])
        elang.append(simulator.print_statistics()[1])
        simulator.close()
        print(f"{traffic} done")
        torch.cuda.empty_cache()

except Exception as e:
    torch.cuda.empty_cache()
    print(f"An error occurred: {e}")
    print(action)
finally:
    torch.cuda.empty_cache()
bbr_values[traffic] = bbr
print(elang)
print(bbr_values)