from collections import namedtuple
import os
import time

import networkx as nx
import simpy
import pexpect
from dotenv import load_dotenv
import torch

load_dotenv()

class Simulator():
    def __init__(self, graph, comm_cost, comp_cost):
        self.graph = graph
        self.comm_cost = comm_cost
        self.comp_cost = comp_cost
    
    def get_cost(self, dev_assign):

        G = self.graph.to_networkx().to_directed()
        nx.set_node_attributes(G, {i: {'start_time': [], 'end_time': [], 'comp_time': [] } for i in G.nodes})
        nx.set_edge_attributes(G, {i: {'arrive_time': [], 'comm_time': []} for i in G.edges})

        Op = namedtuple('Op', 'id inputs outputs comp_time dev')


        def communicate(env, finish_event, e, time):
            # print(f"{env.now: .2f}: Op {e1} --> Op {e2} for {time}")
            yield finish_event
            yield env.timeout(time)
            G.edges[e]['arrive_time'].append(env.now)
            G.edges[e]['comm_time'].append(time)
            return env.now

        def compute(env, op):
            times = (yield simpy.events.AllOf(env, op.inputs.values())).values()
            times = [A.item() for A in times]
            times = dict(zip(op.inputs.keys(), times))

            with op.dev.request() as req:  # Generate a request event
                yield req                    # Wait for access
                G.nodes[op.id]['start_time'].append(env.now)
                yield env.timeout(op.comp_time)
                G.nodes[op.id]['end_time'].append(env.now)
                G.nodes[op.id]['comp_time'].append(op.comp_time)
            # print(f"{self.env.now: .2f}: Op {o} --> Dev {des} finished. Running {self.comp_time: .2f}")

            op.outputs.succeed()

        env = simpy.Environment()
        comm_events = {}
        finish_event = {}

        for o in G.nodes:
            finish_event[o] = env.event()

        for e in G.edges:
            d1 = dev_assign[e[0]]
            d2 = dev_assign[e[1]]
            comm_events[e[:2]] = env.process(communicate(env, finish_event[e[0]], e, self.communication_latency(e[0], e[1], d1, d2)))

        ops = {}
        processes = {}
        devices = {d: simpy.Resource(env) for d in list(set(dev_assign))}

        for o in G.nodes:
            inputs = {e: comm_events[e] for e in G.in_edges(o)}
            outputs = finish_event[o]
            des = dev_assign[o]
            ops[o] = Op(o, inputs, outputs, self.computation_latency(o, des), devices[des])
            processes[o] = env.process(compute(env, ops[o]))

        while env.peek() < float('inf'):
            env.step()
        
        return env.now.clone().detach()/1e10
    
    def communication_latency(self, op1, op2, dev1, dev2):
        if dev1 == dev2:
            return 0
        else:
            a = self.comm_cost[op1]
            return a
        
    def computation_latency(self, op, dev):
        a = self.comp_cost[op]
        return a

class RealExecuteEngine():
    def __init__(self):
        self.cpp_file_path = "system_running_time.txt"
        self.python_file_path = "rl_assignment.txt"
        
    
    def get_cost(self, curr_assignment):
        self.write_rl_assignment_to_file(curr_assignment)
        running_time = self.get_system_cost()
        print("Receive system running time: ", running_time)
        
        return torch.tensor(running_time)
    
    def write_rl_assignment_to_file(self, loc_assignment):
        REMOTE_USER = os.getenv("REMOTE_USER")
        REMOTE_IP = os.getenv("REMOTE_IP")
        REMOTE_PATH = os.getenv("REMOTE_PATH")
        PASSWORD = os.getenv("PASSWORD")

        loc_str = [str(item) for item in loc_assignment]
        rl_assignment = ",".join(loc_str)
        
        with open(self.python_file_path, "w") as f:
            f.write(rl_assignment)
        
        print("REMOTE_USER: ", REMOTE_USER)
        print("REMOTE_IP: ", REMOTE_IP)
        self.send_file(self.python_file_path, REMOTE_USER, REMOTE_IP, REMOTE_PATH, PASSWORD)
    
    def send_file(self, local_path, remote_user, remote_ip, remote_path, password):
        command = f"scp {local_path} {remote_user}@{remote_ip}:{remote_path}"
        child = pexpect.spawn(command)

        # Look for the password prompt and send the password
        child.expect("password:")
        child.sendline(password)
        child.expect(pexpect.EOF)  # Wait for the command to finish
    
    def get_system_cost(self):
        # Wait for the file from C++ node
        self.wait_for_file(self.cpp_file_path)

        # Read and process the file
        with open(self.cpp_file_path, "r") as f:
            content = f.read()

        # Remove the file
        os.remove(self.cpp_file_path)

        return int(content)

    def wait_for_file(self, file_path):
        """Wait until the file exists."""
        while not os.path.exists(file_path):
            time.sleep(3)