"""
On-Demand Sleep-Based Aloha Simulator for M2M Communication

This simulator implements the framework described in the paper:
"On-Demand-Sleep-Based Aloha for M2M Communication: Modeling, Optimization, 
and Tradeoff Between Lifetime and Delay"

Features:
- Slotted Aloha protocol with collision detection
- On-demand sleep mechanism (buffer-based sleep)
- Bernoulli packet arrivals
- Energy consumption tracking
- Performance metrics: lifetime, delay, throughput
"""

import simpy
import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Packet:
    """Represents a data packet in the network"""
    size: int  # Packet size in bytes
    src: str  # Source node name
    dest: str  # Destination (receiver)
    arrival_time: float  # Slot when packet arrived at buffer
    packet_id: int = 0  # Unique packet identifier
    
    def __repr__(self):
        return f"Packet(id={self.packet_id}, src={self.src}, arrival={self.arrival_time})"


class Node:
    """
    Represents a Machine-Type Device (MTD) in the M2M network.
    
    States:
    - active: Node has packets and is actively trying to transmit
    - idle: Buffer is empty, waiting for ts slots before sleeping
    - sleep: Deep sleep mode (very low power)
    - wakeup: Transitioning from sleep to active (takes tw slots)
    """
    
    def __init__(self, env: simpy.Environment, name: str, 
                 lambda_arrival: float, q_transmit: float, 
                 ts_idle: int, tw_wakeup: int, E_initial: float,
                 PS_sleep: float, PW_wakeup: float, 
                 PT_transmit: float, PB_busy: float,
                 simulator=None):
        # SimPy environment
        self.env = env
        self.name = name
        self.simulator = simulator
        
        # Buffer and state
        self.buffer = deque()  # Infinite queue for packets
        self.state = 'active'  # Initial state
        self.idle_timer = 0  # Countdown timer for idle->sleep transition
        self.wakeup_timer = 0  # Countdown timer for wakeup duration
        
        # Protocol parameters
        self.lambda_arrival = lambda_arrival  # Bernoulli arrival rate (prob per slot)
        self.q_transmit = q_transmit  # Transmission probability per slot
        self.ts_idle = ts_idle  # Idle timeout (slots before sleep)
        self.tw_wakeup = tw_wakeup  # Wake-up duration (slots)
        
        # Energy parameters
        self.E_initial = E_initial
        self.energy = E_initial  # Current remaining energy
        self.PS = PS_sleep  # Power consumption in sleep
        self.PW = PW_wakeup  # Power consumption during wake-up
        self.PT = PT_transmit  # Power consumption when transmitting
        self.PB = PB_busy  # Power consumption when active/idle but not transmitting
        
        # Metrics tracking
        self.packets_sent = 0  # Successfully transmitted packets
        self.packets_arrived = 0  # Total packets arrived
        self.total_delay = 0.0  # Sum of queueing delays
        self.lifetime = None  # Time when energy depleted
        self.total_transmit_attempts = 0  # Total transmission attempts
        self.collisions = 0  # Number of collisions experienced
        self.packet_id_counter = 0  # For unique packet IDs
        
        # State history (for debugging/analysis)
        self.state_history = []
        self.energy_history = []
        
    def generate_traffic(self):
        """
        Traffic generation process: Bernoulli arrivals.
        In each slot, a new packet arrives with probability lambda_arrival.
        """
        while self.energy > 0:
            # Bernoulli arrival
            if random.random() < self.lambda_arrival:
                packet = Packet(
                    size=1000,  # Fixed packet size
                    src=self.name,
                    dest='receiver',
                    arrival_time=self.env.now,
                    packet_id=self.packet_id_counter
                )
                self.packet_id_counter += 1
                self.packets_arrived += 1
                self.buffer.append(packet)
                
                # If sleeping, trigger wake-up
                if self.state == 'sleep':
                    self.start_wakeup()
                # If idle, cancel idle timer and become active
                elif self.state == 'idle':
                    self.state = 'active'
                    self.idle_timer = 0
                    
            yield self.env.timeout(1)  # Advance one slot
            
    def start_wakeup(self):
        """Initiate wake-up transition from sleep"""
        self.state = 'wakeup'
        self.wakeup_timer = self.tw_wakeup
        
    def process_slot(self):
        """
        Main slot processing loop.
        Called every slot to handle state transitions, transmissions, and energy consumption.
        """
        while True:
            # Check if energy depleted
            if self.energy <= 0:
                if self.lifetime is None:
                    self.lifetime = self.env.now
                    self.state = 'dead'
                yield self.env.timeout(1)
                continue
                
            # Record state for history
            self.state_history.append((self.env.now, self.state))
            
            power_consumed = 0
            will_transmit = False
            
            # State machine
            if self.state == 'sleep':
                power_consumed = self.PS
                # Wake-up is triggered by packet arrival (in generate_traffic)
                
            elif self.state == 'wakeup':
                power_consumed = self.PW
                self.wakeup_timer -= 1
                if self.wakeup_timer <= 0:
                    self.state = 'active'
                    
            elif self.state == 'active':
                if len(self.buffer) > 0:
                    # Node has packets: decide whether to transmit
                    if random.random() < self.q_transmit:
                        power_consumed = self.PT
                        will_transmit = True
                    else:
                        power_consumed = self.PB  # Active but not transmitting
                else:
                    # Buffer empty: transition to idle
                    self.state = 'idle'
                    self.idle_timer = self.ts_idle
                    power_consumed = self.PB
                    
            elif self.state == 'idle':
                power_consumed = self.PB
                if len(self.buffer) > 0:
                    # New packet arrived: back to active
                    self.state = 'active'
                    self.idle_timer = 0
                else:
                    # Countdown idle timer
                    self.idle_timer -= 1
                    if self.idle_timer <= 0:
                        self.state = 'sleep'
                        
            # Consume energy
            self.energy -= power_consumed
            self.energy_history.append((self.env.now, self.energy))
            
            # Register transmission attempt with simulator
            if will_transmit and len(self.buffer) > 0:
                self.total_transmit_attempts += 1
                self.simulator.register_transmission(self)
                
            yield self.env.timeout(1)  # Advance one slot
            
    def handle_transmission_result(self, success: bool):
        """
        Called by simulator to notify transmission result.
        If successful, remove HOL packet from buffer and record delay.
        """
        if success and len(self.buffer) > 0:
            sent_packet = self.buffer.popleft()
            delay = self.env.now - sent_packet.arrival_time
            self.total_delay += delay
            self.packets_sent += 1
        elif not success:
            self.collisions += 1
            
    def get_avg_delay(self) -> float:
        """Calculate average queueing delay"""
        return self.total_delay / self.packets_sent if self.packets_sent > 0 else 0
    
    def get_throughput(self, sim_time: float) -> float:
        """Calculate throughput (packets/slot)"""
        return self.packets_sent / sim_time if sim_time > 0 else 0


class Simulator:
    """
    Main simulator class that manages all nodes and the shared channel.
    Handles collision detection for slotted Aloha.
    """
    
    def __init__(self, env: simpy.Environment, n_nodes: int,
                 lambda_arrival: float, q_transmit: float,
                 ts_idle: int, tw_wakeup: int, E_initial: float,
                 PS: float, PW: float, PT: float, PB: float,
                 simulation_time: int, seed: Optional[int] = None):
        self.env = env
        self.simulation_time = simulation_time
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Create nodes
        self.nodes = []
        for i in range(n_nodes):
            node = Node(
                env=env,
                name=f'Node_{i}',
                lambda_arrival=lambda_arrival,
                q_transmit=q_transmit,
                ts_idle=ts_idle,
                tw_wakeup=tw_wakeup,
                E_initial=E_initial,
                PS_sleep=PS,
                PW_wakeup=PW,
                PT_transmit=PT,
                PB_busy=PB,
                simulator=self
            )
            self.nodes.append(node)
            
        # Per-slot transmission tracking
        self.transmitting_nodes_current_slot = []
        
        # Global metrics
        self.total_slots_simulated = 0
        self.total_successful_transmissions = 0
        self.total_collisions = 0
        
    def register_transmission(self, node: Node):
        """Called by nodes when they attempt transmission in current slot"""
        self.transmitting_nodes_current_slot.append(node)
        
    def slot_coordinator(self):
        """
        Global slot coordinator that runs at the end of each slot
        to resolve collisions and notify nodes of results.
        """
        while True:
            yield self.env.timeout(0.99)  # Run just before slot boundary
            
            # Resolve transmissions for this slot
            num_transmitters = len(self.transmitting_nodes_current_slot)
            
            if num_transmitters == 1:
                # Success: exactly one transmitter
                self.transmitting_nodes_current_slot[0].handle_transmission_result(True)
                self.total_successful_transmissions += 1
            elif num_transmitters > 1:
                # Collision: multiple transmitters
                for node in self.transmitting_nodes_current_slot:
                    node.handle_transmission_result(False)
                self.total_collisions += 1
                
            # Clear for next slot
            self.transmitting_nodes_current_slot = []
            self.total_slots_simulated += 1
            
            yield self.env.timeout(0.01)  # Complete the slot
            
    def run(self):
        """Start all processes and run simulation"""
        # Start processes for each node
        for node in self.nodes:
            self.env.process(node.generate_traffic())
            self.env.process(node.process_slot())
            
        # Start slot coordinator
        self.env.process(self.slot_coordinator())
        
        # Run simulation
        self.env.run(until=self.simulation_time)
        
    def collect_metrics(self) -> Dict:
        """
        Collect and aggregate metrics from all nodes after simulation.
        Returns dictionary with performance metrics.
        """
        # Per-node metrics
        lifetimes = [node.lifetime if node.lifetime else self.simulation_time 
                     for node in self.nodes]
        delays = [node.get_avg_delay() for node in self.nodes]
        throughputs = [node.get_throughput(self.simulation_time) for node in self.nodes]
        
        # Aggregate metrics
        total_packets_sent = sum(node.packets_sent for node in self.nodes)
        total_packets_arrived = sum(node.packets_arrived for node in self.nodes)
        total_attempts = sum(node.total_transmit_attempts for node in self.nodes)
        total_node_collisions = sum(node.collisions for node in self.nodes)
        
        # Energy metrics
        avg_energy_remaining = np.mean([node.energy for node in self.nodes])
        energy_consumed_ratio = 1 - (avg_energy_remaining / self.nodes[0].E_initial)
        
        metrics = {
            # Lifetime metrics
            'avg_lifetime': np.mean(lifetimes),
            'min_lifetime': np.min(lifetimes),
            'max_lifetime': np.max(lifetimes),
            'std_lifetime': np.std(lifetimes),
            
            # Delay metrics
            'avg_delay': np.mean([d for d in delays if d > 0]),
            'max_delay': np.max([d for d in delays if d > 0]) if any(d > 0 for d in delays) else 0,
            
            # Throughput metrics
            'total_throughput': total_packets_sent / self.simulation_time,
            'avg_node_throughput': np.mean(throughputs),
            
            # Transmission statistics
            'total_packets_sent': total_packets_sent,
            'total_packets_arrived': total_packets_arrived,
            'packet_delivery_ratio': total_packets_sent / total_packets_arrived if total_packets_arrived > 0 else 0,
            'total_transmission_attempts': total_attempts,
            'total_collisions': self.total_collisions,
            'collision_rate': self.total_collisions / total_attempts if total_attempts > 0 else 0,
            
            # Energy metrics
            'avg_energy_remaining': avg_energy_remaining,
            'energy_consumed_ratio': energy_consumed_ratio,
            
            # System metrics
            'n_nodes': len(self.nodes),
            'simulation_time': self.simulation_time,
            'lambda_arrival': self.nodes[0].lambda_arrival,
            'q_transmit': self.nodes[0].q_transmit,
            'ts_idle': self.nodes[0].ts_idle,
            'tw_wakeup': self.nodes[0].tw_wakeup,
        }
        
        return metrics
    
    def get_detailed_node_metrics(self) -> List[Dict]:
        """Get detailed metrics for each individual node"""
        node_metrics = []
        for node in self.nodes:
            metrics = {
                'name': node.name,
                'lifetime': node.lifetime if node.lifetime else self.simulation_time,
                'packets_sent': node.packets_sent,
                'packets_arrived': node.packets_arrived,
                'avg_delay': node.get_avg_delay(),
                'throughput': node.get_throughput(self.simulation_time),
                'energy_remaining': node.energy,
                'total_attempts': node.total_transmit_attempts,
                'collisions': node.collisions,
                'final_state': node.state,
                'buffer_size': len(node.buffer),
            }
            node_metrics.append(metrics)
        return node_metrics


if __name__ == "__main__":
    # Example usage with default parameters
    print("=" * 70)
    print("On-Demand Sleep-Based Aloha Simulator")
    print("=" * 70)
    
    # Simulation parameters
    N_NODES = 10
    LAMBDA_ARRIVAL = 0.1  # Packet arrival probability per slot
    Q_TRANSMIT = 0.05  # Transmission probability
    TS_IDLE = 5  # Idle timeout (slots)
    TW_WAKEUP = 2  # Wake-up duration (slots)
    E_INITIAL = 10000  # Initial energy
    PS = 0.1  # Sleep power
    PW = 1.0  # Wake-up power
    PT = 5.0  # Transmit power
    PB = 0.5  # Busy/idle power
    SIM_TIME = 5000  # Simulation time (slots)
    SEED = 42  # Random seed
    
    print(f"\nSimulation Parameters:")
    print(f"  Number of nodes: {N_NODES}")
    print(f"  Arrival rate (lambda): {LAMBDA_ARRIVAL}")
    print(f"  Transmission prob (q): {Q_TRANSMIT}")
    print(f"  Idle timeout (ts): {TS_IDLE} slots")
    print(f"  Wake-up time (tw): {TW_WAKEUP} slots")
    print(f"  Initial energy: {E_INITIAL}")
    print(f"  Simulation time: {SIM_TIME} slots")
    print(f"\nRunning simulation...")
    
    # Create and run simulation
    env = simpy.Environment()
    sim = Simulator(
        env=env,
        n_nodes=N_NODES,
        lambda_arrival=LAMBDA_ARRIVAL,
        q_transmit=Q_TRANSMIT,
        ts_idle=TS_IDLE,
        tw_wakeup=TW_WAKEUP,
        E_initial=E_INITIAL,
        PS=PS, PW=PW, PT=PT, PB=PB,
        simulation_time=SIM_TIME,
        seed=SEED
    )
    
    sim.run()
    
    # Collect and display metrics
    metrics = sim.collect_metrics()
    
    print(f"\n{'=' * 70}")
    print("Simulation Results")
    print(f"{'=' * 70}")
    print(f"\nLifetime Metrics:")
    print(f"  Average lifetime: {metrics['avg_lifetime']:.2f} slots")
    print(f"  Min lifetime: {metrics['min_lifetime']:.2f} slots")
    print(f"  Max lifetime: {metrics['max_lifetime']:.2f} slots")
    
    print(f"\nDelay Metrics:")
    print(f"  Average delay: {metrics['avg_delay']:.2f} slots")
    print(f"  Max delay: {metrics['max_delay']:.2f} slots")
    
    print(f"\nThroughput Metrics:")
    print(f"  Total throughput: {metrics['total_throughput']:.4f} packets/slot")
    print(f"  Avg node throughput: {metrics['avg_node_throughput']:.4f} packets/slot")
    
    print(f"\nTransmission Statistics:")
    print(f"  Packets sent: {metrics['total_packets_sent']}")
    print(f"  Packets arrived: {metrics['total_packets_arrived']}")
    print(f"  Delivery ratio: {metrics['packet_delivery_ratio']:.2%}")
    print(f"  Total attempts: {metrics['total_transmission_attempts']}")
    print(f"  Collisions: {metrics['total_collisions']}")
    print(f"  Collision rate: {metrics['collision_rate']:.2%}")
    
    print(f"\nEnergy Metrics:")
    print(f"  Avg energy remaining: {metrics['avg_energy_remaining']:.2f}")
    print(f"  Energy consumed: {metrics['energy_consumed_ratio']:.2%}")
    
    print(f"\n{'=' * 70}")
