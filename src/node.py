"""
MTD (Machine-Type Device) Node Implementation for Sleep-Based Random Access Simulator

This module implements the Node class representing individual MTD devices in the 
sleep-based low-latency access simulation for M2M communications.

Date: February 10, 2026
"""

from collections import deque
from enum import Enum
import random
from typing import Optional, Dict, List, Tuple


class NodeState(Enum):
    """States that an MTD node can be in"""
    ACTIVE = "active"      # Node has packets and is actively contending
    IDLE = "idle"          # Node has no packets, idle timer running
    SLEEP = "sleep"        # Node in sleep mode (lowest power)
    WAKEUP = "wakeup"      # Node in wake-up transition


class Node:
    """
    Represents a single Machine-Type Device (MTD) node in the M2M network.
    
    The node implements on-demand sleep with slotted Aloha random access:
    - ACTIVE: Has packets in queue, contends for transmission
    - IDLE: No packets, idle timer (ts) running before sleep
    - SLEEP: Deep sleep mode for energy conservation
    - WAKEUP: Transitioning from sleep to active (takes tw slots)
    
    Attributes:
        node_id (int): Unique identifier for this node
        state (NodeState): Current state of the node
        queue (deque): Queue of packets with arrival times (slot number)
        energy (float): Remaining energy in units
        idle_timer (int): Idle timer value (ts parameter)
        wakeup_counter (int): Wake-up time value (tw parameter)
        current_idle_count (int): Current idle timer countdown
        current_wakeup_count (int): Current wake-up timer countdown
        
        # Statistics tracking
        total_delay (float): Sum of all packet delays (for mean calculation)
        packets_delivered (int): Count of successfully delivered packets
        packets_arrived (int): Count of total packets arrived
        energy_consumed_by_state (Dict): Energy consumed per state
        time_in_state (Dict): Time spent in each state (slots)
        delays (List[float]): List of individual packet delays for analysis
    """
    
    def __init__(
        self,
        node_id: int,
        initial_energy: float,
        idle_timer: int,
        wakeup_time: int,
        power_rates: Dict[str, float]
    ):
        """
        Initialize an MTD node.
        
        Args:
            node_id: Unique identifier for this node
            initial_energy: Initial energy budget in units
            idle_timer: Number of slots to wait idle before sleeping (ts)
            wakeup_time: Number of slots required to wake up (tw)
            power_rates: Dictionary with keys 'PT', 'PB', 'PI', 'PW', 'PS'
                        PT: Transmit power
                        PB: Busy (collision/listen) power
                        PI: Idle power
                        PW: Wake-up power
                        PS: Sleep power
        """
        self.node_id = node_id
        self.state = NodeState.IDLE
        self.queue = deque()  # Each element: (arrival_slot, packet_data)
        
        # Energy tracking
        self.energy = initial_energy
        self.initial_energy = initial_energy
        
        # Timers
        self.idle_timer = idle_timer  # ts parameter
        self.wakeup_time = wakeup_time  # tw parameter
        self.current_idle_count = 0
        self.current_wakeup_count = 0
        
        # Power consumption rates
        self.power_rates = power_rates
        
        # Statistics
        self.total_delay = 0.0
        self.packets_delivered = 0
        self.packets_arrived = 0
        self.delays = []
        
        # Track energy consumed by each state
        self.energy_consumed_by_state = {
            NodeState.ACTIVE: 0.0,
            NodeState.IDLE: 0.0,
            NodeState.SLEEP: 0.0,
            NodeState.WAKEUP: 0.0
        }
        
        # Track time in each state
        self.time_in_state = {
            NodeState.ACTIVE: 0,
            NodeState.IDLE: 0,
            NodeState.SLEEP: 0,
            NodeState.WAKEUP: 0
        }
        
    def arrive_packet(self, current_slot: int, arrival_rate: float) -> bool:
        """
        Generate packet arrival using Bernoulli process.
        
        In each slot, a packet arrives with probability λ (arrival_rate).
        If a packet arrives, it's added to the queue with its arrival time.
        
        Args:
            current_slot: Current simulation slot number
            arrival_rate: Arrival probability λ per slot (Bernoulli parameter)
            
        Returns:
            True if a packet arrived, False otherwise
        """
        if random.random() < arrival_rate:
            # Packet arrives
            packet = (current_slot, {})  # (arrival_slot, packet_data)
            self.queue.append(packet)
            self.packets_arrived += 1
            
            # If node was idle or sleeping, packet arrival triggers state change
            if self.state == NodeState.IDLE:
                self.state = NodeState.ACTIVE
                self.current_idle_count = 0
            elif self.state == NodeState.SLEEP:
                # Start wake-up process
                self.state = NodeState.WAKEUP
                self.current_wakeup_count = self.wakeup_time
                
            return True
        return False
    
    def attempt_transmit(self, transmission_prob: float) -> bool:
        """
        Attempt to transmit a packet with probability q.
        
        Only called when node is in ACTIVE state and has packets.
        Uses Bernoulli trial with probability q.
        
        Args:
            transmission_prob: Transmission probability q
            
        Returns:
            True if node attempts transmission, False otherwise
        """
        if self.state != NodeState.ACTIVE or len(self.queue) == 0:
            return False
            
        return random.random() < transmission_prob
    
    def handle_success(self, current_slot: int) -> Optional[float]:
        """
        Handle successful packet transmission.
        
        Remove packet from queue, record delay, update statistics.
        
        Args:
            current_slot: Current simulation slot number
            
        Returns:
            Delay of the delivered packet (in slots), or None if no packet
        """
        if len(self.queue) == 0:
            return None
            
        # Remove packet from queue
        arrival_slot, packet_data = self.queue.popleft()
        
        # Calculate delay (queueing + access delay)
        delay = current_slot - arrival_slot
        
        # Update statistics
        self.total_delay += delay
        self.packets_delivered += 1
        self.delays.append(delay)
        
        return delay
    
    def update_state(self, current_slot: int) -> None:
        """
        Update node state based on queue status and timers.
        
        State transition logic (on-demand sleep):
        - ACTIVE -> IDLE: When queue becomes empty
        - IDLE -> SLEEP: When idle_timer expires (ts slots)
        - SLEEP -> WAKEUP: When packet arrives (handled in arrive_packet)
        - WAKEUP -> ACTIVE: When wakeup_counter reaches 0
        
        Args:
            current_slot: Current simulation slot number
        """
        # Update time in current state
        self.time_in_state[self.state] += 1
        
        # State transition logic
        if self.state == NodeState.ACTIVE:
            # Active state: check if queue is empty
            if len(self.queue) == 0:
                self.state = NodeState.IDLE
                self.current_idle_count = 0
                
        elif self.state == NodeState.IDLE:
            # Idle state: increment idle counter
            self.current_idle_count += 1
            
            # Check if should transition to sleep
            if self.current_idle_count >= self.idle_timer:
                self.state = NodeState.SLEEP
                self.current_idle_count = 0
                
            # Check if packet arrived (would have been set to ACTIVE in arrive_packet)
            if len(self.queue) > 0:
                self.state = NodeState.ACTIVE
                self.current_idle_count = 0
                
        elif self.state == NodeState.WAKEUP:
            # Wake-up state: decrement wake-up counter
            self.current_wakeup_count -= 1
            
            # Check if wake-up complete
            if self.current_wakeup_count <= 0:
                self.state = NodeState.ACTIVE
                self.current_wakeup_count = 0
                
        elif self.state == NodeState.SLEEP:
            # Sleep state: stay in sleep unless packet arrives
            # (packet arrival handled in arrive_packet method)
            pass
    
    def consume_energy(self, was_transmitting: bool = False, 
                      was_collision: bool = False) -> float:
        """
        Consume energy based on current state and action.
        
        Energy consumption per slot depends on state:
        - ACTIVE (transmitting): PT power
        - ACTIVE (collision/busy): PB power
        - ACTIVE (listening): PI power (if not transmitting)
        - IDLE: PI power
        - SLEEP: PS power (lowest)
        - WAKEUP: PW power
        
        Args:
            was_transmitting: Whether node transmitted in this slot
            was_collision: Whether a collision occurred (for busy power)
            
        Returns:
            Energy consumed in this slot
        """
        energy_consumed = 0.0
        
        if self.state == NodeState.ACTIVE:
            if was_transmitting:
                energy_consumed = self.power_rates['PT']
            elif was_collision:
                energy_consumed = self.power_rates['PB']
            else:
                energy_consumed = self.power_rates['PI']
                
        elif self.state == NodeState.IDLE:
            energy_consumed = self.power_rates['PI']
            
        elif self.state == NodeState.SLEEP:
            energy_consumed = self.power_rates['PS']
            
        elif self.state == NodeState.WAKEUP:
            energy_consumed = self.power_rates['PW']
        
        # Update energy tracking
        self.energy -= energy_consumed
        self.energy_consumed_by_state[self.state] += energy_consumed
        
        return energy_consumed
    
    def is_depleted(self) -> bool:
        """
        Check if node's energy is depleted.
        
        Returns:
            True if energy <= 0, False otherwise
        """
        return self.energy <= 0
    
    def get_mean_delay(self) -> float:
        """
        Calculate mean queueing delay for delivered packets.
        
        Returns:
            Mean delay in slots, or 0.0 if no packets delivered
        """
        if self.packets_delivered == 0:
            return 0.0
        return self.total_delay / self.packets_delivered
    
    def get_tail_delay(self, percentile: float = 0.95) -> float:
        """
        Calculate tail delay (e.g., 95th percentile).
        
        Args:
            percentile: Percentile to compute (default 0.95 for 95th percentile)
            
        Returns:
            Tail delay value, or 0.0 if insufficient data
        """
        if len(self.delays) == 0:
            return 0.0
        
        sorted_delays = sorted(self.delays)
        index = int(percentile * len(sorted_delays))
        index = min(index, len(sorted_delays) - 1)
        
        return sorted_delays[index]
    
    def get_queue_length(self) -> int:
        """
        Get current queue length.
        
        Returns:
            Number of packets in queue
        """
        return len(self.queue)
    
    def get_energy_fraction_by_state(self) -> Dict[str, float]:
        """
        Calculate fraction of total energy consumed by each state.
        
        Returns:
            Dictionary mapping state names to energy fractions
        """
        total_energy_consumed = sum(self.energy_consumed_by_state.values())
        
        if total_energy_consumed == 0:
            return {state.value: 0.0 for state in NodeState}
        
        return {
            state.value: consumed / total_energy_consumed 
            for state, consumed in self.energy_consumed_by_state.items()
        }
    
    def get_state_fractions(self, total_slots: int) -> Dict[str, float]:
        """
        Calculate fraction of time spent in each state.
        
        Args:
            total_slots: Total number of simulation slots
            
        Returns:
            Dictionary mapping state names to time fractions
        """
        if total_slots == 0:
            return {state.value: 0.0 for state in NodeState}
        
        return {
            state.value: time / total_slots 
            for state, time in self.time_in_state.items()
        }
    
    def get_statistics(self, total_slots: int) -> Dict:
        """
        Get comprehensive statistics for this node.
        
        Args:
            total_slots: Total number of simulation slots
            
        Returns:
            Dictionary with all node statistics
        """
        return {
            'node_id': self.node_id,
            'packets_arrived': self.packets_arrived,
            'packets_delivered': self.packets_delivered,
            'packets_in_queue': len(self.queue),
            'mean_delay': self.get_mean_delay(),
            'tail_delay_95': self.get_tail_delay(0.95),
            'tail_delay_99': self.get_tail_delay(0.99),
            'energy_remaining': self.energy,
            'energy_consumed': self.initial_energy - self.energy,
            'energy_fraction_by_state': self.get_energy_fraction_by_state(),
            'state_fractions': self.get_state_fractions(total_slots),
            'final_state': self.state.value,
            'is_depleted': self.is_depleted()
        }
    
    def __repr__(self) -> str:
        """String representation of node for debugging."""
        return (f"Node(id={self.node_id}, state={self.state.value}, "
                f"queue_len={len(self.queue)}, energy={self.energy:.2f})")
