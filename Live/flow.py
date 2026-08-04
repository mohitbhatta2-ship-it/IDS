from dataclasses import dataclass, field
import time

@dataclass
class Flow:

    # Flow identification
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # Timing
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Packet counters
    total_packets: int = 0
    forward_packets: int = 0
    backward_packets: int = 0

    # Byte counters
    total_bytes: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0

    # Packet lengths
    all_packet_lengths: list = field(default_factory=list)
    forward_packet_lengths: list = field(default_factory=list)
    backward_packet_lengths: list = field(default_factory=list)

    # Packet arrival times
    all_packet_times: list = field(default_factory=list)
    forward_packet_times: list = field(default_factory=list)
    backward_packet_times: list = field(default_factory=list)

    # TCP flags
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0

    # Header lengths
    forward_header_lengths: list = field(default_factory=list)
    backward_header_lengths: list = field(default_factory=list)

    # TCP Window
    init_fwd_win_bytes: int = None

    # Segment sizes
    forward_segment_sizes: list = field(default_factory=list)
    backward_segment_sizes: list = field(default_factory=list)