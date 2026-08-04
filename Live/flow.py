from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Flow:

    # Flow identification
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # Timing.
    #
    # Both are packet-capture timestamps, NOT wall-clock time. They used to
    # default to time.time(), which is only harmless while sniffing live: as soon
    # as you read a stored capture, start_time was "now" while last_seen came
    # from the packet, so Flow Duration came out as the age of the capture file
    # (years, for the 2018 dataset) and every IAT feature built on it was
    # meaningless. They are set from the first packet instead -- see
    # feature_extractor.update_flow.
    start_time: Optional[float] = None
    last_seen: Optional[float] = None

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
    init_fwd_win_bytes: Optional[int] = None

    # Segment sizes
    forward_segment_sizes: list = field(default_factory=list)
    backward_segment_sizes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Flow duration in seconds, or 0.0 before any packet has been seen."""
        if self.start_time is None or self.last_seen is None:
            return 0.0
        return float(self.last_seen) - float(self.start_time)

    def __str__(self) -> str:
        proto = {6: "TCP", 17: "UDP"}.get(self.protocol, str(self.protocol))
        return (
            f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port} "
            f"[{proto}] {self.total_packets} pkts"
        )
