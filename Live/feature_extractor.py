from scapy.layers.inet import IP, TCP, UDP


def _timestamp(pkt) -> float:
    """
    scapy hands back packet timestamps as EDecimal. Converting once, here, keeps
    every downstream calculation in plain floats -- mixing the two types is a
    subtle source of type errors inside numpy.
    """
    return float(pkt.time)


def _tcp_header_len(tcp) -> int:
    """
    TCP header length in bytes.

    `dataofs` is populated on packets parsed off the wire or from a capture
    file, but is None on a packet that has been constructed in memory and not
    yet serialised. Falling back to the serialised length keeps this correct in
    both cases instead of raising TypeError on None * 4.
    """
    if tcp.dataofs is not None:
        return tcp.dataofs * 4
    return len(bytes(tcp)) - len(tcp.payload)


def update_flow(flow, pkt):

    ts = _timestamp(pkt)

    # The first packet defines the flow's start. start_time used to be set to
    # wall-clock time when the Flow object was constructed, which broke duration
    # entirely when reading a stored capture.
    if flow.start_time is None:
        flow.start_time = ts

    flow.last_seen = ts

    packet_length = len(pkt)

    flow.total_packets += 1
    flow.total_bytes += packet_length

    flow.all_packet_lengths.append(packet_length)
    flow.all_packet_times.append(ts)

    ip = pkt[IP]

    forward = ip.src == flow.src_ip

    if forward:

        flow.forward_packets += 1
        flow.forward_bytes += packet_length

        flow.forward_packet_lengths.append(packet_length)
        flow.forward_packet_times.append(ts)

    else:

        flow.backward_packets += 1
        flow.backward_bytes += packet_length

        flow.backward_packet_lengths.append(packet_length)
        flow.backward_packet_times.append(ts)

    # Header length.
    #
    # CICFlowMeter -- which produced the training features -- measures the
    # TRANSPORT header for "Fwd Header Len". This previously recorded ip.ihl * 4,
    # the IP header, which is a different quantity and a likely source of drift
    # against the trained models. The TCP header length is the data-offset field;
    # UDP headers are always 8 bytes.
    if TCP in pkt:
        header_length = _tcp_header_len(pkt[TCP])
    elif UDP in pkt:
        header_length = 8
    else:
        header_length = (ip.ihl or 5) * 4

    if forward:
        flow.forward_header_lengths.append(header_length)
    else:
        flow.backward_header_lengths.append(header_length)

    if TCP in pkt:

        tcp = pkt[TCP]

        seg_size = len(tcp.payload)

        if forward:
            flow.forward_segment_sizes.append(seg_size)
        else:
            flow.backward_segment_sizes.append(seg_size)

        if forward and flow.init_fwd_win_bytes is None:
            flow.init_fwd_win_bytes = tcp.window

        flags = tcp.flags

        if flags.S:
            flow.syn_count += 1

        if flags.A:
            flow.ack_count += 1

        if flags.F:
            flow.fin_count += 1

        if flags.R:
            flow.rst_count += 1

        if flags.P:
            flow.psh_count += 1

        if flags.U:
            flow.urg_count += 1
