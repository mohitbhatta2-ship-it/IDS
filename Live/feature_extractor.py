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

    ip = pkt[IP]

    forward = ip.src == flow.src_ip

    # Packet length == transport PAYLOAD (segment) size, matching CICFlowMeter.
    #
    # CICFlowMeter's packet-length and segment-size features measure the L4
    # payload, NOT the full frame. Two facts in the training data prove it: for
    # every class "Fwd Pkt Len Mean" equals "Fwd Seg Size Avg" (CICFlowMeter
    # defines them identically, as the mean payload size), and several classes
    # have a mean below 54 bytes, which is impossible for a whole Ethernet frame.
    # This code previously used len(pkt) -- Ethernet + IP + transport headers
    # included -- which inflated every byte and length feature (TotLen, Subflow,
    # Pkt Len, Seg Size Avg, Pkt Size Avg ...) and pushed them out of the range
    # the models were trained on, even though nothing raised an error.
    if TCP in pkt:
        payload_length = len(pkt[TCP].payload)
    elif UDP in pkt:
        payload_length = len(pkt[UDP].payload)
    else:
        payload_length = len(ip.payload)

    flow.total_packets += 1
    flow.total_bytes += payload_length

    flow.all_packet_lengths.append(payload_length)
    flow.all_packet_times.append(ts)

    if forward:

        flow.forward_packets += 1
        flow.forward_bytes += payload_length

        flow.forward_packet_lengths.append(payload_length)
        flow.forward_packet_times.append(ts)

    else:

        flow.backward_packets += 1
        flow.backward_bytes += payload_length

        flow.backward_packet_lengths.append(payload_length)
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

        # CICFlowMeter's "Init Fwd Win Byts" is the receive-window advertised on
        # the FIRST forward packet. It stays None until we see one; a flow with
        # no forward TCP packet (e.g. UDP) reports -1 -- see feature_calculator.
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
