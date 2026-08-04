from scapy.layers.inet import TCP, UDP, IP


def update_flow(flow, pkt):

    flow.last_seen = pkt.time

    packet_length = len(pkt)

    flow.total_packets += 1
    flow.total_bytes += packet_length

    flow.all_packet_lengths.append(packet_length)
    flow.all_packet_times.append(pkt.time)

    ip = pkt[IP]

    forward = ip.src == flow.src_ip

    if forward:

        flow.forward_packets += 1
        flow.forward_bytes += packet_length

        flow.forward_packet_lengths.append(packet_length)
        flow.forward_packet_times.append(pkt.time)

        flow.forward_header_lengths.append(ip.ihl * 4)

    else:

        flow.backward_packets += 1
        flow.backward_bytes += packet_length

        flow.backward_packet_lengths.append(packet_length)
        flow.backward_packet_times.append(pkt.time)

        flow.backward_header_lengths.append(ip.ihl * 4)

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
  