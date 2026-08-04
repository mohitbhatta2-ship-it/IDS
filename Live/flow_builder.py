from flow import Flow

flows = {}


def make_key(src_ip, dst_ip, src_port, dst_port, proto):
    return (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        proto
    )


def get_or_create_flow(packet):

    ip = packet["IP"]

    proto = ip.proto

    if proto == 6:
        transport = packet["TCP"]
    elif proto == 17:
        transport = packet["UDP"]
    else:
        return None

    forward_key = make_key(
        ip.src,
        ip.dst,
        transport.sport,
        transport.dport,
        proto
    )

    reverse_key = make_key(
        ip.dst,
        ip.src,
        transport.dport,
        transport.sport,
        proto
    )

    # Existing forward flow
    if forward_key in flows:
        return flows[forward_key]

    # Existing reverse flow
    if reverse_key in flows:
        return flows[reverse_key]

    # Create a new flow
    flow = Flow(
        src_ip=ip.src,
        dst_ip=ip.dst,
        src_port=transport.sport,
        dst_port=transport.dport,
        protocol=proto
    )

    flows[forward_key] = flow

    return flow