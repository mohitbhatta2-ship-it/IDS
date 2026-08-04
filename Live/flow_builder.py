import config
from flow import Flow

# Active flows, keyed by 5-tuple. Module-level so the sniffer callback and the
# expiry sweep share one table.
flows = {}


def make_key(src_ip, dst_ip, src_port, dst_port, proto):
    return (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        proto
    )


def reset():
    """Drop all tracked flows. Used between captures and by the tests."""
    flows.clear()


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

    # Guard against unbounded growth. A busy link or a port scan can create
    # flows far faster than the timeout retires them; without a cap the table
    # grows until the process is killed. Dropping the oldest is crude but
    # bounded, and it is better than dying mid-capture.
    if len(flows) >= config.MAX_FLOWS:
        oldest = min(flows, key=lambda k: flows[k].last_seen or 0)
        del flows[oldest]

    # Create a new flow. Timestamps are deliberately left unset -- the first
    # call to update_flow() fills them in from the packet itself.
    flow = Flow(
        src_ip=ip.src,
        dst_ip=ip.dst,
        src_port=transport.sport,
        dst_port=transport.dport,
        protocol=proto
    )

    flows[forward_key] = flow

    return flow
