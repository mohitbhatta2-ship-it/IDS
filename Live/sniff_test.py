from scapy.all import sniff
from scapy.layers.inet import IP

from flow_builder import get_or_create_flow, flows
from feature_extractor import update_flow
from flow_manager import cleanup_flows

INTERFACE = "Wi-Fi"


def process(pkt):

    if IP not in pkt:
        return

    flow = get_or_create_flow(pkt)

    if flow is None:
        return

    update_flow(flow, pkt)

    cleanup_flows(flows)


print("Capturing...")

sniff(
    iface=INTERFACE,
    prn=process,
    store=False
)