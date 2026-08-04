from scapy.all import IFACES

print("Available interfaces:\n")

for iface in IFACES.values():
    print("=" * 60)
    print(f"Index : {iface.index}")
    print(f"Name  : {iface.name}")
    print(f"GUID  : {iface.network_name}")
    print(f"Desc  : {iface.description}")
    print(f"IP    : {iface.ip}")