from scapy.all import get_if_list

print("Available Interfaces:\n")

for i, iface in enumerate(get_if_list()):
    print(f"{i}: {iface}")