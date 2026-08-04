"""
Capture network traffic (live or from a stored capture) and classify each flow.

Usage
-----
    # live capture -- needs root/administrator for packet capture
    sudo python sniff_test.py --iface en0

    # replay a stored capture; no privileges needed
    python sniff_test.py --pcap ../captures/friday.pcap

    # stop after N packets
    sudo python sniff_test.py --count 500

Nothing runs on import: the previous version called sniff() at module level, so
merely importing the file started capturing.
"""

import argparse
import sys

from scapy.all import sniff
from scapy.layers.inet import IP

import config
from feature_extractor import update_flow
from flow_builder import flows, get_or_create_flow, reset
from flow_manager import cleanup_flows, flush_all


class Capture:
    """Feeds packets into the flow table and expires flows as time advances."""

    def __init__(self, quiet=False):
        self.quiet = quiet
        self.packets = 0
        self.classified = 0
        self._last_cleanup = None

    def handle(self, pkt):
        if IP not in pkt:
            return

        flow = get_or_create_flow(pkt)
        if flow is None:          # not TCP or UDP
            return

        update_flow(flow, pkt)
        self.packets += 1

        now = float(pkt.time)

        # Sweep on the capture clock, and only every CLEANUP_INTERVAL seconds.
        # The old code swept on every single packet, which is O(flows) per
        # packet and the reason capture fell behind on a busy link.
        if self._last_cleanup is None:
            self._last_cleanup = now
        elif now - self._last_cleanup >= config.CLEANUP_INTERVAL:
            self.classified += cleanup_flows(flows, now, quiet=self.quiet)
            self._last_cleanup = now

    def finish(self):
        # Anything still open at the end of the capture must still be scored,
        # otherwise a short capture produces no output at all.
        self.classified += flush_all(flows, quiet=self.quiet)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--iface", help="interface to capture on (default: scapy's choice)")
    source.add_argument("--pcap", help="read from a capture file instead of the network")
    parser.add_argument("--count", type=int, default=0, help="stop after N packets (0 = unlimited)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)

    print(config.describe())
    reset()

    capture = Capture(quiet=args.quiet)

    try:
        if args.pcap:
            print(f"reading      : {args.pcap}\n")
            sniff(offline=args.pcap, prn=capture.handle, store=False, count=args.count)
        else:
            iface = args.iface or config.default_interface()
            print(f"interface    : {iface or 'scapy default'}")
            print("capturing... press Ctrl-C to stop\n")
            sniff(iface=iface, prn=capture.handle, store=False, count=args.count)
    except KeyboardInterrupt:
        print("\nstopping...")
    except PermissionError:
        print(
            "\nPermission denied opening the interface. Live capture needs root:\n"
            "    sudo python sniff_test.py",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"\nCapture failed: {exc}", file=sys.stderr)
        print("Run list_interfaces.py to see the available interfaces.", file=sys.stderr)
        return 1

    capture.finish()

    print(f"\npackets processed : {capture.packets}")
    print(f"flows classified  : {capture.classified}")
    print(f"written to        : {config.CSV_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
