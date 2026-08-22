"""
A minimal but REAL FTP server implemented from scratch on raw sockets.

This exists so the INDEPENDENT real-PCAP test set can include traffic from a
genuinely DIFFERENT server implementation than the pyftpdlib used in v1/v2 -- real
diversity, not fabricated. It speaks enough of RFC 959 to drive real benign
sessions (login, listing, download, upload, common commands) and real
brute-force sessions (repeated failed logins), in both passive (PASV) and active
(PORT) data modes. Every byte on the wire is a real FTP exchange.

Run standalone (launched as a subprocess by the capture framework):

    python custom_ftp_server.py <addr> <port> <root> <user> <pw> <passive_lo> <passive_hi>

Loopback / host-local addresses only; the capture framework never points it at an
external host.
"""

from __future__ import annotations

import os
import socket
import sys
import threading


class CustomFTPServer:
    def __init__(self, addr, port, root, user, pw, passive_lo, passive_hi, extra_users=None):
        self.addr, self.port, self.root = addr, port, root
        self.user, self.pw = user, pw
        # optional additional valid credentials (backward compatible: default none)
        self.users = {user: pw}
        if extra_users:
            self.users.update(extra_users)
        self.plo, self.phi, self._pp = passive_lo, passive_hi, passive_lo
        self._lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((addr, port))
        self.sock.listen(32)
        self._run = True

    def serve(self):
        while self._run:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _next_passive_port(self):
        with self._lock:
            p = self._pp
            self._pp = self.plo if self._pp >= self.phi else self._pp + 1
            return p

    def _open_data(self, listener, active_addr):
        try:
            if listener is not None:
                listener.settimeout(5)
                d, _ = listener.accept()
                listener.close()
                return d
            if active_addr is not None:
                return socket.create_connection(active_addr, timeout=5)
        except OSError:
            return None
        return None

    def _handle(self, conn):
        f = conn.makefile("rwb", buffering=0)

        def send(s):
            try:
                f.write((s + "\r\n").encode())
            except OSError:
                pass

        send("220 CustomFTP lab ready")
        authed, user, listener, active_addr = False, None, None, None
        while True:
            try:
                line = f.readline()
            except OSError:
                break
            if not line:
                break
            cmd = line.decode(errors="ignore").strip()
            up = cmd.upper()
            arg = cmd[len(up.split(" ", 1)[0]):].strip()
            if up.startswith("USER"):
                user = arg
                send("331 need password")
            elif up.startswith("PASS"):
                if user in self.users and arg == self.users[user]:
                    authed = True
                    send("230 login ok")
                else:
                    send("530 login incorrect")
            elif up == "SYST":
                send("215 UNIX Type: L8")
            elif up.startswith("TYPE"):
                send("200 type set")
            elif up == "PWD" or up == "XPWD":
                send('257 "/"')
            elif up == "NOOP":
                send("200 noop")
            elif up.startswith("SIZE"):
                p = os.path.join(self.root, os.path.basename(arg))
                send("213 %d" % os.path.getsize(p) if os.path.isfile(p) else "550 no file")
            elif up.startswith("MDTM"):
                send("213 20240101000000")
            elif up == "PASV":
                if not authed:
                    send("530 login first"); continue
                dp = self._next_passive_port()
                listener = socket.socket()
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.addr, dp))
                listener.listen(1)
                active_addr = None
                h = self.addr.split(".")
                send("227 Entering Passive Mode (%s,%s,%s,%s,%d,%d)"
                     % (h[0], h[1], h[2], h[3], dp >> 8, dp & 0xFF))
            elif up.startswith("PORT"):
                if not authed:
                    send("530 login first"); continue
                p = arg.split(",")
                active_addr = (".".join(p[:4]), (int(p[4]) << 8) + int(p[5]))
                listener = None
                send("200 PORT ok")
            elif up.startswith("LIST") or up.startswith("NLST"):
                if not authed:
                    send("530 login first"); continue
                send("150 here comes the listing")
                d = self._open_data(listener, active_addr); listener = active_addr = None
                if d:
                    for n in sorted(os.listdir(self.root)):
                        d.sendall(("-rw-r--r-- 1 lab lab 100 Jan 01 00:00 " + n + "\r\n").encode())
                    d.close()
                send("226 listing done")
            elif up.startswith("RETR"):
                if not authed:
                    send("530 login first"); continue
                p = os.path.join(self.root, os.path.basename(arg))
                if not os.path.isfile(p):
                    send("550 no such file"); continue
                send("150 opening data")
                d = self._open_data(listener, active_addr); listener = active_addr = None
                if d:
                    with open(p, "rb") as fh:
                        d.sendall(fh.read())
                    d.close()
                send("226 transfer complete")
            elif up.startswith("STOR") or up.startswith("APPE"):
                if not authed:
                    send("530 login first"); continue
                p = os.path.join(self.root, os.path.basename(arg))
                mode = "ab" if up.startswith("APPE") else "wb"
                send("150 ready for data")
                d = self._open_data(listener, active_addr); listener = active_addr = None
                if d:
                    with open(p, mode) as fh:
                        while True:
                            b = d.recv(8192)
                            if not b:
                                break
                            fh.write(b)
                    d.close()
                send("226 stored")
            elif up.startswith(("MKD", "XMKD")):
                send('257 "%s" created' % arg)
            elif up.startswith(("RMD", "XRMD", "DELE")):
                send("250 removed")
            elif up.startswith("RNFR"):
                send("350 ready for RNTO")
            elif up.startswith("RNTO"):
                send("250 renamed")
            elif up.startswith("CWD") or up == "CDUP":
                send("250 cwd ok")
            elif up == "STAT":
                send("211 lab status")
            elif up.startswith("QUIT"):
                send("221 bye"); break
            else:
                send("502 command not implemented")
        try:
            conn.close()
        except OSError:
            pass

    def stop(self):
        self._run = False
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    addr, port, root, user, pw, plo, phi = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5],
        int(sys.argv[6]), int(sys.argv[7]))
    # optional argv[8]: extra users as "u1:p1,u2:p2" (backward compatible if absent)
    extra = {}
    if len(sys.argv) > 8 and sys.argv[8]:
        for pair in sys.argv[8].split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                extra[u] = p
    CustomFTPServer(addr, port, root, user, pw, plo, phi, extra_users=extra).serve()


if __name__ == "__main__":
    main()
