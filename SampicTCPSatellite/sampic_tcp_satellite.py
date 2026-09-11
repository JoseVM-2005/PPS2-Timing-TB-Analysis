"""
SAMPIC TCP Constellation Satellite
================================
Integrates the SAMPIC DAQ module into the Constellation framework.
"""

import socket
import time
import logging
from typing import Optional

from constellation.core.satellite import Satellite
from constellation.core.configuration import Configuration

log = logging.getLogger(__name__)


class SampicTCPSatellite(Satellite):
    """Constellation satellite for the SAMPIC waveform digitiser."""

    def __init__(
        self,
        name: str,
        group: str,
        cmd_port: int,
        hb_port: int,
        mon_port: int,
        interface: str,
    ) -> None:
        super().__init__(
            name=name,
            group=group,
            cmd_port=cmd_port,
            hb_port=hb_port,
            mon_port=mon_port,
            interface=[interface] if interface else None,
        )
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _send_cmd(self, cmd: str, timeout: float = 2.0) -> str:
        """Send a command string to SAMPIC and return the reply."""
        if self._sock is None:
            raise RuntimeError("Socket is not connected.")
        if not cmd.endswith("\n"):
            cmd += "\n"
        log.debug("SAMPIC -> %s", cmd.strip())
        self._sock.sendall(cmd.encode())
        self._sock.settimeout(timeout)
        try:
            reply = self._sock.recv(4096).decode(errors="replace").strip()
            log.debug("SAMPIC <- %s", reply)
            return reply
        except socket.timeout:
            log.warning("No reply from SAMPIC for command: %s", cmd.strip())
            return ""

    def _check_ok(self, reply: str, cmd_name: str) -> None:
        """Raise RuntimeError if the SAMPIC reply does not contain EXECUTED OK."""
        if "EXECUTED OK" not in reply:
            raise RuntimeError(
                f"SAMPIC command '{cmd_name}' failed. Reply: '{reply}'"
            )

    # ------------------------------------------------------------------ #
    #  Constellation FSM callbacks                                         #
    # ------------------------------------------------------------------ #

    def do_initializing(self, config: Configuration) -> str:
        self._sampic_ip     = config.get("sampic_ip",      "192.168.1.46")
        self._sampic_port   = config.get("sampic_port",    8261)
        self._save_dir      = config.get("save_directory", r"C:\SAMPICtest")
        self._base_filename = config.get("base_filename",  "run")
        self._run_time      = config.get("run_time_s",     0)
        self._run_hits      = config.get("run_hits",       0)
        self._setup_file    = config.get("setup_file",     r"C:\SAMPICtest\Setup_LabTest.dat")
        return "SAMPIC configuration loaded."

    def do_launching(self) -> str:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self._sampic_ip, self._sampic_port))
        log.info("Connected to SAMPIC at %s:%d", self._sampic_ip, self._sampic_port)

        if self._setup_file:
            reply = self._send_cmd(f"#1 LOAD_SETUP -FROM {self._setup_file}")
            self._check_ok(reply, "LOAD_SETUP")
            log.info("Setup loaded from %s", self._setup_file)

        return "SAMPIC connected and launched."

    def do_starting(self, run_identifier: str) -> str:
        cmd_parts = [f"#2 START_RUN"]
        cmd_parts.append(f"-SAVETO {self._save_dir}")
        cmd_parts.append(f"-BASEFILENAME {self._base_filename}")
        cmd_parts.append("-SAVEDATA")
        cmd_parts.append("-ADDRUNID")
        cmd_parts.append(f"-RUNID {run_identifier}")
        if self._run_time > 0:
            cmd_parts.append(f"-TIME {self._run_time}")
        if self._run_hits > 0:
            cmd_parts.append(f"-HITS {self._run_hits}")

        reply = self._send_cmd(" ".join(cmd_parts), timeout=5.0)
        self._check_ok(reply, "START_RUN")
        log.info("SAMPIC acquisition started (run %s).", run_identifier)
        return f"SAMPIC run {run_identifier} started."

    def do_run(self) -> str:
        while not self.stop_requested():
            time.sleep(0.2)
        return "RUN loop stopped."

    def do_stopping(self) -> str:
        reply = self._send_cmd(f"#3 STOP_RUN", timeout=5.0)
        if reply and "EXECUTED OK" not in reply:
            raise RuntimeError(f"SAMPIC command 'STOP_RUN' failed. Reply: '{reply}'")
        log.info("SAMPIC acquisition stopped.")
        return "SAMPIC run stopped."

    def do_landing(self) -> str:
        self._close_socket()
        log.info("SAMPIC satellite landed.")
        return "SAMPIC landed."

    def do_interrupting(self) -> str:
        try:
            self._send_cmd(f"#3 STOP_RUN", timeout=3.0)
        except Exception as exc:
            log.warning("Could not send STOP_RUN during interrupt: %s", exc)
        self._close_socket()
        return "SAMPIC interrupted."

    # ------------------------------------------------------------------ #
    #  Teardown                                                            #
    # ------------------------------------------------------------------ #

    def _close_socket(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
                log.info("Disconnected from SAMPIC.")
        except OSError:
            pass
        finally:
            self._sock = None

    def __del__(self) -> None:
        self._close_socket()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SAMPIC TCP Constellation Satellite")
    parser.add_argument("--name",      default="One",   help="Satellite instance name")
    parser.add_argument("--group",     default="mylab", help="Constellation group name")
    parser.add_argument("--cmd-port",  type=int, default=23999, help="Command port (ZMQ)")
    parser.add_argument("--hb-port",   type=int, default=24000, help="Heartbeat port (ZMQ)")
    parser.add_argument("--mon-port",  type=int, default=24001, help="Monitoring port (ZMQ)")
    parser.add_argument("--interface", default="*",      help="Network interface to bind")
    parser.add_argument("--level",     default="INFO",   help="Log level")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.level.upper())

    sat = SampicTCPSatellite(
        name=args.name,
        group=args.group,
        cmd_port=args.cmd_port,
        hb_port=args.hb_port,
        mon_port=args.mon_port,
        interface=args.interface,
    )
    sat.run_satellite()
