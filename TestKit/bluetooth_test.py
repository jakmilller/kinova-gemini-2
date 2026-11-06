
import asyncio
import ctypes
import os
import io
import socket
import threading
import struct
import time
import collections

RFCOMM_CHANNEL = 4

# Anaconda's CPython is built without <bluetooth/bluetooth.h>, so its socket module has no
# AF_BLUETOOTH and cannot parse an RFCOMM address tuple. The kernel (BlueZ) supports RFCOMM
# regardless, so we skip the missing Python layer and call libc directly for the three
# address-taking syscalls. Everything after accept() -- recv/sendall/close -- is
# address-agnostic and works on a plain socket object on any interpreter.
AF_BLUETOOTH = 31
BTPROTO_RFCOMM = 3

import socket

def rfcomm_listen(channel=1, backlog=1):
    """Bind an RFCOMM listening socket natively on Windows/Linux."""
    server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    
    try:
        server.bind((socket.BDADDR_ANY, channel))
        
        server.listen(backlog)
        print(f"Listening for Bluetooth connections on RFCOMM channel {channel}...")
        
    except Exception:
        server.close()
        raise
        
    return server

def find_free_channel(start=1, end=30):
        errors = []
        for ch in range(start, end + 1):
            try:
                return rfcomm_listen(ch), ch
            except OSError as e:
                errors.append(f'rfcomm ch {ch} OSError: {e}')
        raise OSError("No free RFCOMM channel, checked 1-30: \n" + "\n".join(errors))

class MockBluetoothKit:

    def __init__(self):
        self.client_conn = None
        self.client_lock = threading.Lock()

        self._out_cv = threading.Condition()
        self._text_q = collections.deque(maxlen=200)
        threading.Thread(target=self.send_loop, daemon=True).start()

    def queue_message(self, packet_type, payload, timeData):
        with self._out_cv:
            self._text_q.append((packet_type, payload, timeData))
            self._out_cv.notify()

    def send_loop(self):
        while True:
            with self._out_cv:
                while not self._text_q:
                    self._out_cv.wait()
                if self._text_q:
                    packet_type, payload, timeData = self._text_q.popleft()
            self.send_packet(packet_type, payload, timeData)

    def send_packet(self, packet_type: bytes, payload: bytes, timeData):
        with self.client_lock:
            conn = self.client_conn
        if conn is not None:
            t_start_bt = time.time()
            try:
                conn.sendall(packet_type + struct.pack('!I', len(payload)) + payload)
            except OSError as e:
                self.get_logger().warn(f'Failed to send to tablet: {e}')
            t_done_bt = time.time()
            print(
                f'time before received by node: {(timeData['time_ev'] - timeData['time_node_rec']) * 1000:.0f}ms \n'
                f'time in queue: {(t_start_bt - timeData['time_ev']) * 1000:.0f}ms \n'
                f'time to send over bt: {(t_done_bt - t_start_bt) * 1000:.0f}ms \n'
                f'total time: {(t_done_bt - timeData['time_ev']) * 1000:.0f}ms'
            )


    def handle_client_connection(self, conn, addr):

        with self.client_lock:
            self.client_conn = conn

        buf = b""

        try:
            while True:
                
                data = conn.recv(1024)
                if not data:
                    break
                buf += data
                time_start = time.time()
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    instruction = line.decode("utf-8", errors="replace").strip()
                    if instruction:
                        time_finished = time.time()
                        fakeTimeData = {'time_ev': time_start, 'time_node_rec': time_finished}
                        self.queue_message(b'\x02', (instruction + "\n").encode("utf-8"), fakeTimeData)
        except OSError as e:
            print(f'Connection ended: {e}')
        finally:

            with self.client_lock:
                self.client_conn = None
            try:
                conn.close()
            except OSError:
                pass
            print('tablet disconnected.')

    def run_server(self):
        try:
            server, ch = find_free_channel()
    
        except OSError as e:
            print(f'Cannot make BT socket ({e})')
            return
    
        try:
            while True:
                try:
                    conn, info = server.accept()
                except OSError as e:
                    print(f'BT accept failed, stopping server ({e})')
                    break
                self.handle_client_connection(conn, info[0])
    
        finally:
            server.close()


if __name__ == "__main__":
    try:
        MockBluetoothKit().run_server()
    except KeyboardInterrupt:
        print("\nStopped.")