# -*- coding: utf-8 -*-
import socket
import json
import struct
import os
import threading
import time
import base64 # Required for decoding files
from typing import Optional
from datetime import datetime
date_str = lambda: datetime.now().strftime("%Y-%m-%d")

# --- Global Configuration ---
SERVER_HOST = '0.0.0.0'
UNITY_PORT = 12345
CONTROL_PORT = 12346
OUTPUT_DIR = 'rendered_from_unity'
HEARTBEAT_INTERVAL = 5
HEARTBEAT_TIMEOUT = 15

# --- Ensure output directory exists ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Custom Exceptions ---
class ConnectionClosedError(Exception):
    """Custom exception for closed connections"""
    pass

# --- Global State Management ---
class GlobalState:
    def __init__(self):
        self.unity_client_socket: Optional[socket.socket] = None
        self.control_client_socket: Optional[socket.socket] = None
        self.lock = threading.Lock()
        self.last_unity_heartbeat = time.time()

    def set_unity_client(self, client_socket: Optional[socket.socket]):
        with self.lock:
            self.unity_client_socket = client_socket
            if client_socket:
                self.last_unity_heartbeat = time.time()

    def get_unity_client(self) -> Optional[socket.socket]:
        with self.lock:
            return self.unity_client_socket

    def set_control_client(self, client_socket: Optional[socket.socket]):
        with self.lock:
            self.control_client_socket = client_socket

    def get_control_client(self) -> Optional[socket.socket]:
        with self.lock:
            return self.control_client_socket
            
    def update_heartbeat(self):
        with self.lock:
            self.last_unity_heartbeat = time.time()

    def is_unity_alive(self) -> bool:
        with self.lock:
            return (time.time() - self.last_unity_heartbeat) < HEARTBEAT_TIMEOUT

# --- Utility Functions ---
def send_packet(sock: socket.socket, data: bytes):
    try:
        length_prefix = struct.pack('!I', len(data))
        sock.sendall(length_prefix + data)
    except (socket.error, BrokenPipeError) as e:
        raise ConnectionClosedError(f"Failed to send packet: {e}")

def receive_packet(sock: socket.socket) -> bytes:
    try:
        length_prefix = sock.recv(4)
        if not length_prefix:
            raise ConnectionClosedError("Connection closed by peer (prefix).")
        
        length = struct.unpack('!I', length_prefix)[0]
        
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                raise ConnectionClosedError("Connection closed by peer (data).")
            data += packet
        return data
    except (socket.error, struct.error, ConnectionResetError) as e:
        raise ConnectionClosedError(f"Failed to receive packet: {e}")

# --- Client Handling Logic ---
def handle_unity_client(client_socket: socket.socket, addr: tuple, state: GlobalState):
    """Handles connection from the Unity client, receives data, and forwards it."""
    print(f"[Unity-Server] Unity client connected from {addr}")
    state.set_unity_client(client_socket)
    output_dir = os.path.join(OUTPUT_DIR, date_str())
    os.makedirs(output_dir, exist_ok=True)
    try:
        while True:
            data_bytes = receive_packet(client_socket)
            message = json.loads(data_bytes.decode('utf-8'))
            
            msg_type = message.get("type")
            
            if msg_type == "render_result":
                print("[Unity-Server] Received render result from Unity.")
                files_saved_successfully = True
                error_message = ""
                
                # Correctly decode and save files sent from Unity.
                try:
                    files = message.get("files", {})
                    if not files:
                        print("[Unity-Server] Warning: Render result message contained no files.")
                    
                    for filename, b64_data_str in files.items():
                        file_path = os.path.join(output_dir, filename)
                        decoded_data = base64.b64decode(b64_data_str)
                        with open(file_path, "wb") as f:
                            f.write(decoded_data)
                    print(f"[Unity-Server] Render files saved to '{output_dir}' directory.")
                except (TypeError, base64.binascii.Error, IOError) as e:
                    error_message = f"Error decoding or saving file: {e}"
                    print(f"[Unity-Server] {error_message}")
                    files_saved_successfully = False
                
                # Notify the control client about the final result of the render task.
                control_client = state.get_control_client()
                if control_client:
                    if files_saved_successfully:
                        response = {
                            "type": "render_notification",
                            "status": "success",
                            "message": f"Render completed and files saved to '{output_dir}'."
                        }
                    else:
                        response = {
                            "type": "render_notification",
                            "status": "error",
                            "message": error_message
                        }
                    send_packet(control_client, json.dumps(response).encode('utf-8'))

            elif msg_type == "ack":
                print(f"[Unity-Server] Received ACK from Unity: {message.get('message')}")
                # Forward the ACK to the control client.
                control_client = state.get_control_client()
                if control_client:
                    send_packet(control_client, data_bytes)
                    
            elif msg_type == "heartbeat":
                state.update_heartbeat()

            else:
                print(f"[Unity-Server] Received unknown message type from Unity: {message}")

    except (ConnectionClosedError, json.JSONDecodeError) as e:
        print(f"[Unity-Server] Unity client {addr} disconnected: {e}")
    finally:
        state.set_unity_client(None)
        client_socket.close()
        print(f"[Unity-Server] Connection with {addr} closed.")

def handle_control_client(client_socket: socket.socket, addr: tuple, state: GlobalState):
    """Handles connection from the external API and forwards commands to Unity."""
    print(f"[Control-Server] Control client connected from {addr}")
    
    # Only allow one control client at a time for simplicity.
    if state.get_control_client() is not None:
        print(f"[Control-Server] Another control client is already connected. Closing connection to {addr}.")
        error_response = {
            "type": "ack",
            "status": "error",
            "message": "Server busy. Another control client is already connected."
        }
        try:
            send_packet(client_socket, json.dumps(error_response).encode('utf-8'))
        except ConnectionClosedError:
            pass
        finally:
            client_socket.close()
        return

    state.set_control_client(client_socket)

    try:
        while True:
            command_bytes = receive_packet(client_socket)
            command = json.loads(command_bytes.decode('utf-8'))
            print(f"[Control-Server] Received command: {command.get('command')}")

            unity_client = state.get_unity_client()
            if unity_client and state.is_unity_alive():
                send_packet(unity_client, command_bytes)
                print(f"[Control-Server] Forwarded command to Unity.")
            else:
                # If Unity is not connected, return an immediate error.
                error_response = {
                    "type": "ack", # Use a consistent response type.
                    "status": "error",
                    "message": "Unity client is not connected or not responding."
                }
                send_packet(client_socket, json.dumps(error_response).encode('utf-8'))
                print("[Control-Server] Sent error: Unity client not available.")

    except (ConnectionClosedError, json.JSONDecodeError) as e:
        print(f"[Control-Server] Control client {addr} disconnected: {e}")
    finally:
        state.set_control_client(None)
        client_socket.close()
        print(f"[Control-Server] Connection with {addr} closed.")

# --- Server Start and Listening ---
def start_listener(host: str, port: int, handler_func, state: GlobalState):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server listening on {host}:{port} for {handler_func.__name__}")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            thread = threading.Thread(target=handler_func, args=(client_socket, addr, state))
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"Error accepting connection on port {port}: {e}")

def unity_heartbeat_checker(state: GlobalState):
    """Periodically sends heartbeat requests to Unity and checks its response."""
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        unity_client = state.get_unity_client()
        if unity_client:
            if state.is_unity_alive():
                try:
                    heartbeat_command = {"command": "heartbeat", "payload": {}}
                    send_packet(unity_client, json.dumps(heartbeat_command).encode('utf-8'))
                except ConnectionClosedError:
                    pass
            else:
                print("[Heartbeat] Unity client timed out. Closing connection.")
                unity_client.close()
                state.set_unity_client(None)

# --- Main Entry Point ---
def main():
    global_state = GlobalState()

    unity_server_thread = threading.Thread(target=start_listener, args=(SERVER_HOST, UNITY_PORT, handle_unity_client, global_state))
    unity_server_thread.daemon = True
    unity_server_thread.start()

    control_server_thread = threading.Thread(target=start_listener, args=(SERVER_HOST, CONTROL_PORT, handle_control_client, global_state))
    control_server_thread.daemon = True
    control_server_thread.start()

    heartbeat_thread = threading.Thread(target=unity_heartbeat_checker, args=(global_state,))
    heartbeat_thread.daemon = True
    heartbeat_thread.start()

    print("--- IPC Server is running ---")
    print(f"Unity clients connect to: {SERVER_HOST}:{UNITY_PORT}")
    print(f"Control APIs connect to:  {SERVER_HOST}:{CONTROL_PORT}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    main()
