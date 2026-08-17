# -*- coding: utf-8 -*-
import socket
import json
import struct
import time
from typing import Dict, Any, Optional

CONTROL_SERVER_HOST = 'localhost'
CONTROL_SERVER_PORT = 12346

class ApiClient:
    """
    API client for connecting to the ControlServer and sending standardized commands.
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client_socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """Connects to the server."""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            print(f"Successfully connected to control server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to control server: {e}")
            self.close()
            return False

    def close(self):
        """Closes the connection."""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            print("Connection to control server closed.")

    def _send_packet(self, data: bytes):
        if not self.client_socket:
            raise ConnectionError("Not connected to server.")
        length_prefix = struct.pack('!I', len(data))
        self.client_socket.sendall(length_prefix + data)

    def _receive_packet(self) -> Optional[bytes]:
        if not self.client_socket:
            raise ConnectionError("Not connected to server.")
        try:
            length_prefix = self.client_socket.recv(4)
            if not length_prefix: return None
            length = struct.unpack('!I', length_prefix)[0]

            data = b''
            while len(data) < length:
                packet = self.client_socket.recv(length - len(data))
                if not packet: return None
                data += packet
            return data
        except Exception as e:
            print(f"Receive error: {e}")
            self.close()
            return None

    def send_command(self, command: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends a command and waits for an immediate response (ACK).
        """
        if not self.client_socket:
            print("Error: Not connected. Command not sent.")
            return {"status": "error", "message": "Not connected to server."}
        
        try:
            full_command = {
                "command": command,
                "payload": payload
            }
            json_command = json.dumps(full_command)
            self._send_packet(json_command.encode('utf-8'))
            print(f"--> Sent command '{command}' with payload: {payload}")

            response_data = self._receive_packet()
            if not response_data:
                return {"status": "error", "message": "No response from server."}

            response = json.loads(response_data.decode('utf-8'))
            return response

        except (ConnectionError, json.JSONDecodeError) as e:
            print(f"Error during command execution: {e}")
            self.close()
            return {"status": "error", "message": str(e)}

def run_demo():
    """
    Runs a demo showing how to interact with the IPC system,
    with clear success/failure logging for each step.
    """
    client = ApiClient(CONTROL_SERVER_HOST, CONTROL_SERVER_PORT)
    if not client.connect():
        print("\n[DEMO FAILED] Could not connect to the control server.")
        return

    # Define a sequence of tasks to execute.
    tasks = [
        # ... (other tasks remain the same) ...
        {
            "name": "Setting Camera Pose",
            "command": "set_camera_pose",
            "payload": {
                "position": {"x": 1, "y": 2, "z": -8},
                "rotation": {"x": 10, "y": 20, "z": 0}
            }
        },
        {
            "name": "Setting Pose for object 'Cube'",
            "command": "set_pose",
            "payload": {
                "object_name": "Cube",
                "position": {"x": 1.5, "y": 0.5, "z": 1},
                "rotation": {"x": 10, "y": 45, "z": 30}
            }
        },
        {
            "name": "Sending Render Command with custom filename", # Updated task name
            "command": "render",
            "payload": {
                "save_path": "ipc_render_results",
                "width": 1024,
                "height": 1024,
                "filename": "image1" # *** NEW: Add the desired filename here ***
            },
            "is_async": True
        }
    ]

    all_tasks_successful = True
    try:
        for i, task in enumerate(tasks):
            print(f"\n--- [Step {i+1}/{len(tasks)}] {task['name']} ---")
            
            # Send the command and wait for the initial ACK.
            response = client.send_command(
                command=task["command"],
                payload=task["payload"]
            )
            print(f"<-- Received ACK: {response}")

            # Check if the ACK was successful.
            if not response or response.get("status") != "success":
                print(f"!!! [TASK FAILED] Task '{task['name']}' failed with response: {response.get('message', 'No message')}")
                all_tasks_successful = False
                break # Stop processing further tasks.

            print(f"+++ [TASK SUCCESS] Task '{task['name']}' acknowledged successfully.")

            # If the task is asynchronous (like render), wait for the final result.
            if task.get("is_async"):
                print("\n--- Waiting for final result notification ---")
                try:
                    # Set a timeout in case the final result never arrives.
                    client.client_socket.settimeout(60) 
                    final_response_data = client._receive_packet()
                    client.client_socket.settimeout(None)

                    if final_response_data:
                        final_response = json.loads(final_response_data.decode('utf-8'))
                        print(f"<-- Received final notification: {final_response}")
                        # 建议的修改：增加对 type 的检查
                        if final_response.get("type") != "render_notification" or final_response.get("status") != "success":
                            print(f"!!! [TASK FAILED] Final result for '{task['name']}' was unexpected or indicated failure: {final_response.get('message', 'No message')}")
                            all_tasks_successful = False
                            break
                        else:
                            print(f"+++ [TASK SUCCESS] Final result for '{task['name']}' received successfully.")
                    else:
                        print("<-- [ERROR] Did not receive final render notification. The connection may have been lost.")
                        all_tasks_successful = False
                        break
                except socket.timeout:
                    print("<-- [ERROR] Timed out waiting for final render notification.")
                    all_tasks_successful = False
                    break
                except Exception as e:
                    print(f"<-- [ERROR] Error while waiting for final notification: {e}")
                    all_tasks_successful = False
                    break
            
            time.sleep(0.5) # Small delay between commands.

    except Exception as e:
        print(f"\nAn unexpected error occurred during the demo: {e}")
        all_tasks_successful = False
    finally:
        # Print a final summary report.
        print("\n" + "="*50)
        if all_tasks_successful:
            print("✅ [DEMO SUCCEEDED] All communication tasks completed successfully.")
        else:
            print("❌ [DEMO FAILED] One or more communication tasks failed.")
        print("="*50)
        
        print("\nAPI demo finished.")
        client.close()

if __name__ == "__main__":
    run_demo()
