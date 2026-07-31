import requests
import time

class ControllerClient:
    def __init__(self, host="12.216.3.117", port=11434, api_key="ps-voice-secret-key"):
        self.base_url = f"http://{host}:{port}"
        self.headers = {"X-API-Key": api_key}

    def check_status(self):
        """Check if the server is running and get its PID."""
        try:
            response = requests.get(f"{self.base_url}/status", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error checking status: {e}")
            return None

    def start_server(self):
        """Start the remote server."""
        try:
            response = requests.post(f"{self.base_url}/start", headers=self.headers)
            return response.json()
        except requests.RequestException as e:
            print(f"Error starting server: {e}")
            return None

    def stop_server(self):
        """Stop the remote server."""
        try:
            response = requests.post(f"{self.base_url}/stop", headers=self.headers)
            return response.json()
        except requests.RequestException as e:
            print(f"Error stopping server: {e}")
            return None

    def restart_server(self):
        """Restart the remote server."""
        try:
            response = requests.post(f"{self.base_url}/restart", headers=self.headers)
            return response.json()
        except requests.RequestException as e:
            print(f"Error restarting server: {e}")
            return None

if __name__ == "__main__":
    # Initialize the client pointing to your controller API
    client = ControllerClient(host="12.216.3.117", port=11434)

    print("--- 1. Checking Initial Status ---")
    print(client.check_status())

    print("\n--- 2. Starting the Server ---")
    print(client.start_server())

    print("\n--- 3. Checking Status After Start ---")
    time.sleep(1) # Give it a second to spin up
    print(client.check_status())

    # Uncomment below to test stopping or restarting
    # print("\n--- 4. Restarting the Server ---")
    # print(client.restart_server())

    # print("\n--- 5. Stopping the Server ---")
    # print(client.stop_server())
