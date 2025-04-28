# udp_server.py
import socket
import platform  # To help suggest finding the IP

# --- Configuration ---
# Use '0.0.0.0' to listen on all available network interfaces (essential for accepting connections from other devices on the same WiFi)
# Alternatively, you could bind to a specific local IP if you know it.
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 9999  # Choose a port number (above 1023 recommended)
BUFFER_SIZE = 1024  # Max amount of data to receive at once


# --- Get Local IP Suggestion ---
# This helps the user know which IP clients might need to connect to
def get_local_ip_suggestion():
    system = platform.system()
    try:
        # Create a temporary socket to find the preferred outbound IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        # Doesn't have to be reachable
        s.connect(('10.254.254.254', 1))
        ip_addr = s.getsockname()[0]
        s.close()
        return ip_addr
    except Exception:
        # Fallback if the above fails
        try:
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            return "?.?.?.? (Could not automatically determine)"


# --- Create and Bind Socket ---
# AF_INET: Use IPv4 addresses
# SOCK_DGRAM: Use UDP protocol
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    # Bind the socket to the specified IP address and port
    server_socket.bind((LISTEN_IP, LISTEN_PORT))

    local_ip = get_local_ip_suggestion()
    print(f"[*] UDP Server listening on {LISTEN_IP}:{LISTEN_PORT}")
    print(f"    Clients on the same WiFi should try connecting to: {local_ip}:{LISTEN_PORT}")
    print(f"    (If {local_ip} doesn't work, find your machine's local IP address manually)")
    print("[*] Waiting for messages... Press Ctrl+C to stop.")

    # --- Main Server Loop ---
    while True:
        # Wait and receive data from a client
        # recvfrom() returns (bytes_data, address_tuple)
        # address_tuple is (client_ip, client_port)
        try:
            data, client_address = server_socket.recvfrom(BUFFER_SIZE)

            # Decode the received bytes into a string (assuming UTF-8 encoding)
            message = data.decode('utf-8')

            print(f"\n[+] Received message from {client_address}:")
            print(f"    Message: '{message}'")

            # --- Optional: Send a response back to the client ---
            response_message = f"Server received: '{message}'"
            # Encode the response string into bytes before sending
            response_bytes = response_message.encode('utf-8')
            server_socket.sendto(response_bytes, client_address)
            print(f"[+] Sent response back to {client_address}")

        except ConnectionResetError:
            # (Windows) Sometimes happens with UDP if the client vanishes unexpectedly
            print(f"[!] Connection reset error from {client_address}. Client might have closed abruptly.")
        except Exception as e:
            print(f"[!] An error occurred during receive/send: {e}")


except OSError as e:
    print(f"[!] Error binding to {LISTEN_IP}:{LISTEN_PORT} - {e}")
    print("    Is the port already in use? Do you have permissions?")
except KeyboardInterrupt:
    print("\n[*] Server shutting down.")
finally:
    # Ensure the socket is closed when the server stops
    print("[*] Closing socket.")
    server_socket.close()
