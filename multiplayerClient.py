# udp_client.py
import socket

# Prompt user for server IP
server_ip_input = input("Enter the server's local IP address (e.g., 192.168.1.5): ")
SERVER_IP = server_ip_input.strip() if server_ip_input else "127.0.0.1"  # Fallback to localhost

SERVER_PORT = 9999  # Must match the server's listening port
BUFFER_SIZE = 1024

# --- Create Socket ---
# AF_INET: Use IPv4 addresses
# SOCK_DGRAM: Use UDP protocol
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Optional: Set a timeout for receiving responses ---
# If the server doesn't reply within 5 seconds, recvfrom will raise a timeout error
client_socket.settimeout(5.0)

print(f"[*] Attempting to send messages to UDP Server at {SERVER_IP}:{SERVER_PORT}")
print("    Type your message and press Enter to send.")
print("    Type 'quit' or 'exit' to stop.")

try:
    while True:
        # Get message input from the user
        message = input("Message> ")
        if message.lower() in ['quit', 'exit']:
            break

        # Encode the message string into bytes
        message_bytes = message.encode('utf-8')

        try:
            # Send the message bytes to the server's address
            client_socket.sendto(message_bytes, (SERVER_IP, SERVER_PORT))
            print(f"[+] Message sent to {SERVER_IP}:{SERVER_PORT}")

            # --- Optional: Wait for a response from the server ---
            print("[*] Waiting for server response...")
            try:
                # Wait to receive data back from the server
                data, server_address = client_socket.recvfrom(BUFFER_SIZE)
                response_message = data.decode('utf-8')
                print(f"[+] Received response from {server_address}: '{response_message}'")

            except socket.timeout:
                print("[!] No response received from the server (timeout).")
            except Exception as e:
                print(f"[!] Error receiving response: {e}")

        except socket.gaierror:
            print(f"[!] Error: Hostname or IP address '{SERVER_IP}' could not be resolved. Is it correct?")
            break  # Exit if the address is invalid
        except Exception as e:
            print(f"[!] An error occurred during send: {e}")


except KeyboardInterrupt:
    print("\n[*] Client shutting down.")
finally:
    # Ensure the socket is closed when the client stops
    print("[*] Closing socket.")
    client_socket.close()
