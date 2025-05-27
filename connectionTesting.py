import pygame
import math
import time
import random
import string
import threading
import socket
import struct
import queue

from text import drawText
from fontDict import fonts

# --- Networking helpers -----------------------------------------------------

MSG_QUEUE = queue.Queue()  # all received messages end up here

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase  # Base62
BASE = len(ALPHABET)


def base62_encode(number):
    if number == 0:
        return ALPHABET[0]
    result = []
    while number > 0:
        number, rem = divmod(number, BASE)
        result.append(ALPHABET[rem])
    return ''.join(reversed(result))


def base62_decode(s):
    number = 0
    for char in s:
        number = number * BASE + ALPHABET.index(char)
    return number


def make_short_code(ip_suffix, port):
    """ip_suffix: tuple of (3rd_octet, 4th_octet), e.g., (1, 170)"""
    ip_bytes = bytes(ip_suffix)  # 2 bytes
    port_bytes = struct.pack(">H", port)  # 2 bytes
    combined = ip_bytes + port_bytes  # 4 bytes
    num = int.from_bytes(combined, 'big')
    return base62_encode(num).zfill(6)  # padded to fixed length


def decode_short_code(code):
    num = base62_decode(code)
    full_bytes = num.to_bytes(4, 'big')
    ip_suffix = list(full_bytes[:2])
    port = struct.unpack(">H", full_bytes[2:])[0]
    ip = f"192.168.{ip_suffix[0]}.{ip_suffix[1]}"
    return ip, port


# --- helper to pick an unused port ----------------------------------------
def find_free_port(start_port, max_tries=100):
    for p in range(start_port, start_port + max_tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("0.0.0.0", p))
            s.close()
            return p
        except OSError:
            continue
    raise RuntimeError(f"No free UDP port in {start_port}–{start_port+max_tries}")


def get_local_ip_suggestion():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    s.connect(('10.254.254.254', 1))
    ip = s.getsockname()[0]
    s.close()
    return ip


def server_thread(listen_ip, listen_port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((listen_ip, listen_port))
    while True:
        data, address = sock.recvfrom(1024)
        MSG_QUEUE.put((address, data.decode()))


def client_recv_thread(sock):
    while True:
        try:
            data, address = sock.recvfrom(1024)
            MSG_QUEUE.put((address, data.decode()))
        except socket.timeout:
            continue


# ---------------- Setting up the screen, assigning some global variables, and loading text fonts
pygame.init()
screen = pygame.display.set_mode((800, 500))
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 1
screen_width = int(screen.get_width() / scaleDownFactor)
screen_height = int(screen.get_height() / scaleDownFactor)
screen_center = [screen_width / 2, screen_height / 2]
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT.set_alpha(100)
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()
timer = 0
shake = [0, 0]
shake_strength = 3

loaded_fonts = {}
for name, (path, size) in fonts.items():
    loaded_fonts[name] = pygame.font.Font(path, int(size / scaleDownFactor))

Alkhemikal20 = loaded_fonts.get('Alkhemikal20')
Alkhemikal40 = loaded_fonts.get('Alkhemikal40')
Alkhemikal80 = loaded_fonts.get('Alkhemikal80')


class Endesga:
    black = [19, 19, 19]
    white = [255, 255, 255]
    greyL = [200, 200, 200]
    grey = [150, 150, 150]
    my_blue = [32, 36, 46]
    debug_red = [255, 96, 141]


# Defining some more variables to use in the game loop
oscillating_random_thing = 0
ShakeCounter = 0
toggle = True
click = False

# Keyboard input
userString = ""
userStringErrorDisplay = None
keyHoldFrames = {}
delayThreshold = 10
shifting = False

# Connection
username = ''.join(random.choice(string.ascii_uppercase) for _ in range(5))
local_ip = get_local_ip_suggestion()[8:]
connectingPort = 4000 + random.randint(0, 999)
room_code = ""
mode = "INIT"  # INIT / HOST_LOBBY / CLIENT_LOBBY / IN_GAME
players = {}  # host: addr→name
joined = False  # client flag
server_socket = None

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.settimeout(0.5)
threading.Thread(target=client_recv_thread, args=(client_socket,), daemon=True).start()

seed_to_send = None
seed = None

requestSentTime = None

# ---------------- Main Game Loop
last_time = time.time()
running = True
while running:

    # ---------------- Reset Variables and Clear screens
    mx, my = pygame.mouse.get_pos()
    mx, my = mx / scaleDownFactor, my / scaleDownFactor
    screen.fill(Endesga.my_blue)
    screen2.fill(Endesga.my_blue)
    screenT.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))
    dt = time.time() - last_time
    dt *= fps
    last_time = time.time()
    timer -= 1 * dt
    shake = [random.uniform(-1, 1) * shake_strength, random.uniform(-1, 1) * shake_strength] if timer > 0 else [0, 0]
    oscillating_random_thing += math.pi / fps * dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
            if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                shifting = True

            if event.key == pygame.K_RETURN:
                txt = userString.strip()

                if mode == "INIT":
                    if len(userString) != 6 and userString != 'start':
                        timer = 0.5 * fps
                        userStringErrorDisplay = "type smth" if userString == "" else "that's not a code"
                        continue
                    if txt.lower() == "start":

                        # host: pick an available port and fire up the UDP listener
                        connectingPort = find_free_port(connectingPort)
                        threading.Thread(target=server_thread, args=("0.0.0.0", connectingPort), daemon=True).start()

                        mode = "HOST_LOBBY"
                        ipSuffix = tuple(map(int, local_ip.split('.')))
                        room_code = make_short_code(ipSuffix, connectingPort)
                    else:
                        host_ip, host_port = decode_short_code(txt)
                        client_socket.sendto(f"JOIN:{username}".encode(), (host_ip, host_port))
                        mode = "CLIENT_LOBBY"
                        requestSentTime = time.time()

                elif mode == "HOST_LOBBY":
                    if userString != 'begin':
                        timer = 0.5 * fps
                        userStringErrorDisplay = "you gotta type smth" if userString == "" else "that's not 'begin'"
                        continue
                    # host starts game
                    seed_to_send = random.randint(0, 2 ** 31 - 1)
                    for addr in players:
                        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        server_socket.sendto(f"SEED:{seed_to_send}".encode(), addr)
                    mode = "IN_GAME"
                userString = ""

            elif event.key not in keyHoldFrames:
                keyHoldFrames[event.key] = 0
        if event.type == pygame.KEYUP:
            keyHoldFrames.pop(event.key, None)
            if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                shifting = False

    for key, hold_time in list(keyHoldFrames.items()):
        keyHoldFrames[key] += 1
        if hold_time == 0 or hold_time > delayThreshold:
            if key == pygame.K_BACKSPACE:
                if userString:
                    userString = userString[:-1]
            elif pygame.K_0 <= key <= pygame.K_9:
                if len(userString) < 6:
                    userString += chr(key)
            elif pygame.K_a <= key <= pygame.K_z:
                if len(userString) < 6:
                    userString += chr(key - 32 * shifting)

            if hold_time > delayThreshold:
                keyHoldFrames[key] = delayThreshold

    if mode == "CLIENT_LOBBY" and not joined:
        if time.time() - requestSentTime > 2.0:
            timer = 1.0 * fps
            userString = ""
            userStringErrorDisplay = "lobby doesn't exist"
            requestSentTime = None
            mode = "INIT"

    try:
        addr, msg = MSG_QUEUE.get_nowait()
    except queue.Empty:
        pass
    else:
        if mode == "HOST_LOBBY":
            if msg.startswith("JOIN:"):
                name = msg.split(":", 1)[1]
                players[addr] = name
                # ack
                ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ack_sock.sendto(b"ACK_JOIN", addr)
        elif mode == "CLIENT_LOBBY":
            if msg == "ACK_JOIN":
                joined = True
            elif msg.startswith("SEED:"):
                seed = int(msg.split(":", 1)[1])
                mode = "IN_GAME"

    if mode == "INIT":
        if timer < 0:
            userStringErrorDisplay = ("->  <-" if userString == "" else None)
        prompt = "type 'start' to host or enter a code to join"
        drawText(screen2, Endesga.grey, Alkhemikal40, screen_center[0], 200, prompt, Endesga.black, 2, justify="middle")
        drawText(screen2, Endesga.debug_red if timer > 0 else Endesga.greyL, Alkhemikal80, screen_center[0] + shake[0], 280 + shake[1], userString if userStringErrorDisplay is None else userStringErrorDisplay, Endesga.black, 2, justify="middle")

    elif mode == "HOST_LOBBY":
        drawText(screen2, Endesga.greyL, Alkhemikal40, screen_center[0], 50, "HOST LOBBY", Endesga.black, 2, justify="middle")
        drawText(screen2, Endesga.grey, Alkhemikal40, screen_center[0], 120, f"Room code: {room_code}", Endesga.black, 2, justify="middle")
        drawText(screen2, Endesga.grey, Alkhemikal20, screen_center[0], 160, f"Players joined:", Endesga.black, 1, justify="middle")
        for i, name in enumerate(players.values(), start=1):
            drawText(screen2, Endesga.white, Alkhemikal20, screen_center[0], 160 + i * 20, name, Endesga.black, 1, justify="middle")
        drawText(screen2, Endesga.grey, Alkhemikal40, screen_center[0], 440, "type 'begin' + ENTER to start", Endesga.black, 2, justify="middle")

        if timer < 0:
            userStringErrorDisplay = ("->  <-" if userString == "" else None)
        drawText(screen2, Endesga.debug_red if timer > 0 else Endesga.greyL, Alkhemikal80, screen_center[0] + shake[0], 280 + shake[1], userString if userStringErrorDisplay is None else userStringErrorDisplay, Endesga.black, 2, justify="middle")

    elif mode == "CLIENT_LOBBY":
        drawText(screen2, Endesga.greyL, Alkhemikal40, screen_center[0], 200, "CLIENT LOBBY", Endesga.black, 2, justify="middle")
        status = "JOINED! waiting for host..." if joined else "joining..."
        drawText(screen2, Endesga.grey, Alkhemikal40, screen_center[0], 260, status, Endesga.black, 2, justify="middle")

    elif mode == "IN_GAME":
        drawText(screen2, Endesga.greyL, Alkhemikal40, screen_center[0], 200, "GAME START!", Endesga.black, 2, justify="middle")
        if seed_to_send is not None:
            drawText(screen2, Endesga.grey, Alkhemikal20, screen_center[0], 240, f"you are host, seed={seed_to_send}", Endesga.black, 1, justify="middle")
        else:
            drawText(screen2, Endesga.grey, Alkhemikal20, screen_center[0], 240, f"you are client, seed={seed}", Endesga.black, 1, justify="middle")

    # ---------------- Updating Screen
    if toggle:
        drawText(screenUI, Endesga.greyL, Alkhemikal40, screen_width / 2, 30, f"your name is {username}", Endesga.black, 3, justify="middle", centeredVertically=True)
        items = {round(clock.get_fps()): None, }
        for i, label in enumerate(items.keys()):
            string = str(label)
            if items[label] is not None:
                string = f"{items[label]}: " + string
            drawText(screenUI, Endesga.debug_red, Alkhemikal20, 5, screen_height - (20 + 15 * i) / (scaleDownFactor ** (1 / 1.8)), string, Endesga.black, int(3 / scaleDownFactor) + int(3 / scaleDownFactor) < 1, antiAliasing=False)
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, Endesga.black, (mx + 1, my + 1), 5, 2)
        pygame.draw.circle(screenUI, Endesga.white, (mx, my), 5, 2)
    screen.blit(pygame.transform.scale(screen2, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    screen.blit(pygame.transform.scale(screenT, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    screen.blit(pygame.transform.scale(screenUI, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    pygame.display.update()
    clock.tick(fps)
