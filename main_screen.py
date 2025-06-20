import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pygame
import time
import sys
import os
import csv
import statistics
import socket
import struct
import queue
import threading
import string
import random

from text import drawText
from fontDict import fonts as fonts_definitions
from controlPanel import GenerationInfo, ResourceInfo, StructureInfo, Cols
from player import Player
from calcs import normalize

MSG_QUEUE = queue.Queue()

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
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
    ip_bytes = bytes(ip_suffix)
    port_bytes = struct.pack(">H", port)
    combined = ip_bytes + port_bytes
    num = int.from_bytes(combined, 'big')
    return base62_encode(num).zfill(6)


def decode_short_code(code):
    num = base62_decode(code)
    full_bytes = num.to_bytes(4, 'big')
    ip_suffix = list(full_bytes[:2])
    port = struct.unpack(">H", full_bytes[2:])[0]
    ip = f"192.168.{ip_suffix[0]}.{ip_suffix[1]}"
    return ip, port


def find_free_port(start_port, max_tries=100):
    for p in range(start_port, start_port + max_tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("0.0.0.0", p))
            s.close()
            return p
        except OSError:
            continue
    raise RuntimeError(f"No free UDP port in {start_port}–{start_port + max_tries}")


def get_local_ip_suggestion():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    s.connect(('10.254.254.254', 1))
    ip = s.getsockname()[0]
    s.close()
    return ip


def server_thread(sock_instance, listen_ip, listen_port):
    while True:
        try:
            data, address = sock_instance.recvfrom(1024)
            MSG_QUEUE.put((address, data.decode()))
        except socket.timeout:
            continue
        except OSError as e:
            print(f"Server thread error: {e}. Exiting server thread.")
            break
        except Exception as e:
            print(f"Unexpected error in server thread: {e}")
            break


def client_recv_thread(sock):
    while True:
        try:
            data, address = sock.recvfrom(1024)
            MSG_QUEUE.put((address, data.decode()))
        except socket.timeout:
            continue
        except OSError as e:
            print(f"Client receive thread error: {e}. Exiting client receive thread.")
            break
        except Exception as e:
            print(f"Unexpected error in client receive thread: {e}")
            break


TIMES_CSV_FILE = "execution_times.csv"
INITIAL_PRESET_PLACEHOLDER_TIME = 999.0

PRESET_EXECUTION_TIMES = {}


def load_and_calculate_average_times():
    global PRESET_EXECUTION_TIMES
    new_preset_times = {}
    all_step_durations = {}

    if os.path.exists(TIMES_CSV_FILE):
        try:
            with open(TIMES_CSV_FILE, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames and 'step_name' in reader.fieldnames and 'duration' in reader.fieldnames:
                    for row in reader:
                        try:
                            step_name = row['step_name']
                            duration = float(row['duration'])
                            if step_name not in all_step_durations:
                                all_step_durations[step_name] = []
                            all_step_durations[step_name].append(duration)
                        except (ValueError, KeyError):
                            pass

            for step_name, durations in all_step_durations.items():
                if durations:
                    new_preset_times[step_name] = statistics.mean(durations)

            if new_preset_times:
                PRESET_EXECUTION_TIMES = new_preset_times
                return
        except Exception as e_csv_load:
            print(f"Error loading or processing '{TIMES_CSV_FILE}': {e_csv_load}. Using default placeholder times.")
    PRESET_EXECUTION_TIMES = {}


def save_execution_times(new_times_dict):
    if not new_times_dict:
        return
    file_exists = os.path.exists(TIMES_CSV_FILE)
    try:
        with open(TIMES_CSV_FILE, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'step_name', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(TIMES_CSV_FILE) == 0:
                writer.writeheader()

            current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            for step_name, duration in new_times_dict.items():
                writer.writerow({'timestamp': current_timestamp, 'step_name': step_name, 'duration': duration})
    except Exception as e_csv_save:
        print(f"Error saving execution times to '{TIMES_CSV_FILE}': {e_csv_save}")


load_and_calculate_average_times()


def build_tile_handler_worker(args_tuple):
    map_width, map_height, viewport_width, viewport_height, gen_info, font_name_to_load, font_definitions_dict, cols_class, resource_info_class, structure_info_class, local_status_q, current_preset_times, worker_seed = args_tuple

    try:
        from generation import TileHandler
    except ImportError as e_import:
        if local_status_q:
            local_status_q.put_nowait(("Error: Import Failed in Worker (TileHandler)", "ERROR", str(e_import)))
        raise
    _font = None
    if font_name_to_load and font_name_to_load in font_definitions_dict:
        try:
            pygame.init()
            font_path, font_size = font_definitions_dict[font_name_to_load]
            _font = pygame.font.Font(font_path, font_size)
        except Exception as e_font:
            print(f"Worker: Error loading font '{font_name_to_load}': {e_font}")
    TH_instance = TileHandler(map_width, map_height, gen_info.tileSize, cols_class, gen_info.waterThreshold, gen_info.mountainThreshold, gen_info.territorySize, font=_font, font_name=font_name_to_load, resource_info=resource_info_class, structure_info=structure_info_class, status_queue=local_status_q, preset_times=current_preset_times, seed=worker_seed, viewport_width=viewport_width, viewport_height=viewport_height)
    TH_instance.prepare_for_pickling()
    return TH_instance


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    multiprocessing.freeze_support()

    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    fps = 60
    screen_width, screen_height = screen.get_width(), screen.get_height()
    screen_center = [screen_width / 2, screen_height / 2]

    screen2_loading_background = pygame.Surface((screen_width, screen_height)).convert_alpha()
    screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()

    loaded_fonts = {}
    for name, (path, size) in fonts_definitions.items():
        try:
            if not os.path.exists(path):
                continue
            loaded_fonts[name] = pygame.font.Font(path, size)
        except Exception as e_font_load:
            print(f"Main Error: Failed to load font {name} from {path} (size {size}): {e_font_load}")

    Alkhemikal20 = loaded_fonts.get('Alkhemikal20')
    Alkhemikal30 = loaded_fonts.get('Alkhemikal30')
    Alkhemikal50 = loaded_fonts.get('Alkhemikal50')
    Alkhemikal80 = loaded_fonts.get('Alkhemikal80')
    Alkhemikal150 = loaded_fonts.get('Alkhemikal150')
    Alkhemikal200 = loaded_fonts.get('Alkhemikal200')

    player = None

    generationScreenBackgroundImg = pygame.transform.scale(pygame.image.load("assets/UI/LoadingPageBackground.png"), (screen_width, screen_height))

    manager = multiprocessing.Manager()
    status_queue_for_main_thread = manager.Queue()

    executor = ProcessPoolExecutor(max_workers=1)
    font_name_needed_by_worker = 'Alkhemikal30'

    username = ''.join(random.choice(string.ascii_uppercase) for _ in range(5))
    local_ip_full = get_local_ip_suggestion()
    local_ip_suffix = tuple(map(int, local_ip_full.split('.')))[2:]
    connectingPort = 4000 + random.randint(0, 999)

    room_code = ""
    mode = "INIT"
    players = {}
    joined = False
    server_socket = None
    server_thread_instance = None

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(0.5)
    threading.Thread(target=client_recv_thread, args=(client_socket,), daemon=True).start()

    seed_to_send = None
    seed = None
    requestSentTime = None

    target_host_ip = None
    target_host_port = None

    client_ping_interval = 0.5
    client_timeout_threshold = 2.0
    last_ping_sent_time = 0.0
    client_last_ping_time = {}
    last_host_check_time = 0.0

    toggle = True
    userString = ""
    userStringErrorDisplay = None
    keyHoldFrames = {}
    delayThreshold = 10
    shifting = False
    lobby_timer_for_error_display = 0

    future = None
    numPeriods = 0
    PHASE_WORKER_INIT = "Initializing World Generation"
    PHASE_DATA_TRANSFER_PREP = "Preparing Data for Transfer"
    PHASE_RETRIEVING_MAP_DATA = "Retrieving World Data"
    PHASE_GFX_INIT = "Initializing Graphics"

    LOADING_STEPS_ORDER = ["tileGen", "linkAdj", "generationCycles", "setTileColors", "cloudPrecompParallel", "findLandRegionsParallel", "indexOceansParallel", "assignCoastTiles", "createTerritories", "connectHarborsParallel", "precomputeTerritoryVision", "workerInit", "dataSerialization", "retrieveMapData", "gfxTotalInit"]

    DISPLAY_NAMES_MAP = {"tileGen": "Generating Tiles", "linkAdj": "Connecting Adjacent Tiles", "cloudPrecompParallel": "Precomp Cloud Patterns (Parallel)", "generationCycles": "Simulating Biomes (50 cycles)", "setTileColors": "Coloring Map Tiles", "findLandRegionsParallel": "Identifying Landmasses (Parallel)", "indexOceansParallel": "Indexing Oceans (Parallel)", "assignCoastTiles": "Assigning Coastline Tiles", "createTerritories": "Forming Territories",
        "connectHarborsParallel": "Connecting Harbors (Parallel)", "precomputeTerritoryVision": "Precomputing Territory Vision", "workerInit": "World Generation Complete (Worker)", "dataSerialization": "Serializing World Data", "retrieveMapData": "Retrieving World Data", "gfxTotalInit": "Initializing Game Graphics"}

    task_display_states = {}
    for step_name_key in LOADING_STEPS_ORDER:
        task_display_states[step_name_key] = {'status': 'Pending', 'start_time': 0.0, 'duration': 0.0, 'expected_time': PRESET_EXECUTION_TIMES.get(step_name_key, INITIAL_PRESET_PLACEHOLDER_TIME)}

    loading_screen_start_time = time.time()

    TH_fully_initialized = False
    TH = None
    all_current_run_times = {}

    worker_tasks_complete = False
    retrieving_result_active = False

    main_title_x = screen_width * 0.25
    main_overall_phase_x = screen_width * 0.25

    tasks_list_x = screen_width * 0.5
    line_height = 30
    progress_bar_width = screen_width * 0.17
    progress_bar_height = 15
    progress_bar_corner_radius = int(progress_bar_height / 3)
    progress_bar_y_offset = 5

    single_task_display_start_y = screen_center[1] - (len(LOADING_STEPS_ORDER) * line_height / 2)

    last_time = time.time()
    running = True
    pygame.mouse.set_visible(False)

    screen2_loading_background.blit(generationScreenBackgroundImg, (0, 0))
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 8, 10, 150))
    screen2_loading_background.blit(overlay, (0, 0))
    screen.blit(screen2_loading_background, (0, 0))

    while running:
        screen.fill(Cols.dark)
        screenUI.fill((0, 0, 0, 0))
        dt = time.time() - last_time
        dt *= fps
        last_time = time.time()
        mx, my = pygame.mouse.get_pos()

        lobby_timer_for_error_display -= 1 * dt

        try:
            while not MSG_QUEUE.empty():
                addr, msg = MSG_QUEUE.get_nowait()
                if mode == "HOST_LOBBY":
                    if msg.startswith("JOIN:"):
                        name = msg.split(":", 1)[1]
                        players[addr] = name
                        client_last_ping_time[addr] = time.time()
                        if server_socket:
                            server_socket.sendto(b"ACK_JOIN", addr)
                        print(f"Main: Player '{name}' joined from {addr}")
                    elif msg.startswith("PING:"):
                        if addr in players:
                            client_last_ping_time[addr] = time.time()

                elif mode == "CLIENT_LOBBY":
                    if msg == "ACK_JOIN":
                        joined = True
                        last_ping_sent_time = time.time()
                        print("Main: Successfully joined lobby")
                    elif msg.startswith("SEED:"):
                        seed = int(msg.split(":", 1)[1])
                        mode = "IN_GAME"
                        loading_screen_start_time = time.time()
                        print(f"Main: Client received seed {seed}. Starting generation.")
        except queue.Empty:
            pass
        except Exception as e_queue_process:
            print(f"Main Error processing network queue: {e_queue_process}")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                click = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_TAB:
                    toggle = not toggle
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shifting = True

                if event.key == pygame.K_RETURN:
                    txt = userString.strip()

                    if mode == "INIT":
                        if len(userString) != 6 and userString.lower() not in ['start', 's']:
                            lobby_timer_for_error_display = 0.5 * fps
                            userStringErrorDisplay = "type smth" if userString == "" else "that's not a code"
                            continue
                        if txt.lower() in ["start", "s"]:
                            connectingPort = find_free_port(connectingPort)
                            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            server_socket.bind(("0.0.0.0", connectingPort))
                            server_thread_instance = threading.Thread(target=server_thread, args=(server_socket, "0.0.0.0", connectingPort), daemon=True)
                            server_thread_instance.start()

                            mode = "HOST_LOBBY"
                            room_code = make_short_code(local_ip_suffix, connectingPort)
                            userString = ""
                            userStringErrorDisplay = None
                            target_host_ip = local_ip_full
                            target_host_port = connectingPort
                        else:
                            try:
                                host_ip, host_port = decode_short_code(txt)
                                client_socket.sendto(f"JOIN:{username}".encode(), (host_ip, host_port))
                                mode = "CLIENT_LOBBY"
                                requestSentTime = time.time()
                                userString = ""
                                userStringErrorDisplay = None
                                target_host_ip = host_ip
                                target_host_port = host_port
                            except Exception:
                                lobby_timer_for_error_display = 0.5 * fps
                                userStringErrorDisplay = "invalid code"
                                continue

                    elif mode == "HOST_LOBBY":
                        if txt.lower() in ["begin", "b"]:
                            seed_to_send = random.randint(0, 2 ** 31 - 1)
                            for addr in players:
                                if server_socket:
                                    server_socket.sendto(f"SEED:{seed_to_send}".encode(), addr)
                            mode = "IN_GAME"
                            seed = seed_to_send
                            loading_screen_start_time = time.time()
                            userString = ""
                            userStringErrorDisplay = None
                            print(f"Main: Host starting game with seed {seed}")
                        elif txt.lower() in ['quit', 'q']:
                            print("Host: Returning to INIT screen. Closing server.")
                            if server_socket:
                                server_socket.close()
                                server_socket = None

                            mode = "INIT"
                            room_code = ""
                            players = {}
                            client_last_ping_time = {}
                            seed_to_send = None
                            seed = None
                            requestSentTime = None
                            target_host_ip = None
                            target_host_port = None
                            userString = ""
                            userStringErrorDisplay = None
                            lobby_timer_for_error_display = 0
                            continue

                        else:
                            lobby_timer_for_error_display = 0.5 * fps
                            userStringErrorDisplay = "you gotta type smth" if userString == "" else "that's not 'begin' or 'quit'"
                            continue

                    elif mode == "CLIENT_LOBBY":
                        if userString.lower() == 'quit':
                            print("Client: Returning to INIT screen.")
                            mode = "INIT"
                            joined = False
                            requestSentTime = None
                            target_host_ip = None
                            target_host_port = None
                            userString = ""
                            userStringErrorDisplay = None
                            lobby_timer_for_error_display = 0
                            last_ping_sent_time = 0.0
                            continue
                        else:
                            lobby_timer_for_error_display = 0.5 * fps
                            userStringErrorDisplay = "press enter to continue" if userString == "" else "no other input for client"
                            continue

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
                        if shifting:
                            userString += chr(key - 32)
                        else:
                            userString += chr(key)

                if hold_time > delayThreshold:
                    keyHoldFrames[key] = delayThreshold

        if mode != "IN_GAME":
            if mode == "CLIENT_LOBBY" and joined:
                if time.time() - last_ping_sent_time > client_ping_interval:
                    if target_host_ip and target_host_port:
                        try:
                            client_socket.sendto(f"PING:{username}".encode(), (target_host_ip, target_host_port))
                            last_ping_sent_time = time.time()
                        except OSError as err:
                            print(f"Client: Error sending ping: {err}. Host likely disconnected.")
                            lobby_timer_for_error_display = 1.0 * fps
                            userString = ""
                            userStringErrorDisplay = "Host disconnected or timed out"
                            requestSentTime = None
                            mode = "INIT"
                    else:
                        print("Client: Cannot send ping, target_host_ip/port not set after joining. Resetting.")
                        mode = "INIT"
                        userStringErrorDisplay = "Connection error"

            if mode == "CLIENT_LOBBY" and not joined:
                if requestSentTime and time.time() - requestSentTime > 2.0:
                    lobby_timer_for_error_display = 1.0 * fps
                    userString = ""
                    userStringErrorDisplay = "lobby doesn't exist or timed out"
                    requestSentTime = None
                    mode = "INIT"

            if mode == "HOST_LOBBY" and time.time() - last_host_check_time > 1.0:
                disconnected_players = []
                for addr, last_time_ping in client_last_ping_time.items():
                    if time.time() - last_time_ping > client_timeout_threshold:
                        disconnected_players.append(addr)

                for addr_to_remove in disconnected_players:
                    if addr_to_remove in players:
                        print(f"Host: Player '{players[addr_to_remove]}' ({addr_to_remove}) timed out.")
                        del players[addr_to_remove]
                    del client_last_ping_time[addr_to_remove]
                last_host_check_time = time.time()

            if mode == "INIT":
                drawText(screenUI, Cols.crimson, Alkhemikal200, main_title_x, screen_center[1] - 80, "Crimson", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
                drawText(screenUI, Cols.crimson, Alkhemikal200, main_title_x, screen_center[1] + 80, "Wakes", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
                if lobby_timer_for_error_display < 0:
                    userStringErrorDisplay = (f"-> {'~' if int(lobby_timer_for_error_display / fps * 2) % 2 else '  '} <-" if userString == "" else None)
                prompt = "type 'start' to host or enter a code to join"
                drawText(screenUI, Cols.light, Alkhemikal50, screen_width * 0.75, screen_center[1] - 40, prompt, Cols.dark, 3, justify="middle", centeredVertically=True, maxLen=screen_width / 3, wrap=True)
                drawText(screenUI, Cols.crimson if lobby_timer_for_error_display > 0 else Cols.light, Alkhemikal80 if userStringErrorDisplay else Alkhemikal200, screen_width * 0.75, screen_center[1] + 100, userString if userStringErrorDisplay is None else userStringErrorDisplay, Cols.dark, 3, justify="middle", centeredVertically=True)

            elif mode == "HOST_LOBBY":
                drawText(screenUI, Cols.crimson, Alkhemikal150, screen_center[0], screen_center[1] - 260, "HOST LOBBY", Cols.dark, 3, justify="middle", centeredVertically=True)
                drawText(screenUI, Cols.light, Alkhemikal80, screen_center[0], screen_center[1] - 160, f"Room code: {room_code}", Cols.dark, 3, justify="middle", centeredVertically=True)
                drawText(screenUI, Cols.light, Alkhemikal20, screen_center[0], screen_center[1] - 100, f"Players joined:", Cols.dark, 3, justify="middle", centeredVertically=True)

                current_players_list = list(players.values())
                drawText(screenUI, Cols.light, Alkhemikal20, screen_center[0], screen_center[1] - 90 + 1 * 25, f"{username} (You)", Cols.dark, 3, justify="middle", centeredVertically=True)
                for i, name in enumerate(current_players_list, start=2):
                    drawText(screenUI, Cols.light, Alkhemikal20, screen_center[0], screen_center[1] - 90 + i * 25, name, Cols.dark, 3, justify="middle", centeredVertically=True)

                drawText(screenUI, Cols.light, Alkhemikal30, screen_center[0], screen_center[1] + 280, "type 'begin' to start or 'quit' to exit", Cols.dark, 3, justify="middle", centeredVertically=True)

                if lobby_timer_for_error_display < 0:
                    userStringErrorDisplay = (f"-> {'~' if int(lobby_timer_for_error_display / fps * 2) % 2 else '  '} <-" if userString == "" else None)
                drawText(screenUI, Cols.crimson if lobby_timer_for_error_display > 0 else Cols.light, Alkhemikal80 if userStringErrorDisplay else Alkhemikal200, screen_center[0], screen_center[1] + 180, userString if userStringErrorDisplay is None else userStringErrorDisplay, Cols.dark, 3, justify="middle", centeredVertically=True)

            elif mode == "CLIENT_LOBBY":
                drawText(screenUI, Cols.light, Alkhemikal50, screen_center[0], screen_center[1] - 100, "CLIENT LOBBY", Cols.dark, 3, justify="middle", centeredVertically=True)
                status = "JOINED! waiting for host..." if joined else "joining..."
                drawText(screenUI, Cols.light, Alkhemikal30, screen_center[0], screen_center[1], status, Cols.dark, 3, justify="middle", centeredVertically=True)
                drawText(screenUI, Cols.light, Alkhemikal30, screen_center[0], screen_center[1] + 180, "type 'quit' to exit", Cols.dark, 3, justify="middle", centeredVertically=True)
                drawText(screenUI, Cols.crimson if lobby_timer_for_error_display > 0 else Cols.light, Alkhemikal80 if userStringErrorDisplay else Alkhemikal200, screen_center[0], screen_center[1] + 80, userString if userStringErrorDisplay is None else userStringErrorDisplay, Cols.dark, 3, justify="middle", centeredVertically=True)

            if toggle:
                string_fps = f"FPS: {round(clock.get_fps())}"
                drawText(screenUI, Cols.crimson, Alkhemikal30, 5, screen_height - 30, string_fps, Cols.dark, 3, antiAliasing=False)
                drawText(screenUI, Cols.light, Alkhemikal30, screen_width / 2, 30, f"your name is {username}", Cols.dark, 3, justify="middle", centeredVertically=True)
                pygame.draw.circle(screenUI, Cols.dark, (mx + 2, my + 2), 7, 2)
                pygame.draw.circle(screenUI, Cols.light, (mx, my), 7, 2)

            screen.blit(screen2_loading_background, (0, 0))
            screen.blit(overlay, (0, 0))
            screen.blit(screenUI, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
            continue

        if future is None:
            print(f"Main: Submitting TileHandler generation task to worker with seed: {seed}.")
            map_gen_width = int(screen_width * GenerationInfo.mapSizeScalar)
            map_gen_height = int(screen_height * GenerationInfo.mapSizeScalar)
            worker_args = (map_gen_width, map_gen_height, screen_width, screen_height, GenerationInfo, font_name_needed_by_worker, fonts_definitions, Cols, ResourceInfo, StructureInfo, status_queue_for_main_thread, PRESET_EXECUTION_TIMES, seed)
            future = executor.submit(build_tile_handler_worker, worker_args)
            loading_screen_start_time = time.time()

        if not TH_fully_initialized:
            numPeriods = (numPeriods + 3 / fps) % 4

            if not worker_tasks_complete and future.done() and not retrieving_result_active:
                worker_tasks_complete = True
                retrieving_result_active = True
                task_data = task_display_states["retrieveMapData"]
                task_data['status'] = 'Starting'
                task_data['start_time'] = time.time()
                task_data['expected_time'] = PRESET_EXECUTION_TIMES.get("retrieveMapData", INITIAL_PRESET_PLACEHOLDER_TIME)

            elif retrieving_result_active:
                actual_retrieval_start_time = time.time()
                try:
                    TH = future.result()
                    retrieval_duration = time.time() - actual_retrieval_start_time
                    all_current_run_times["retrieveMapData"] = retrieval_duration

                    task_data = task_display_states["retrieveMapData"]
                    task_data['status'] = 'Finished'
                    task_data['duration'] = retrieval_duration

                    retrieving_result_active = False

                    if TH and hasattr(TH, 'execution_times'):
                        all_current_run_times.update(TH.execution_times)

                    task_data_gfx_total = task_display_states["gfxTotalInit"]
                    task_data_gfx_total['status'] = 'Starting'
                    task_data_gfx_total['start_time'] = time.time()
                    task_data_gfx_total['expected_time'] = PRESET_EXECUTION_TIMES.get("gfxTotalInit", INITIAL_PRESET_PLACEHOLDER_TIME)

                    if TH:
                        TH.initialize_graphics_and_external_libs(loaded_fonts, status_queue_for_main_thread, PRESET_EXECUTION_TIMES)
                    else:
                        task_data = task_display_states["retrieveMapData"]
                        task_data['status'] = 'Error'
                        task_data['duration'] = 0.0
                        TH_fully_initialized = True
                        print("Main Error: TH is None after future.result().")

                except Exception as e_future_result:
                    retrieval_duration = time.time() - actual_retrieval_start_time
                    all_current_run_times["retrieveMapData (Error)"] = retrieval_duration

                    task_data = task_display_states["retrieveMapData"]
                    task_data['status'] = 'Error'
                    task_data['duration'] = retrieval_duration
                    print(f"Main Error: Retrieving map data failed: {e_future_result}")
                    TH_fully_initialized = True
                    retrieving_result_active = False

            try:
                while not status_queue_for_main_thread.empty():
                    step_name_key_from_worker, status_type, time_value = status_queue_for_main_thread.get_nowait()

                    display_name_human_readable = DISPLAY_NAMES_MAP.get(step_name_key_from_worker, step_name_key_from_worker)

                    if step_name_key_from_worker not in task_display_states:
                        print(f"Main: Received unknown task status key: '{step_name_key_from_worker}' (mapped to '{display_name_human_readable}'). Please check config.")
                        continue

                    current_task_data = task_display_states[step_name_key_from_worker]

                    if status_type == "START":
                        current_task_data['status'] = 'Starting'
                        current_task_data['start_time'] = time.time()
                        current_task_data['expected_time'] = time_value
                    elif status_type == "SENT":
                        current_task_data['status'] = 'Sent'
                        current_task_data['start_time'] = time.time()
                        current_task_data['expected_time'] = time_value
                    elif status_type == "FINISHED":
                        current_task_data['status'] = 'Finished'
                        current_task_data['duration'] = time_value
                        if step_name_key_from_worker == "gfxTotalInit":
                            TH_fully_initialized = True
                    elif status_type == "ERROR":
                        current_task_data['status'] = 'Error'
                        current_task_data['duration'] = 0.0
                        print(f"Main (Error from queue): Task '{display_name_human_readable}' failed - Details: {time_value}")
                        TH_fully_initialized = True
            except (multiprocessing.queues.Empty, EOFError):
                pass
            except Exception as e_queue:
                print(f"Main: Error processing status queue: {e_queue}")
                TH_fully_initialized = True

            screen.blit(screen2_loading_background, (0, 0))
            screen.blit(overlay, (0, 0))

            drawText(screenUI, Cols.crimson, Alkhemikal200, main_title_x, screen_center[1] - 150, "Crimson", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
            drawText(screenUI, Cols.crimson, Alkhemikal200, main_title_x, screen_center[1] - 10, "Wakes", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)

            current_overall_phase = PHASE_WORKER_INIT
            if task_display_states["gfxTotalInit"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
                current_overall_phase = PHASE_GFX_INIT
            elif task_display_states["retrieveMapData"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
                current_overall_phase = PHASE_RETRIEVING_MAP_DATA
            elif task_display_states["dataSerialization"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
                current_overall_phase = PHASE_DATA_TRANSFER_PREP

            top_loading_text = current_overall_phase + ("." * int(numPeriods))
            if TH_fully_initialized and TH:
                top_loading_text = "Loading Complete!"
            elif TH_fully_initialized and not TH:
                top_loading_text = "Generation Error"

            drawText(screenUI, Cols.light, Alkhemikal50, main_overall_phase_x, screen_center[1] + 90, top_loading_text, Cols.dark, shadowSize=5, justify="center", centeredVertically=True)

            y_pos_offset = 0
            for task_name_key in LOADING_STEPS_ORDER:
                task_data = task_display_states[task_name_key]
                status = task_data['status']
                task_y_pos = single_task_display_start_y + y_pos_offset

                display_name_human_readable = DISPLAY_NAMES_MAP.get(task_name_key, task_name_key)
                if display_name_human_readable == "World Generation Complete (Worker)":
                    continue

                infoText = ""
                progress_ratio = 0.0
                show_progress_bar = False

                if status == 'Pending':
                    infoText = "Pending"
                elif status == 'Starting':
                    elapsed_time = time.time() - task_data['start_time']
                    expected = task_data['expected_time']
                    expected_str = f"{expected:.2f}s" if expected != INITIAL_PRESET_PLACEHOLDER_TIME else "Calc..."
                    infoText = f"{elapsed_time:.2f}s / {expected_str}"
                    progress_ratio = normalize(elapsed_time, 0, expected, clamp=True) if expected > 0 else 0.0
                    show_progress_bar = True
                elif status == 'Sent':
                    elapsed_time = time.time() - task_data['start_time']
                    expected = task_data['expected_time']
                    expected_str = f"{expected:.2f}s" if expected != INITIAL_PRESET_PLACEHOLDER_TIME else "Calc..."
                    infoText = f"SENT ({elapsed_time:.2f}s / {expected_str})"
                    progress_ratio = normalize(elapsed_time, 0, expected, clamp=True) if expected > 0 else 0.0
                    show_progress_bar = True
                elif status == 'Finished':
                    infoText = f"Done ({task_data['duration']:.2f}s)"
                    progress_ratio = 1.0
                    show_progress_bar = True
                elif status == 'Error':
                    infoText = "Error!"
                    progress_ratio = 0.0
                    show_progress_bar = False

                if Alkhemikal20:
                    drawText(screenUI, Cols.light, Alkhemikal20, tasks_list_x, task_y_pos, display_name_human_readable, Cols.dark, shadowSize=2, justify="left")
                    drawText(screenUI, Cols.light, Alkhemikal20, screen_width - 10, task_y_pos, infoText, Cols.dark, shadowSize=2, justify="right")

                if show_progress_bar:
                    bar_start_x = screen_width * 0.7
                    bar_y = task_y_pos + progress_bar_y_offset
                    outline_rect = pygame.Rect(bar_start_x, bar_y, progress_bar_width, progress_bar_height)
                    pygame.draw.rect(screenUI, Cols.dark, outline_rect, 2, border_radius=progress_bar_corner_radius)

                    fill_width = progress_bar_width * progress_ratio
                    if fill_width >= 1:
                        current_corner_radius = progress_bar_corner_radius
                        if fill_width < 2 * progress_bar_corner_radius:
                            current_corner_radius = int(fill_width / 2)
                        if current_corner_radius < 0:
                            current_corner_radius = 0

                        fill_rect = pygame.Rect(bar_start_x, bar_y, fill_width, progress_bar_height)
                        pygame.draw.rect(screenUI, Cols.crimson, fill_rect, 0, border_radius=current_corner_radius)

                y_pos_offset += line_height

            if toggle:
                string_fps = f"FPS: {round(clock.get_fps())}"
                drawText(screenUI, Cols.crimson, Alkhemikal30, 5, screen_height - 30, string_fps, Cols.dark, 3, antiAliasing=False)
                pygame.draw.circle(screenUI, Cols.dark, (mx + 2, my + 2), 7, 2)
                pygame.draw.circle(screenUI, Cols.light, (mx, my), 7, 2)
            screen.blit(screenUI, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
            continue

        if loading_screen_start_time != 0:
            total_loading_screen_time = time.time() - loading_screen_start_time
            print(f"Main: Loading screen displayed for: {total_loading_screen_time:.4f} seconds.")
            loading_screen_start_time = 0

        print("Main: Shutting down executor and manager.")
        executor.shutdown(wait=True)
        manager.shutdown()

        if all_current_run_times:
            save_execution_times(all_current_run_times)

        if TH is None:
            print("Error: TileHandler failed to initialize. Exiting.")
            pygame.quit()
            sys.exit()
        if TH.playersSurfScreen is None:
            print("CRITICAL MAIN ERROR: TH.playersSurfScreen is None post-init. Exiting.")
            pygame.quit()
            sys.exit()
        print("Main: TileHandler fully initialized. Starting game.")

        player = Player(target_host_ip, target_host_port, None, (screen_width, screen_height), {'30': Alkhemikal30, '50': Alkhemikal50, '80': Alkhemikal80, '150': Alkhemikal150, '200': Alkhemikal200}, Cols)

        break

    scrollSpeed = 50
    scroll = [0.0, 0.0]
    targetScroll = [0.0, 0.0]
    momentum = [0.0, 0.0]
    moving = [0.0, 0.0]

    from controlPanel import uiInfo

    bottomUIBarSize = uiInfo.bottomUIBarSize * screen_height

    max_scroll_x = 0
    min_scroll_x = -(TH.mapWidth - screen_width)
    max_scroll_y = 0
    min_scroll_y = -(TH.mapHeight - screen_height)

    debug = False
    mouseSize = 1
    click = False
    showClouds = True
    pygame.mouse.set_visible(False)

    game_screen_surface = pygame.Surface((screen_width, screen_height)).convert_alpha()

    while running:
        mx, my = pygame.mouse.get_pos()
        game_screen_surface.fill(Cols.dark)
        screenUI.fill((0, 0, 0, 0))
        dt = (time.time() - last_time) * fps
        last_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                click = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    toggle = not toggle
                if event.key == pygame.K_a:
                    moving[0] += 1
                if event.key == pygame.K_d:
                    moving[0] -= 1
                if event.key == pygame.K_w:
                    moving[1] += 1
                if event.key == pygame.K_s:
                    moving[1] -= 1

                if event.key == pygame.K_x:
                    debug = not debug
                if event.key == pygame.K_m:
                    mouseSize = (mouseSize + 1) % 4
                if event.key == pygame.K_c:
                    showClouds = not showClouds
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    moving[0] -= 1
                if event.key == pygame.K_d:
                    moving[0] += 1
                if event.key == pygame.K_w:
                    moving[1] -= 1
                if event.key == pygame.K_s:
                    moving[1] += 1

        targetScroll[0] += scrollSpeed * moving[0]
        targetScroll[1] += scrollSpeed * moving[1]
        targetScroll[0] = min(max(targetScroll[0], min_scroll_x), max_scroll_x)
        targetScroll[1] = min(max(targetScroll[1], min_scroll_y), max_scroll_y)

        diffs = [targetScroll[0] - scroll[0], targetScroll[1] - scroll[1]]
        for idx, diff in enumerate(diffs):
            momentum[idx] += diff / 25
            momentum[idx] *= 0.7
            scroll[idx] += momentum[idx]
            scroll[idx] = min(max(scroll[idx], [min_scroll_x, min_scroll_y][idx]), [max_scroll_x, max_scroll_y][idx])
        adjustedMx, adjustedMy = [mx - scroll[0], my - scroll[1]]

        tile_under_mouse = TH.getTileAtPosition(adjustedMx, adjustedMy)
        hovered_territory = None
        if tile_under_mouse and tile_under_mouse.territory_id != -1:
            potential_hovered_terr = TH.territories_by_id.get(tile_under_mouse.territory_id)
            if potential_hovered_terr and hasattr(potential_hovered_terr, 'polygon') and potential_hovered_terr.polygon:
                from shapely.geometry import Point

                if Point(adjustedMx, adjustedMy).intersects(potential_hovered_terr.polygon):
                    hovered_territory = potential_hovered_terr

        player.handleClick(click, dt, hovered_territory)
        player.update(dt)

        visible_map_rect_on_full_map = pygame.Rect(-scroll[0], -scroll[1], screen_width, screen_height)

        if TH.baseMapSurf:
            game_screen_surface.blit(TH.baseMapSurf.subsurface(visible_map_rect_on_full_map), (0, 0))

        if debug and TH.debugOverlayFullMap:
            game_screen_surface.blit(TH.debugOverlayFullMap.subsurface(visible_map_rect_on_full_map), (0, 0))

        TH.drawTerritoryHighlights(game_screen_surface, hovered_territory, player.selectedTerritory, scroll)

        TH.playersSurfScreen.fill((0, 0, 0, 0))
        if TH.playersSurfScreen:
            player.draw(TH.playersSurfScreen, screenUI, False, scroll)
            game_screen_surface.blit(TH.playersSurfScreen, (0, 0))
        if showClouds and TH.cloudSurfFullMap:
            TH.drawClouds(game_screen_surface, adjustedMx, adjustedMy, mouseSize, player, scroll, [screen_width, screen_height], player.visibleTerritoryIDs)

        pygame.draw.line(screenUI, Cols.debugRed, (0, screen_height - bottomUIBarSize), (screen_width, screen_height - bottomUIBarSize), 2)

        if toggle:
            fps_text = f"{clock.get_fps():.1f}"
            if Alkhemikal30:
                sel_terr_text = "No Territory"
                if player.selectedTerritory and hasattr(player.selectedTerritory, 'id'):
                    sel_terr_text = f"Territory ID: {player.selectedTerritory.id}"
                drawText(screenUI, Cols.light, Alkhemikal30, screen_width / 2, 30, f"your name is {username}", Cols.dark, 3, justify="middle", centeredVertically=True)
                drawText(screenUI, Cols.debugRed, Alkhemikal30, 5, screen_height - 90, sel_terr_text, Cols.dark, 3, antiAliasing=False)
                drawText(screenUI, Cols.debugRed, Alkhemikal30, 5, screen_height - 60, fps_text, Cols.dark, 3, antiAliasing=False)
                drawText(screenUI, Cols.debugRed, Alkhemikal30, 5, screen_height - 30, "[spc] UI, [x] Debug, [m] Mouse Size, [c] Clouds", Cols.dark, 3, antiAliasing=False)
            pygame.draw.circle(screenUI, Cols.dark, (mx + 2, my + 2), 7, 2)
            pygame.draw.circle(screenUI, Cols.light, (mx, my), 7, 2)

        screen.blit(game_screen_surface, (0, 0))
        screen.blit(screenUI, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()
