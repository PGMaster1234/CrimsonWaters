import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pygame
import time
import sys
import os
import csv
import statistics

from text import drawText
from fontDict import fonts as fonts_definitions
from controlPanel import GenerationInfo, ResourceInfo, StructureInfo, Cols
from player import Player
from calcs import normalize

TIMES_CSV_FILE = "execution_times.csv"
INITIAL_PRESET_PLACEHOLDER_TIME = 999.0


def load_and_calculate_average_times():
    global PRESET_EXECUTION_TIMES
    new_preset_times = {}
    all_step_durations = {}

    if os.path.exists(TIMES_CSV_FILE):
        try:
            with open(TIMES_CSV_FILE, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                if not reader.fieldnames or 'step_name' not in reader.fieldnames or 'duration' not in reader.fieldnames:
                    pass
                else:
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


PRESET_EXECUTION_TIMES = {}
load_and_calculate_average_times()


def build_tile_handler_worker(args_tuple):
    width, height, gen_info, font_name_to_load, font_definitions_dict, cols_class, resource_info_class, structure_info_class, local_status_q, current_preset_times = args_tuple

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
    TH_instance = TileHandler(width, height, gen_info.tileSize, cols_class, gen_info.waterThreshold, gen_info.mountainThreshold, gen_info.territorySize, font=_font, font_name=font_name_to_load, resource_info=resource_info_class, structure_info=structure_info_class, status_queue=local_status_q, preset_times=current_preset_times)
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
    screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
    screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()

    loaded_fonts = {}
    for name, (path, size) in fonts_definitions.items():
        try:
            if not os.path.exists(path):
                continue
            loaded_fonts[name] = pygame.font.Font(path, size)
        except Exception as e_font_load:
            print(f"Main Error: Failed to load font {name} from {path} (size {size}): {e_font_load}")

    # Ensure Alkhemikal20 is loaded if used
    Alkhemikal20 = loaded_fonts.get('Alkhemikal20') 
    Alkhemikal30 = loaded_fonts.get('Alkhemikal30')
    Alkhemikal50 = loaded_fonts.get('Alkhemikal50')
    Alkhemikal80 = loaded_fonts.get('Alkhemikal80')
    Alkhemikal150 = loaded_fonts.get('Alkhemikal150')
    Alkhemikal200 = loaded_fonts.get('Alkhemikal200')

    try:
        generationScreenBackgroundImg = pygame.transform.scale(pygame.image.load("assets/UI/LoadingPageBackground.png"), (screen_width, screen_height))
    except pygame.error as e_img_load:
        print(f"Main Error: Loading background image failed: {e_img_load}")
        generationScreenBackgroundImg = pygame.Surface((screen_width, screen_height))
        generationScreenBackgroundImg.fill(Cols.dark)

    manager = multiprocessing.Manager()
    status_queue_for_main_thread = manager.Queue()

    executor = ProcessPoolExecutor(max_workers=1)
    font_name_needed_by_worker = 'Alkhemikal30'
    print(f"Main: Submitting TileHandler generation task to worker.")

    worker_args = (screen_width, screen_height, GenerationInfo, font_name_needed_by_worker, fonts_definitions, Cols, ResourceInfo, StructureInfo, status_queue_for_main_thread, PRESET_EXECUTION_TIMES)
    future = executor.submit(build_tile_handler_worker, worker_args)

    numPeriods = 0
    PHASE_WORKER_INIT = "Initializing World Generation"
    PHASE_DATA_TRANSFER_PREP = "Preparing Data for Transfer"
    PHASE_RETRIEVING_MAP_DATA = "Retrieving World Data"
    PHASE_GFX_INIT = "Initializing Graphics"

    # New, concise camelCase internal names for steps
    LOADING_STEPS_ORDER = ["tileGen", "linkAdj", "cloudPrecompParallel", "generationCycles", "setTileColors", "findLandRegionsParallel", "indexOceansParallel", "assignCoastTiles", "createTerritories", "connectHarborsParallel", "workerInit", "dataSerialization", "retrieveMapData", "gfxSurfaceSetup", "gfxFontSetup", "gfxRebuildMaps", "gfxRestoreTerrLinks", "gfxRestoreAdjLinks", "gfxTerrHarborInit", "gfxUpdateReachableHarbors", "gfxDrawInternal", "gfxCloudPrecompConditional", "gfxInitCloudSurf", "gfxTotalInit"]

    # Map for displaying human-readable names on screen
    DISPLAY_NAMES_MAP = {
        "tileGen": "Tile Generation",
        "linkAdj": "Linking Adjacent Objects",
        "cloudPrecompParallel": "Cloud Precomputation (Parallel)",
        "generationCycles": "50 Generation Cycles",
        "setTileColors": "Setting Tile Colors",
        "findLandRegionsParallel": "Finding Land Regions (Parallel)",
        "indexOceansParallel": "Indexing Oceans (Parallel)",
        "assignCoastTiles": "Assigning Coast Tiles",
        "createTerritories": "Creating Territories",
        "connectHarborsParallel": "Connecting Harbors (Parallel)",
        "workerInit": "Worker Initialization",
        "dataSerialization": "Data Serialization for Transfer",
        "retrieveMapData": "Retrieving World Data",
        "gfxSurfaceSetup": "Graphics: Surface Setup",
        "gfxFontSetup": "Graphics: Font Setup",
        "gfxRebuildMaps": "Graphics: Rebuild Maps",
        "gfxRestoreTerrLinks": "Graphics: Restore Territory Links",
        "gfxRestoreAdjLinks": "Graphics: Restore Adjacent Links",
        "gfxTerrHarborInit": "Graphics: Territory & Harbor Init",
        "gfxUpdateReachableHarbors": "Graphics: Update Reachable Harbors",
        "gfxDrawInternal": "Graphics: Draw Internal Screen",
        "gfxCloudPrecompConditional": "Graphics: Cloud Precomp (Cond)",
        "gfxInitCloudSurf": "Graphics: Initialize Cloud Surface",
        "gfxTotalInit": "Graphics: Total Initialization"
    }

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
    progress_bar_y_offset = 5
    progress_bar_corner_radius = int(progress_bar_height / 3)

    single_task_display_start_y = screen_center[1] - (len(LOADING_STEPS_ORDER) * line_height / 2)

    mx, my = pygame.mouse.get_pos()
    toggle = True

    while not TH_fully_initialized:
        numPeriods = (numPeriods + 3 / fps) % 4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Main: QUIT event received. Shutting down.")
                if not future.done():
                    future.cancel()
                executor.shutdown(wait=False)
                manager.shutdown()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("Main: ESCAPE key pressed. Shutting down.")
                if not future.done():
                    future.cancel()
                executor.shutdown(wait=False)
                manager.shutdown()
                pygame.quit()
                sys.exit()

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

                if TH:
                    task_data_gfx_total = task_display_states["gfxTotalInit"]
                    task_data_gfx_total['status'] = 'Starting'
                    task_data_gfx_total['start_time'] = time.time()
                    task_data_gfx_total['expected_time'] = PRESET_EXECUTION_TIMES.get("gfxTotalInit", INITIAL_PRESET_PLACEHOLDER_TIME)

                    TH.initialize_graphics_and_external_libs(loaded_fonts, status_queue_for_main_thread, PRESET_EXECUTION_TIMES)

                    gfx_total_duration = time.time() - task_data_gfx_total['start_time']
                    task_data_gfx_total['status'] = 'Finished'
                    task_data_gfx_total['duration'] = gfx_total_duration

                    if hasattr(TH, 'execution_times'):
                        all_current_run_times.update(TH.execution_times)
                else:
                    task_data = task_display_states["retrieveMapData"]
                    task_data['status'] = 'Error'
                    task_data['duration'] = 0.0
                    TH_fully_initialized = True

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

        screen.blit(generationScreenBackgroundImg, (0, 0))
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 10, 10, 200))
        screen.blit(overlay, (0, 0))

        mx, my = pygame.mouse.get_pos()

        if Alkhemikal150:
            drawText(screen, Cols.brightCrimson, Alkhemikal200, main_title_x, screen_center[1] - 150, "Crimson", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
            drawText(screen, Cols.brightCrimson, Alkhemikal200, main_title_x, screen_center[1] - 10, "Wakes", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)

        current_overall_phase = PHASE_WORKER_INIT
        if task_display_states["retrieveMapData"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
            current_overall_phase = PHASE_RETRIEVING_MAP_DATA
        elif task_display_states["dataSerialization"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
            current_overall_phase = PHASE_DATA_TRANSFER_PREP
        elif task_display_states["gfxTotalInit"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
            current_overall_phase = PHASE_GFX_INIT
        elif task_display_states["workerInit"]['status'] in ['Starting', 'Sent', 'Finished', 'Error']:
            current_overall_phase = PHASE_WORKER_INIT

        top_loading_text = current_overall_phase + ("." * int(numPeriods))
        if TH_fully_initialized and TH:
            top_loading_text = "Loading Complete!"
        elif TH_fully_initialized and not TH:
            top_loading_text = "Generation Error"

        if Alkhemikal50:
            drawText(screen, Cols.light, Alkhemikal50, main_overall_phase_x, screen_center[1] + 90, top_loading_text, Cols.dark, shadowSize=5, justify="center", centeredVertically=True)

        y_pos_offset = 0
        for task_name_key in LOADING_STEPS_ORDER:
            task_data = task_display_states[task_name_key]
            status = task_data['status']
            task_y_pos = single_task_display_start_y + y_pos_offset

            display_name_human_readable = DISPLAY_NAMES_MAP.get(task_name_key, task_name_key)

            infoText = ""
            progress_ratio = 0.0
            show_progress_bar = False

            if status == 'Pending':
                infoText = "Pending"
            elif status == 'Starting':
                elapsed_time = time.time() - task_data['start_time']
                expected = task_data['expected_time']
                expected_str = f"{expected:.2f}s" if expected != INITIAL_PRESET_PLACEHOLDER_TIME else "Calculating..."
                infoText = f"{elapsed_time:.2f}s / {expected_str}"
                progress_ratio = normalize(elapsed_time, 0, expected, clamp=True) if expected > 0 else 0.0
                show_progress_bar = True
            elif status == 'Sent':
                elapsed_time = time.time() - task_data['start_time']
                expected = task_data['expected_time']
                expected_str = f"{expected:.2f}s" if expected != INITIAL_PRESET_PLACEHOLDER_TIME else "Calculating..."
                infoText = f"SENT ({elapsed_time:.2f}s / {expected_str})"
                progress_ratio = normalize(elapsed_time, 0, expected, clamp=True) if expected > 0 else 0.0
                show_progress_bar = True
            elif status == 'Finished':
                infoText = f"Done ({task_data['duration']:.2f}s)"
                progress_ratio = 1.0
                show_progress_bar = True
            elif status == 'Error':
                infoText = f"Error!"
                progress_ratio = 0.0
                show_progress_bar = False

            if Alkhemikal20:
                drawText(screen, Cols.light, Alkhemikal20, tasks_list_x, task_y_pos, display_name_human_readable, Cols.dark, shadowSize=2, justify="left")
                drawText(screen, Cols.light, Alkhemikal20, screen_width - 10, task_y_pos, infoText, Cols.dark, shadowSize=2, justify="right")

            if show_progress_bar:
                bar_start_x = screen_width * 0.7
                bar_y = task_y_pos + progress_bar_y_offset
                outline_rect = pygame.Rect(bar_start_x, bar_y, progress_bar_width, progress_bar_height)
                pygame.draw.rect(screen, Cols.dark, outline_rect, 2, border_radius=progress_bar_corner_radius)

                fill_width = progress_bar_width * progress_ratio
                if fill_width >= 1:
                    current_corner_radius = progress_bar_corner_radius
                    if fill_width < 2 * progress_bar_corner_radius:
                        current_corner_radius = int(fill_width / 2)
                    if current_corner_radius < 0:
                        current_corner_radius = 0

                    fill_rect = pygame.Rect(bar_start_x, bar_y, fill_width, progress_bar_height)
                    pygame.draw.rect(screen, Cols.crimson, fill_rect, 0, border_radius=current_corner_radius)

            y_pos_offset += line_height

        if toggle:
            pygame.mouse.set_visible(False)
            pygame.draw.circle(screen, Cols.dark, (mx + 2, my + 2), 7, 2)
            pygame.draw.circle(screen, Cols.light, (mx, my), 7, 2)
        else:
            pygame.mouse.set_visible(True)

        pygame.display.flip()
        clock.tick(fps)

    total_loading_screen_time = time.time() - loading_screen_start_time
    print(f"Main: Loading screen displayed for: {total_loading_screen_time:.4f} seconds.")

    print("Main: Shutting down executor and manager.")
    executor.shutdown(wait=True)
    manager.shutdown()

    if all_current_run_times:
        save_execution_times(all_current_run_times)

    if TH is None:
        print("Error: TileHandler failed to initialize. Exiting.")
        pygame.quit()
        sys.exit()
    if TH.playersSurf is None:
        print("CRITICAL MAIN ERROR: TH.playersSurf is None post-init. Exiting.")
        pygame.quit()
        sys.exit()
    print("Main: TileHandler fully initialized. Starting game.")

    player = Player(None, None, None, (TH.screenWidth, TH.screenHeight), {'30': Alkhemikal30, '50': Alkhemikal50, '80': Alkhemikal80, '150': Alkhemikal150, '200': Alkhemikal200}, Cols)
    debug = False
    toggle = True
    mouseSize = 1
    click = False
    showClouds = True
    last_time = time.time()
    running = True
    pygame.mouse.set_visible(False)
    while running:
        mx, my = pygame.mouse.get_pos()
        screen.fill(Cols.oceanBlue)
        screen2.fill((0, 0, 0, 0))
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
                if event.key == pygame.K_d:
                    debug = not debug
                if event.key == pygame.K_m:
                    mouseSize = (mouseSize + 1) % 4
                if event.key == pygame.K_c:
                    showClouds = not showClouds

        # Calculate hovered_territory based on getTileAtPosition for efficient drawing
        tile_under_mouse = TH.getTileAtPosition(mx, my)
        hovered_territory = None
        if tile_under_mouse and tile_under_mouse.territory_id != -1:
            potential_hovered_terr = TH.territories_by_id.get(tile_under_mouse.territory_id)
            if potential_hovered_terr and hasattr(potential_hovered_terr, 'polygon') and potential_hovered_terr.polygon:
                from shapely.geometry import Point 

                if Point(mx, my).intersects(potential_hovered_terr.polygon):
                    hovered_territory = potential_hovered_terr

        TH.draw(screen2, showArrows=False, showDebugOverlay=debug, showWaterLand=False, hovered_territory=hovered_territory, selected_territory=player.selectedTerritory)

        player.handleClick(click, dt, hovered_territory)
        player.update(dt)
        if TH.playersSurf:
            player.draw(TH.playersSurf, screenUI, False)
            screen2.blit(TH.playersSurf, (0, 0))
        if showClouds:
            TH.drawClouds(screen2, mx, my, mouseSize, player)

        if toggle:
            fps_text = f"{clock.get_fps():.1f}"
            if Alkhemikal30:
                sel_terr_text = "No Territory"
                if player.selectedTerritory and hasattr(player.selectedTerritory, 'id'):
                    sel_terr_text = f"Territory ID: {player.selectedTerritory.id}"
                drawText(screen2, Cols.debugRed, Alkhemikal30, 5, screen_height - 90, sel_terr_text, Cols.dark, 3, antiAliasing=False)
                drawText(screen2, Cols.debugRed, Alkhemikal30, 5, screen_height - 60, fps_text, Cols.dark, 3, antiAliasing=False)
                drawText(screen2, Cols.debugRed, Alkhemikal30, 5, screen_height - 30, "[spc] UI, [d] Debug, [m] Mouse Size, [c] Clouds", Cols.dark, 3, antiAliasing=False)
            pygame.draw.circle(screenUI, Cols.dark, (mx + 2, my + 2), 7, 2)
            pygame.draw.circle(screenUI, Cols.light, (mx, my), 7, 2)

        screen.blit(screen2, (0, 0))
        screen.blit(screenUI, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()
