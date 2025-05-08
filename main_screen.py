import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pygame
import time
import sys
import os
from text import drawText
from fontDict import fonts as fonts_definitions
from controlPanel import GenerationInfo, ResourceInfo, StructureInfo


class Cols:
    oceanBlue = [59, 95, 111]
    oceanGreen = [73, 120, 122]
    lightOceanGreen = [86, 142, 143]
    oceanFoam = [148, 182, 180]
    sandyBrown = [172, 146, 95]
    darkSandyBrown = [159, 143, 91]
    oliveGreen = [95, 115, 84]
    darkOliveGreen = [54, 64, 57]
    mountainBlue = (83, 78, 90)
    darkMountainBlue = [42, 40, 52]
    light = [220, 216, 201]
    dark = [18, 22, 27]
    accentOrange = [155, 110, 83]
    crimson = [94, 32, 32]
    debugRed = [255, 96, 141]


def build_tile_handler_worker(args):
    width, height, gen_info, font_name_to_load, font_definitions_dict, cols_class, resource_info_class, structure_info_class = args
    from generation import TileHandler
    print(f"Worker: Received request for font: {font_name_to_load}")
    _font = None
    if font_name_to_load and font_name_to_load in font_definitions_dict:
        pygame.init()
        pygame.font.init()
        font_path, font_size = font_definitions_dict[font_name_to_load]
        _font = pygame.font.Font(font_path, font_size)
    TH_instance = TileHandler(width, height, gen_info.tileSize, cols_class, gen_info.waterThreshold, gen_info.mountainThreshold, gen_info.territorySize, font=_font, font_name=font_name_to_load, resource_info=resource_info_class, structure_info=structure_info_class)
    TH_instance.prepare_for_pickling()
    return TH_instance


# --- Entry Point ---
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    multiprocessing.freeze_support()
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    fps = 60
    screen_width, screen_height = screen.get_width(), screen.get_height()
    screen_center = [screen_width / 2, screen_height / 2]
    screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()

    # Load fonts (Keep as is)
    loaded_fonts = {}
    print(f"Main: Loading fonts based on {len(fonts_definitions)} definitions...")
    for name, (path, size) in fonts_definitions.items():
        try:
            if not os.path.exists(path):
                print(f"Main Warning: Font path not found for '{name}': {path}")
                continue
            loaded_fonts[name] = pygame.font.Font(path, size)
        except Exception as e:
            print(f"Main Error: Failed to load font {name} from {path} (size {size}): {e}")
    print(f"Main: Successfully loaded {len(loaded_fonts)} fonts.")
    Alkhemikal30 = loaded_fonts.get('Alkhemikal30')
    Alkhemikal50 = loaded_fonts.get('Alkhemikal50')
    Alkhemikal80 = loaded_fonts.get('Alkhemikal80')
    Alkhemikal200 = loaded_fonts.get('Alkhemikal200')

    try:
        generationScreenBackgroundImg = pygame.transform.scale(pygame.image.load("assets/UI/LoadingPageBackground.png"), (screen_width, screen_height))
    except pygame.error as e:
        print(f"Error loading background image: {e}")
        generationScreenBackgroundImg = pygame.Surface((screen_width, screen_height))

    executor = ProcessPoolExecutor(max_workers=1)
    font_name_needed_by_worker = 'Alkhemikal30'
    print(f"Main: Submitting task, requesting worker load font: {font_name_needed_by_worker}")
    future = executor.submit(build_tile_handler_worker, (screen_width, screen_height, GenerationInfo, font_name_needed_by_worker, fonts_definitions, Cols, ResourceInfo, StructureInfo))

    while not future.done():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                executor.shutdown(wait=False, cancel_futures=True)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                executor.shutdown(wait=False, cancel_futures=True)
                pygame.quit()
                sys.exit()
        screen.blit(generationScreenBackgroundImg, (0, 0))
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 10, 10, 200))
        screen.blit(overlay, (0, 0))
        drawText(screen, Cols.light, Alkhemikal80, screen_center[0], screen_center[1] + 50, "Generating map...", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
        drawText(screen, Cols.crimson, Alkhemikal200, screen_center[0], screen_center[1] - 50, "Crimson Wakes", Cols.dark, shadowSize=5, justify="center", centeredVertically=True)
        pygame.display.flip()
        clock.tick(fps)

    TH = None
    try:
        print("Main: Waiting for generation result...")
        TH = future.result()
        print("Main: Received TileHandler instance.")
        print("Main: Initializing graphics and external libs for TileHandler...")
        TH.initialize_graphics_and_external_libs(loaded_fonts)
        print("Main: TileHandler graphics ready.")
    except Exception as e:
        print(f"Main Error: An error occurred during generation or restoration: {e}")
        import traceback

        traceback.print_exc()
    finally:
        executor.shutdown()
        if TH is None:
            print("Error: TileHandler failed to initialize. Exiting.")
            pygame.quit()
            sys.exit()

    # Game loop variables (Keep as is)
    oscillating_random_thing = 0
    ShakeCounter = 0
    toggle = True
    click = False

    # --- Main Game Loop --- (Keep as is)
    last_time = time.time()
    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        screen.fill(Cols.oceanBlue)
        screen2.fill((0, 0, 0, 0))
        dt = time.time() - last_time
        last_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN: click = True
            if event.type == pygame.MOUSEBUTTONUP: click = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_SPACE: toggle = not toggle

        TH.draw(screen2, mx, my, click, showArrows=False, showDebugOverlay=False, showWaterLand=False)

        if toggle:
            fps_val = clock.get_fps()
            fps_text = f"{fps_val:.1f}"
            drawText(screen2, Cols.debugRed, Alkhemikal30, 5, screen_height - 30, fps_text, Cols.dark, 3, antiAliasing=False)
            pygame.mouse.set_visible(False)
            pygame.draw.circle(screen2, Cols.dark, (mx + 2, my + 2), 7, 2)
            pygame.draw.circle(screen2, Cols.light, (mx, my), 7, 2)

        screen.blit(screen2, (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()
