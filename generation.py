import pygame
import math
import random
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from text import drawText
from calcs import distance, ang, normalize_angle, draw_arrow, linearGradient, normalize
from territory import Territory
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from controlPanel import GenerationInfo
import os
import pickle
import gzip

CLOUD_PRECOMP_DIR = "cloudPrecomp"


class Hex:
    def __init__(self, grid_x, grid_y, x, y, size, tile_id, col=(0, 0, 0), cloudCol=(50, 50, 50)):
        self.tile_id = tile_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = x
        self.y = y
        self.size = size
        self.col = col
        self.cloudCol = cloudCol
        self.center = [self.x, self.y]
        self.floatHexVertices = [(self.x + self.size * math.cos(math.pi / 3 * angle), self.y + self.size * math.sin(math.pi / 3 * angle)) for angle in range(6)]
        self.hex = [(int(round(p[0])), int(round(p[1]))) for p in self.floatHexVertices]
        self.adjacent_tile_ids = []
        self.territory_id = -1
        self.adjacent = []
        self.territory = None
        self.region = None
        self.mountainRegion = None
        self.regionCol = None
        self.waterLand = random.random()
        self.mountainous = random.random()
        self.cloudy = random.random()
        self.isLand = False
        self.isMountain = False
        self.isCoast = False
        self.connectedOceanID = -1
        self.cloudOpacity = 1.0
        self.precomp_chunk_coords = None

    def prepare_for_pickling(self):
        self.adjacent_tile_ids = []
        for adj in self.adjacent:
            if hasattr(adj, 'tile_id'):
                self.adjacent_tile_ids.append(adj.tile_id)
        self.adjacent = []
        self.territory = None
        self.region = None
        self.mountainRegion = None

    @property
    def bounding_rect(self):
        min_x = min(p[0] for p in self.hex)
        max_x = max(p[0] for p in self.hex)
        min_y = min(p[1] for p in self.hex)
        max_y = max(p[1] for p in self.hex)
        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def draw(self, s):
        pygame.draw.polygon(s, self.col, self.hex)

    def drawCloud(self, s, scroll_x, scroll_y):
        alpha = int(self.cloudOpacity * 255)
        color_rgb = self.cloudCol[:3] if isinstance(self.cloudCol, (list, tuple)) and len(self.cloudCol) >= 3 else (50, 50, 50)
        final_color_rgba = tuple(color_rgb) + (alpha,)
        shifted_hex = [(p[0] + scroll_x, p[1] + scroll_y) for p in self.hex]
        pygame.draw.polygon(s, final_color_rgba, shifted_hex)

    def drawArrows(self, s, color, scroll_x, scroll_y):
        if not self.adjacent:
            return
        arrow_col = tuple(color) if color else (255, 0, 0)
        for adj in self.adjacent:
            angle = normalize_angle(ang((self.x, self.y), (adj.x, adj.y)))
            dist_val = distance((self.x, self.y), (adj.x, adj.y))
            factor = 0.35
            start_pos = (self.x + scroll_x, self.y + scroll_y)
            end_pos = (self.x + dist_val * factor * math.cos(angle) + scroll_x, self.y + dist_val * factor * math.sin(angle) + scroll_y)
            draw_arrow(s, start_pos, end_pos, arrow_col, pygame, 2, 5, 25)

    def showWaterLand(self, s, font, color, scroll_x, scroll_y):
        if font:
            text_col = tuple(color) if color else (0, 0, 0)
            drawText(s, text_col, font, self.x + scroll_x, self.y + scroll_y, f"{self.waterLand:.2f}", justify="center", centeredVertically=True)


class TileHandler:
    def __init__(self, map_width, map_height, size, cols, waterThreshold=0.51, mountainThreshold=0.51, territorySize=100, font=None, font_name=None, resource_info=None, structure_info=None, status_queue: multiprocessing.Queue = None, preset_times: dict = None, seed: int = None, viewport_width=0, viewport_height=0):

        self.execution_times = {}
        self.status_queue = status_queue
        self.preset_times = preset_times if preset_times else {}

        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            print(f"WORKER STDOUT: Initialized TileHandler with provided seed: {self.seed}")
        else:
            self.seed = random.randint(0, 2 ** 32 - 1)
            random.seed(self.seed)
            np.random.seed(self.seed)
            print(f"WORKER STDOUT: Initialized TileHandler with new random seed: {self.seed}")

        self.mapWidth = map_width
        self.mapHeight = map_height
        self.viewportWidth = viewport_width
        self.viewportHeight = viewport_height
        self.cols = cols
        self.font = font
        self.font_name = font_name
        self.resource_info = resource_info
        self.structure_info = structure_info
        self.size = size
        self.territorySize = territorySize
        self.tiles = []
        self.tiles_by_id = {}
        self.tiles_by_grid_coords = {}
        self.territories_by_id = {}
        self.harbors_by_id = {}
        self.contiguousTerritoryIDs = []
        self.all_territories_for_unpickling = []
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        self.waterThreshold = waterThreshold
        self.mountainThreshold = mountainThreshold
        self.borderSize = 0
        self.horizontal_distance = (3 / 2 * size)
        self.vertical_distance = (math.sqrt(3) * size)
        self.gridSizeX = int(self.mapWidth / self.horizontal_distance) - self.borderSize + 1
        self.gridSizeY = int(self.mapHeight / self.vertical_distance) - self.borderSize + 2
        self.allWaterTiles = []
        self.allLandTiles = []
        self.allCoastalTiles = []

        self.allHarbors = None

        self.baseMapSurf = None
        self.debugOverlayFullMap = None
        self.cloudSurfFullMap = None

        self.territoryHighlightSurfScreen = None
        self.playersSurfScreen = None

        self.cloud_surf_initialized = False
        self.active_transparent_tiles = set()

        print(self.size)
        self.PRECOMP_CHUNK_SIZE = 6 * self.size
        self.precomp_radii = sorted([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600])
        self.precomp_chunk_grid_tiles = {}
        self.precomp_grid_width = math.ceil(self.mapWidth / self.PRECOMP_CHUNK_SIZE)
        self.precomp_grid_height = math.ceil(self.mapHeight / self.PRECOMP_CHUNK_SIZE)
        self._temp_contiguous_territories_objs = None
        self.territory_vision_cache = {}
        self._relative_hex_offsets_pattern = []
        self.precomputed_cloud_info = {}

        self.STEP_NAMES = {"TILE_GEN": "tileGen", "CLOUD_PRECOMP": "cloudPrecompParallel", "LINK_ADJ": "linkAdj", "GEN_CYCLES": "generationCycles", "SET_COLORS": "setTileColors", "FIND_REGIONS": "findLandRegionsParallel", "INDEX_OCEANS": "indexOceansParallel", "ASSIGN_COAST": "assignCoastTiles", "CREATE_TERR": "createTerritories", "CONNECT_HARBORS": "connectHarborsParallel", "PRECOMP_TERR_VISION": "precomputeTerritoryVision", "TOTAL_INIT": "workerInit", "PREP_PICKLING": "dataSerialization",
                           "GFX_TOTAL_INIT": "gfxTotalInit"}

        total_init_start_time_timer = time.time()
        landRegionsRaw_result = None

        # Helper function for sequential steps (main thread of worker process)
        def _run_step_sequential(step_key, func_to_run, *args):
            step_full_name = self.STEP_NAMES[step_key]
            expected_time = self.preset_times.get(step_full_name, 999.0)
            if self.status_queue:
                self.status_queue.put_nowait((step_full_name, "START", expected_time))

            s_time = time.time()
            result = func_to_run(*args)
            e_time = time.time()
            self.execution_times[step_full_name] = e_time - s_time
            if self.status_queue:
                self.status_queue.put_nowait((step_full_name, "FINISHED", e_time - s_time))
            return result

        # Helper function for tasks submitted to ThreadPoolExecutor
        def _threaded_task_wrapper(step_key_for_timing, func_to_run, *args):
            step_full_name = self.STEP_NAMES[step_key_for_timing]
            expected_time = self.preset_times.get(step_full_name, 999.0)

            if self.status_queue:
                self.status_queue.put_nowait((step_full_name, "START", expected_time))

            s_time = time.time()
            result = func_to_run(*args)
            e_time = time.time()
            duration = e_time - s_time
            self.execution_times[step_full_name] = duration
            if self.status_queue:
                self.status_queue.put_nowait((step_full_name, "FINISHED", duration))
            return result

        # Initialize the ThreadPoolExecutor for within-worker parallelization
        internal_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

        # --- Execution Flow ---

        # 1. Core Map Generation (Sequential)
        _run_step_sequential("TILE_GEN", self.generateTiles)
        _run_step_sequential("LINK_ADJ", self._link_adjacent_objects)

        # 2. Start independent parallel tasks early
        cloud_precomp_future = internal_executor.submit(_threaded_task_wrapper, "CLOUD_PRECOMP", self._precompute_cloud_visibility)

        # 3. Iterative Generation (Sequential)
        _run_step_sequential("GEN_CYCLES", lambda: [self.generationCycle() for _ in range(50)])

        # 4. Coloring Map Tiles (Parallel - P1)
        set_colors_future = internal_executor.submit(_threaded_task_wrapper, "SET_COLORS", self.setTileCols)
        # Wait for coloring to complete before starting dependent tasks
        set_colors_future.result()

        # 5. Identifying Landmasses and Indexing Oceans (Parallel - P2 & P3)
        find_regions_future = internal_executor.submit(_threaded_task_wrapper, "FIND_REGIONS", self.findContiguousRegions, [t for t in self.tiles if t.waterLand >= self.waterThreshold])
        index_oceans_future = internal_executor.submit(_threaded_task_wrapper, "INDEX_OCEANS", self.indexOceans)

        # Wait for both region finding and ocean indexing to complete
        landRegionsRaw_result = find_regions_future.result()
        index_oceans_future.result()

        # 6. Assigning Coastal Tiles (Parallel - P4)
        assign_coast_future = internal_executor.submit(_threaded_task_wrapper, "ASSIGN_COAST", self.assignCoastTiles)
        # Wait for coastal assignment to complete
        assign_coast_future.result()

        # 7. Create Territories (Sequential - depends on find_regions_future result)
        _run_step_sequential("CREATE_TERR", lambda: self.createTerritories(landRegionsRaw_result))

        # 8. Connecting Harbors and Precomputing Territory Vision (Parallel - P5 & P6)
        connect_harbors_future = internal_executor.submit(_threaded_task_wrapper, "CONNECT_HARBORS", self.connectTerritoryHarbors)
        precompute_vision_future = internal_executor.submit(_threaded_task_wrapper, "PRECOMP_TERR_VISION", self._precompute_territory_vision)

        # 9. Ensure all remaining parallel tasks are complete
        cloud_precomp_future.result()  # Wait for the early-started cloud precomputation
        connect_harbors_future.result()
        precompute_vision_future.result()

        internal_executor.shutdown(wait=True)

        self.execution_times[self.STEP_NAMES["TOTAL_INIT"]] = time.time() - total_init_start_time_timer
        if self.status_queue:
            self.status_queue.put_nowait((self.STEP_NAMES["TOTAL_INIT"], "FINISHED", self.execution_times[self.STEP_NAMES["TOTAL_INIT"]]))

    def prepare_for_pickling(self):
        method_start_time = time.time()

        if self.status_queue:
            expected_time = self.preset_times.get(self.STEP_NAMES["PREP_PICKLING"], 999.0)
            self.status_queue.put_nowait((self.STEP_NAMES["PREP_PICKLING"], "START", expected_time))

        self.baseMapSurf = None
        self.debugOverlayFullMap = None
        self.cloudSurfFullMap = None
        self.territoryHighlightSurfScreen = None
        self.playersSurfScreen = None

        self.font = None
        self.cloud_surf_initialized = False
        self.active_transparent_tiles = set()

        self.all_territories_for_unpickling = list(self.territories_by_id.values())
        if hasattr(self, '_temp_contiguous_territories_objs') and self._temp_contiguous_territories_objs:
            self.contiguousTerritoryIDs = []
            for terr_list in self._temp_contiguous_territories_objs:
                current_id_list = []
                for terr in terr_list:
                    if hasattr(terr, 'id'):
                        current_id_list.append(terr.id)
                self.contiguousTerritoryIDs.append(current_id_list)
            del self._temp_contiguous_territories_objs
        else:
            self.contiguousTerritoryIDs = []

        for tile in self.tiles:
            tile.prepare_for_pickling()
        for territory_obj in self.all_territories_for_unpickling:
            territory_obj.prepare_for_pickling()
            if hasattr(territory_obj, 'harbors'):
                for harbor in territory_obj.harbors:
                    if hasattr(harbor, 'prepare_for_pickling'):
                        harbor.prepare_for_pickling()

        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        self.territories_by_id = {}
        self.harbors_by_id = {}
        self.tiles_by_id = {}
        self.tiles_by_grid_coords = {}

        duration = time.time() - method_start_time
        self.execution_times[self.STEP_NAMES["PREP_PICKLING"]] = duration
        if self.status_queue:
            self.status_queue.put_nowait((self.STEP_NAMES["PREP_PICKLING"], "FINISHED", duration))

    def initialize_graphics_and_external_libs(self, fonts_dict, status_queue=None, preset_times=None):
        _local_q_gfx = status_queue
        _local_preset_times_gfx = preset_times if preset_times else {}

        if not hasattr(self, 'execution_times'):
            self.execution_times = {}

        GFX_TOTAL_INIT_STEP_NAME = self.STEP_NAMES["GFX_TOTAL_INIT"]

        expected_time = _local_preset_times_gfx.get(GFX_TOTAL_INIT_STEP_NAME, 999.0)
        if _local_q_gfx:
            _local_q_gfx.put_nowait((GFX_TOTAL_INIT_STEP_NAME, "START", expected_time))

        method_total_start_time = time.time()

        def _surface_setup_fn():
            self.baseMapSurf = pygame.Surface((self.mapWidth, self.mapHeight)).convert_alpha()
            self.debugOverlayFullMap = pygame.Surface((self.mapWidth, self.mapHeight), pygame.SRCALPHA)
            self.cloudSurfFullMap = pygame.Surface((self.mapWidth, self.mapHeight), pygame.SRCALPHA)

            self.territoryHighlightSurfScreen = pygame.Surface((self.viewportWidth, self.viewportHeight), pygame.SRCALPHA)
            self.playersSurfScreen = pygame.Surface((self.viewportWidth, self.viewportHeight), pygame.SRCALPHA)

            self.baseMapSurf.fill((0, 0, 0, 0))
            self.debugOverlayFullMap.fill((0, 0, 0, 0))
            self.cloudSurfFullMap.fill((0, 0, 0, 0))

            self.territoryHighlightSurfScreen.fill((0, 0, 0, 0))
            self.playersSurfScreen.fill((0, 0, 0, 0))

        def _font_setup_fn():
            if self.font_name and self.font_name in fonts_dict:
                self.font = fonts_dict[self.font_name]

        def _rebuild_maps_fn():
            if not hasattr(self, 'all_territories_for_unpickling'):
                self.all_territories_for_unpickling = []
            self.territories_by_id = {}
            self.harbors_by_id = {}
            all_harbors_temp = []
            for ter_obj in self.all_territories_for_unpickling:
                if hasattr(ter_obj, 'id'):
                    self.territories_by_id[ter_obj.id] = ter_obj
                    if hasattr(ter_obj, 'harbors') and isinstance(ter_obj.harbors, list):
                        all_harbors_temp.extend(ter_obj.harbors)
            h_id_counter = 0
            for h_obj in all_harbors_temp:
                n_id = h_id_counter
                if hasattr(h_obj, 'harbor_id') and h_obj.harbor_id != -1:
                    n_id = h_obj.harbor_id

                self.harbors_by_id[n_id] = h_obj
                if not hasattr(h_obj, 'harbor_id') or h_obj.harbor_id == -1:
                    h_obj.harbor_id = n_id
                h_id_counter = max(h_id_counter, n_id + 1)

        def _restore_terr_links_fn():
            for tile in self.tiles:
                tile.territory = None
                if hasattr(tile, 'territory_id') and tile.territory_id != -1:
                    tile.territory = self.territories_by_id.get(tile.territory_id)

        def _restore_adj_links_fn():
            if not self.tiles_by_id or not self.tiles_by_grid_coords:
                self.tiles_by_id = {}
                self.tiles_by_grid_coords = {}
                for tile in self.tiles:
                    self.tiles_by_id[tile.tile_id] = tile
                    self.tiles_by_grid_coords[(tile.grid_x, tile.grid_y)] = tile

            for tile in self.tiles:
                tile.adjacent = []
                if hasattr(tile, 'adjacent_tile_ids'):
                    for neighbor_id in tile.adjacent_tile_ids:
                        neighbor_obj = self.tiles_by_id.get(neighbor_id)
                        if neighbor_obj:
                            tile.adjacent.append(neighbor_obj)

        def _terr_harbor_gfx_fn():
            for ter_obj in self.territories_by_id.values():
                if hasattr(ter_obj, 'initialize_graphics_and_external_libs'):
                    ter_obj.initialize_graphics_and_external_libs(self.tiles_by_id, self.harbors_by_id, self.baseMapSurf, self.debugOverlayFullMap)

        def _update_reach_harbors_fn():
            for territory_obj in self.territories_by_id.values():
                if hasattr(territory_obj, 'update_reachable_harbors'):
                    territory_obj.update_reachable_harbors()

        _surface_setup_fn()
        _font_setup_fn()
        _rebuild_maps_fn()
        _restore_terr_links_fn()
        _restore_adj_links_fn()
        _terr_harbor_gfx_fn()
        _update_reach_harbors_fn()

        self.drawBaseMapStaticContent()
        self._initialize_full_map_cloud_surface_state()

        self.execution_times[GFX_TOTAL_INIT_STEP_NAME] = time.time() - method_total_start_time
        if _local_q_gfx:
            _local_q_gfx.put_nowait((GFX_TOTAL_INIT_STEP_NAME, "FINISHED", self.execution_times[GFX_TOTAL_INIT_STEP_NAME]))

    def print_all_execution_times(self):
        pass

    def _initialize_full_map_cloud_surface_state(self):
        if not self.cloudSurfFullMap:
            self.cloudSurfFullMap = pygame.Surface((self.mapWidth, self.mapHeight), pygame.SRCALPHA)
        self.cloudSurfFullMap.fill((0, 0, 0, 0))

        self.active_transparent_tiles.clear()
        for tile in self.tiles:
            tile.drawCloud(self.cloudSurfFullMap, 0, 0)
            if tile.cloudOpacity < 0.999:
                self.active_transparent_tiles.add(tile)
        self.cloud_surf_initialized = True

    def generationCycle(self):
        shifts = {'waterLand': [], 'mountainous': [], 'cloudy': []}
        for tile in self.tiles:
            if tile.adjacent:
                for prop_name in shifts:
                    avg_val = sum(getattr(adj, prop_name) for adj in tile.adjacent) / len(tile.adjacent)
                    shifts[prop_name].append(avg_val)
            else:
                for prop_name in shifts:
                    shifts[prop_name].append(getattr(tile, prop_name))

        for i, tile in enumerate(self.tiles):
            for prop_name in shifts:
                current_val = getattr(tile, prop_name)
                new_val = max(0.0, min(1.0, current_val + (shifts[prop_name][i] - current_val) / 2.0))
                setattr(tile, prop_name, new_val)

    def generateTiles(self):
        tile_id_counter = 0
        self.tiles = []
        self.tiles_by_id = {}
        self.tiles_by_grid_coords = {}
        self.precomp_grid_width = math.ceil(self.mapWidth / self.PRECOMP_CHUNK_SIZE)
        self.precomp_grid_height = math.ceil(self.mapHeight / self.PRECOMP_CHUNK_SIZE)
        self.precomp_chunk_grid_tiles = {}
        for cx_idx in range(self.precomp_grid_width):
            for cy_idx in range(self.precomp_grid_height):
                self.precomp_chunk_grid_tiles[(cx_idx, cy_idx)] = []

        for x_grid_idx in range(self.gridSizeX):
            for y_grid_idx in range(self.gridSizeY):
                x_pos = self.horizontal_distance * x_grid_idx + self.size * (self.borderSize + 0.5)
                y_pos = self.vertical_distance * y_grid_idx + self.size * (self.borderSize - 0.5)
                if x_grid_idx % 2 == 1:
                    y_pos += self.vertical_distance / 2

                hex_obj = Hex(x_grid_idx, y_grid_idx, x_pos, y_pos, self.size, tile_id_counter)

                pcx = int(hex_obj.x // self.PRECOMP_CHUNK_SIZE)
                pcy = int(hex_obj.y // self.PRECOMP_CHUNK_SIZE)
                hex_obj.precomp_chunk_coords = (pcx, pcy)

                if 0 <= pcx < self.precomp_grid_width and 0 <= pcy < self.precomp_grid_height:
                    self.precomp_chunk_grid_tiles[(pcx, pcy)].append(hex_obj)

                self.tiles.append(hex_obj)
                self.tiles_by_id[tile_id_counter] = hex_obj
                self.tiles_by_grid_coords[(x_grid_idx, y_grid_idx)] = hex_obj
                tile_id_counter += 1

    def _link_adjacent_objects(self):
        grid_obj_map = {(tile.grid_x, tile.grid_y): tile for tile in self.tiles}
        for tile in self.tiles:
            tile.adjacent = []
            if tile.grid_x % 2 == 0:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, -1)]
            else:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, 1)]

            for dx, dy in offsets:
                neighbor_grid_x = tile.grid_x + dx
                neighbor_grid_y = tile.grid_y + dy
                neighbor_obj = grid_obj_map.get((neighbor_grid_x, neighbor_grid_y))
                if neighbor_obj:
                    tile.adjacent.append(neighbor_obj)

    def getTileAtPosition(self, x_map, y_map):
        target_pcx = int(x_map // self.PRECOMP_CHUNK_SIZE)
        target_pcy = int(y_map // self.PRECOMP_CHUNK_SIZE)

        for dcx in [-1, 0, 1]:
            for dcy in [-1, 0, 1]:
                check_pcx = target_pcx + dcx
                check_pcy = target_pcy + dcy

                if 0 <= check_pcx < self.precomp_grid_width and 0 <= check_pcy < self.precomp_grid_height:

                    chunk_tiles = self.precomp_chunk_grid_tiles.get((check_pcx, check_pcy))
                    if chunk_tiles:
                        for tile in chunk_tiles:
                            if distance(tile.center, (x_map, y_map)) < tile.size:
                                return tile
        return None

    def setTileCols(self):
        for tile in self.tiles:
            tile.isLand = (tile.waterLand >= self.waterThreshold)
            tile.isMountain = (tile.isLand and tile.mountainous >= self.mountainThreshold)

        value_sets = {'water': [], 'land': [], 'mountain': [], 'cloud': []}
        for t in self.tiles:
            if not t.isLand:
                value_sets['water'].append(t.waterLand)
            elif t.isLand and not t.isMountain:
                value_sets['land'].append(t.waterLand)
            elif t.isMountain:
                value_sets['mountain'].append(t.mountainous)
            if hasattr(t, 'cloudy'):
                value_sets['cloud'].append(t.cloudy)

        bounds = {}
        for k, v_list in value_sets.items():
            if v_list:
                bounds[k] = [min(v_list), max(v_list)]
            else:
                bounds[k] = [0.0, 1.0]

        if bounds.get('water') and len(bounds['water']) == 2:
            bounds['water'][1] = self.waterThreshold
        if bounds.get('land') and len(bounds['land']) == 2:
            bounds['land'][0] = self.waterThreshold
        if bounds.get('mountain') and len(bounds['mountain']) == 2:
            bounds['mountain'][0] = self.mountainThreshold

        noise_levels = {'water': 0.0035, 'land': 0.004, 'mountain': 0.0021, 'cloud': 0.008}
        dist_funcs = {'water': lambda x_norm: (x_norm ** 2) / 2 + (1 - (1 - x_norm) ** 2) ** 10 / 2, 'land': lambda x_norm: (1 - 2 ** (-3 * x_norm)) * 8 / 7, 'cloud': lambda x_norm: (1 - 2 ** (-3 * x_norm)) * 8 / 7}

        self.allWaterTiles, self.allLandTiles, self.allCoastalTiles = [], [], []
        for tile in self.tiles:
            tile.isCoast = False

            cloud_noise = random.uniform(-noise_levels['cloud'], noise_levels['cloud'])
            norm_cloud = 0.5
            if bounds['cloud'][1] > bounds['cloud'][0]:
                norm_cloud = normalize(tile.cloudy + cloud_noise, bounds['cloud'][0], bounds['cloud'][1], clamp=True)
            tile.cloudCol = linearGradient([self.cols.cloudDark, self.cols.cloudMedium, self.cols.cloudLight], dist_funcs['cloud'](norm_cloud))

            if not tile.isLand:
                noise = random.uniform(-noise_levels['water'], noise_levels['water'])
                norm_val = 0.5
                if bounds['water'][1] > bounds['water'][0]:
                    norm_val = normalize(tile.waterLand + noise, bounds['water'][0], bounds['water'][1], clamp=True)
                tile.col = linearGradient([self.cols.oceanBlue, self.cols.oceanGreen, self.cols.lightOceanGreen, self.cols.oceanFoam], dist_funcs['water'](norm_val))
                self.allWaterTiles.append(tile)
            elif tile.isMountain:
                noise = random.uniform(-noise_levels['mountain'], noise_levels['mountain'])
                norm_val = 0.5
                if bounds['mountain'][1] > bounds['mountain'][0]:
                    norm_val = normalize(tile.mountainous + noise, bounds['mountain'][0], bounds['mountain'][1], clamp=True)
                tile.col = linearGradient([self.cols.mountainBlue, self.cols.darkMountainBlue], norm_val)
                self.allLandTiles.append(tile)
            else:
                noise = random.uniform(-noise_levels['land'], noise_levels['land'])
                norm_val = 0.5
                if bounds['land'][1] > bounds['land'][0]:
                    norm_val = normalize(tile.waterLand + noise, bounds['land'][0], bounds['land'][1], clamp=True)
                tile.col = linearGradient([self.cols.oliveGreen, self.cols.darkOliveGreen], dist_funcs['land'](norm_val))
                self.allLandTiles.append(tile)

        allWaterTilesSet = set(self.allWaterTiles)
        for tile in self.allLandTiles:
            is_coastal_tile = False
            for adj in tile.adjacent:
                if adj in allWaterTilesSet:
                    is_coastal_tile = True
                    break
            if is_coastal_tile:
                self.allCoastalTiles.append(tile)
                tile.isCoast = True

    def indexOceans(self):
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}

        unvisited_water_tiles = set(self.allWaterTiles)
        visited = set()
        current_ocean_id = 0
        while unvisited_water_tiles:
            start_tile = unvisited_water_tiles.pop()

            current_ocean_set = {start_tile}
            queue = deque([start_tile])
            visited.add(start_tile)
            self._ocean_id_map[start_tile] = current_ocean_id
            start_tile.connectedOceanID = current_ocean_id

            while queue:
                tile = queue.popleft()
                for neighbor in tile.adjacent:
                    if neighbor in unvisited_water_tiles:
                        neighbor.connectedOceanID = current_ocean_id
                        self._ocean_id_map[neighbor] = current_ocean_id
                        unvisited_water_tiles.remove(neighbor)
                        current_ocean_set.add(neighbor)
                        queue.append(neighbor)

            if current_ocean_set:
                self.oceanTiles[current_ocean_id] = current_ocean_set
                self._ocean_water[current_ocean_id] = current_ocean_set
                current_ocean_id += 1

    def assignCoastTiles(self):
        allWaterTilesSet = set(self.allWaterTiles)
        for coastTile in self.allCoastalTiles:
            ocean_ids_for_coast_tile = set()
            for adj in coastTile.adjacent:
                if adj in allWaterTilesSet and hasattr(adj, 'connectedOceanID') and adj.connectedOceanID != -1:
                    ocean_ids_for_coast_tile.add(adj.connectedOceanID)

            if ocean_ids_for_coast_tile:
                coastTile.connectedOceanID = max(ocean_ids_for_coast_tile)
            else:
                coastTile.connectedOceanID = -1

    @staticmethod
    def findContiguousRegions(tiles_to_check):
        visited = set()
        regions = []
        tilesSet = set(tiles_to_check)

        for tile in tiles_to_check:
            if tile not in visited:
                current_region = []
                q = deque([tile])
                visited.add(tile)
                while q:
                    curr = q.popleft()
                    current_region.append(curr)
                    for adj in curr.adjacent:
                        if adj in tilesSet and adj not in visited:
                            visited.add(adj)
                            q.append(adj)
                if current_region:
                    regions.append(current_region)
        return regions

    def createTerritories(self, land_regions_list):
        self.contiguousTerritoryIDs = []
        self.territories_by_id = {}
        self.all_territories_for_unpickling = []
        self._temp_contiguous_territories_objs = []

        tid_counter = 0
        for region_tiles in land_regions_list:
            if not region_tiles:
                continue

            centers = np.array([(t.x, t.y) for t in region_tiles])
            num_actual_tiles_in_region = len(region_tiles)

            n_clusters = max(1, math.ceil(num_actual_tiles_in_region / self.territorySize))
            n_clusters = min(n_clusters, num_actual_tiles_in_region)

            if n_clusters == 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=random.randint(0, 10000), n_init='auto', init='k-means++')
            assigned_labels = kmeans.fit_predict(centers)

            grouped_tiles_for_territories = [[] for _ in range(n_clusters)]
            for i, tile_obj in enumerate(region_tiles):
                grouped_tiles_for_territories[assigned_labels[i]].append(tile_obj)

            region_territory_objects_list = []
            for i in range(n_clusters):
                if grouped_tiles_for_territories[i]:
                    current_territory_tiles = grouped_tiles_for_territories[i]

                    cx = sum(t.x for t in current_territory_tiles) / len(current_territory_tiles)
                    cy = sum(t.y for t in current_territory_tiles) / len(current_territory_tiles)

                    terr = Territory(self.mapWidth, self.mapHeight, [cx, cy], current_territory_tiles, self.allWaterTiles, self.cols, self.resource_info, self.structure_info)
                    terr.id = tid_counter

                    self.all_territories_for_unpickling.append(terr)
                    self.territories_by_id[terr.id] = terr
                    region_territory_objects_list.append(terr)
                    for t_in_terr in current_territory_tiles:
                        t_in_terr.territory_id = terr.id
                    tid_counter += 1

            if region_territory_objects_list:
                self._temp_contiguous_territories_objs.append(region_territory_objects_list)

    def connectTerritoryHarbors(self):
        self.allHarbors = []
        for terr_obj in self.all_territories_for_unpickling:
            if hasattr(terr_obj, 'harbors') and isinstance(terr_obj.harbors, list):
                self.allHarbors.extend(terr_obj.harbors)

        if not self.allHarbors:
            return 0

        self.harbors_by_id = {}
        hid_counter = 0
        for h_obj in self.allHarbors:
            if not hasattr(h_obj, 'harbor_id') or h_obj.harbor_id == -1:
                h_obj.harbor_id = hid_counter
            self.harbors_by_id[h_obj.harbor_id] = h_obj
            hid_counter = max(hid_counter, h_obj.harbor_id + 1)

        harbors_by_ocean = {}
        ocean_harbors_by_id_map = {}

        for h_obj in self.allHarbors:
            if hasattr(h_obj, 'tile') and h_obj.tile and hasattr(h_obj.tile, 'adjacent'):
                for adj_tile in h_obj.tile.adjacent:
                    if adj_tile in self._ocean_id_map:
                        ocean_id = self._ocean_id_map[adj_tile]
                        if ocean_id not in harbors_by_ocean:
                            harbors_by_ocean[ocean_id] = []
                            ocean_harbors_by_id_map[ocean_id] = {}
                        harbors_by_ocean[ocean_id].append(h_obj)
                        ocean_harbors_by_id_map[ocean_id][h_obj.harbor_id] = h_obj
                        break

        routes_found_count = 0
        for ocean_id, harbors_in_ocean_list in harbors_by_ocean.items():
            if len(harbors_in_ocean_list) < 2:
                continue

            water_tile_set_for_ocean = self._ocean_water.get(ocean_id)
            if not water_tile_set_for_ocean:
                continue

            current_ocean_harbors_id_map = ocean_harbors_by_id_map.get(ocean_id, {})

            for i, src_harbor in enumerate(harbors_in_ocean_list):
                destination_harbors = [h for h in harbors_in_ocean_list[i + 1:] if h.harbor_id != -1]
                if not destination_harbors:
                    continue

                if not hasattr(src_harbor, 'generateAllRoutes'):
                    continue

                routes_found_count += src_harbor.generateAllRoutes(destination_harbors, water_tile_set_for_ocean, current_ocean_harbors_id_map)

        print(f"WORKER STDOUT: Found/Generated {routes_found_count} harbor routes.")
        return len(self.allHarbors)

    def _precompute_territory_vision(self):
        self.territory_vision_cache = {}
        vision_range_multiplier = GenerationInfo.territoryVisionRange

        for terr_id, terr_obj in self.territories_by_id.items():
            if not terr_obj.tiles:
                continue

            reveal_radius_px = vision_range_multiplier * self.size
            temp_vision_opacities = {}

            for tile_in_terr in terr_obj.tiles:
                self._calculate_fov_opacities_for_source(tile_in_terr.center, reveal_radius_px, temp_vision_opacities, source_tile_obj=tile_in_terr)

            self.territory_vision_cache[terr_id] = {(tile_obj.tile_id, opacity) for tile_obj, opacity in temp_vision_opacities.items()}

    @staticmethod
    def _does_chunk_intersect_radius(chunk_relative_coords, reveal_center_px, radius_px, chunk_size_px):
        cmin_x = chunk_relative_coords[0] * chunk_size_px
        cmin_y = chunk_relative_coords[1] * chunk_size_px
        cmax_x = (chunk_relative_coords[0] + 1) * chunk_size_px
        cmax_y = (chunk_relative_coords[1] + 1) * chunk_size_px

        closest_x = max(cmin_x, min(reveal_center_px[0], cmax_x))
        closest_y = max(cmin_y, min(reveal_center_px[1], cmax_y))

        dist_sq = (reveal_center_px[0] - closest_x) ** 2 + (reveal_center_px[1] - closest_y) ** 2
        return dist_sq < (radius_px * radius_px)

    def _generate_all_cloud_precomp_data(self):
        print(f"WORKER STDOUT: Generating all cloud precomputation data for {self.mapWidth}x{self.mapHeight} grid with size {self.size}")

        self._relative_hex_offsets_pattern = []
        max_precomp_radius = self.precomp_radii[-1] if self.precomp_radii else 600
        max_hex_grid_dist = math.ceil(max_precomp_radius / (self.size * math.sqrt(3) / 2)) + 3

        for q_grid_idx in range(-max_hex_grid_dist, max_hex_grid_dist + 1):
            for r_grid_idx in range(-max_hex_grid_dist, max_hex_grid_dist + 1):
                rel_x_px = self.horizontal_distance * q_grid_idx
                rel_y_px = self.vertical_distance * r_grid_idx
                if q_grid_idx % 2 == 1:
                    rel_y_px += self.vertical_distance / 2

                dist_from_origin_px = math.sqrt(rel_x_px ** 2 + rel_y_px ** 2)

                if dist_from_origin_px <= max_precomp_radius + self.size * 1.5:
                    self._relative_hex_offsets_pattern.append({'rel_grid_x': q_grid_idx, 'rel_grid_y': r_grid_idx, 'dist_px': dist_from_origin_px})
        self._relative_hex_offsets_pattern.sort(key=lambda x: x['dist_px'])
        print(f"WORKER STDOUT: Precomputed {len(self._relative_hex_offsets_pattern)} relative hex distance patterns.")

        self.precomputed_cloud_info = {}
        max_r_val = self.precomp_radii[-1] if self.precomp_radii else 0
        max_dist_chunks_buffer = self.size * 2
        max_dist_chunks = math.ceil((max_r_val + max_dist_chunks_buffer) / self.PRECOMP_CHUNK_SIZE)
        pattern_reveal_center_rel_px = (0.5 * self.PRECOMP_CHUNK_SIZE, 0.5 * self.PRECOMP_CHUNK_SIZE)

        for r_idx, rad_px in enumerate(self.precomp_radii):
            relevant_relative_chunks = set()
            for drx in range(-max_dist_chunks, max_dist_chunks + 1):
                for dry in range(-max_dist_chunks, max_dist_chunks + 1):
                    relative_chunk_coord = (drx, dry)
                    if self._does_chunk_intersect_radius(relative_chunk_coord, pattern_reveal_center_rel_px, rad_px, self.PRECOMP_CHUNK_SIZE):
                        relevant_relative_chunks.add(relative_chunk_coord)
            self.precomputed_cloud_info[r_idx] = relevant_relative_chunks
        print(f"WORKER STDOUT: Precomputed {len(self.precomputed_cloud_info)} chunk-based relative patterns.")

    def _precompute_cloud_visibility(self):
        if not os.path.exists(CLOUD_PRECOMP_DIR):
            os.makedirs(CLOUD_PRECOMP_DIR)

        filename = f"cloud_precomp_data_{self.mapWidth}x{self.mapHeight}_size{self.size}_chunk{self.PRECOMP_CHUNK_SIZE}.pkl.gz"
        filepath = os.path.join(CLOUD_PRECOMP_DIR, filename)

        loaded = False
        if os.path.exists(filepath):
            print(f"WORKER STDOUT: Attempting to load precomputed cloud data from {filepath}")
            try:
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self._relative_hex_offsets_pattern = data.get('relative_hex_offsets_pattern', [])
                    self.precomputed_cloud_info = {k: set(v) for k, v in data.get('cloud_info_patterns', {}).items()}
                print(f"WORKER STDOUT: Successfully loaded precomputed cloud data.")
                loaded = True
            except (IOError, pickle.UnpicklingError, EOFError) as e:
                print(f"WORKER STDOUT: Error loading precomputed data: {e}. Re-generating.")
                self._relative_hex_offsets_pattern = []
                self.precomputed_cloud_info = {}

        if not loaded:
            self._generate_all_cloud_precomp_data()

            try:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump({'relative_hex_offsets_pattern': self._relative_hex_offsets_pattern, 'cloud_info_patterns': self.precomputed_cloud_info}, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"WORKER STDOUT: Generated and saved precomputed cloud data to {filepath}.")
            except Exception as e:
                print(f"WORKER STDOUT: Error saving precomputed data: {e}")

    def _calculate_fov_opacities_for_source(self, center_coords_px, desired_reveal_radius_px, tile_opacity_map_to_update, source_tile_obj=None):
        if source_tile_obj:
            for entry in self._relative_hex_offsets_pattern:
                rel_grid_x = entry['rel_grid_x']
                rel_grid_y = entry['rel_grid_y']
                dist_px = entry['dist_px']

                if dist_px > desired_reveal_radius_px + self.size * 1.5:
                    break

                target_grid_x = source_tile_obj.grid_x + rel_grid_x
                target_grid_y = source_tile_obj.grid_y + rel_grid_y

                tile_obj = self.tiles_by_grid_coords.get((target_grid_x, target_grid_y))

                if tile_obj:
                    opacity_val = (dist_px / desired_reveal_radius_px) ** 0.9 if desired_reveal_radius_px > 1e-6 else (0.0 if dist_px < 1e-6 else 1.0)
                    current_min_opacity = tile_opacity_map_to_update.get(tile_obj, 1.0)
                    tile_opacity_map_to_update[tile_obj] = min(current_min_opacity, max(0.0, min(1.0, opacity_val)))
        else:
            center_pcx = int(center_coords_px[0] // self.PRECOMP_CHUNK_SIZE)
            center_pcy = int(center_coords_px[1] // self.PRECOMP_CHUNK_SIZE)

            radius_idx_for_precomp = len(self.precomp_radii) - 1
            for i, r_val in enumerate(self.precomp_radii):
                if desired_reveal_radius_px <= r_val:
                    radius_idx_for_precomp = i
                    break

            relevant_relative_chunks = self.precomputed_cloud_info.get(radius_idx_for_precomp)
            if not relevant_relative_chunks:
                return

            for drx, dry in relevant_relative_chunks:
                abs_chunk_x = center_pcx + drx
                abs_chunk_y = center_pcy + dry

                if 0 <= abs_chunk_x < self.precomp_grid_width and 0 <= abs_chunk_y < self.precomp_grid_height:

                    chunk_tiles = self.precomp_chunk_grid_tiles.get((abs_chunk_x, abs_chunk_y))
                    if chunk_tiles:
                        for tile_obj in chunk_tiles:
                            d = distance(center_coords_px, tile_obj.center)
                            if desired_reveal_radius_px > 1e-6:
                                if d < desired_reveal_radius_px:
                                    opacity_val = (d / desired_reveal_radius_px) ** 0.9
                                else:
                                    opacity_val = 1.0
                            else:
                                opacity_val = 0.0 if d < 1e-6 else 1.0

                            current_min_opacity = tile_opacity_map_to_update.get(tile_obj, 1.0)
                            tile_opacity_map_to_update[tile_obj] = min(current_min_opacity, max(0.0, min(1.0, opacity_val)))

    def drawBaseMapStaticContent(self):
        if not self.baseMapSurf or not self.debugOverlayFullMap:
            return

        base_map_fill_color = (0, 0, 100)
        if hasattr(self.cols, 'oceanBlue'):
            base_map_fill_color = self.cols.oceanBlue
        self.baseMapSurf.fill(base_map_fill_color)

        for tile in self.tiles:
            tile.draw(self.baseMapSurf)

        self.debugOverlayFullMap.fill((0, 0, 0, 0))
        for tile in self.tiles:
            if tile.territory and hasattr(tile.territory, 'territoryCol'):
                pygame.draw.polygon(self.debugOverlayFullMap, tile.territory.territoryCol, tile.hex)

        for id_list in self.contiguousTerritoryIDs:
            for tid in id_list:
                terr = self.territories_by_id.get(tid)
                if terr:
                    terr.drawInternalTerritoryBaseline(self.baseMapSurf, self.debugOverlayFullMap)
        for id_list in self.contiguousTerritoryIDs:
            for tid in id_list:
                terr = self.territories_by_id.get(tid)
                if terr:
                    terr.drawInternalStructures(self.baseMapSurf)

    def drawTerritoryHighlights(self, s, hovered_territory=None, selected_territory=None, scroll=(0, 0)):
        if not self.territoryHighlightSurfScreen:
            return

        self.territoryHighlightSurfScreen.fill((0, 0, 0, 0))

        scroll_x, scroll_y = scroll[0], scroll[1]

        if hovered_territory:
            if selected_territory is None:
                hovered_territory.drawCurrent(self.territoryHighlightSurfScreen, 'b', scroll_x, scroll_y)
            else:
                if hovered_territory == selected_territory:
                    hovered_territory.drawCurrent(self.territoryHighlightSurfScreen, 'r', scroll_x, scroll_y)
                else:
                    selected_territory.drawCurrent(self.territoryHighlightSurfScreen, 'r', scroll_x, scroll_y)
                    hovered_territory.drawCurrent(self.territoryHighlightSurfScreen, 'b', scroll_x, scroll_y)
                selected_territory.drawRoutes(self.territoryHighlightSurfScreen, self.cols.brightCrimson, scroll_x, scroll_y)
            if (selected_territory is None) or (selected_territory == hovered_territory):
                hovered_territory.drawRoutes(self.territoryHighlightSurfScreen, self.cols.brightCrimson, scroll_x, scroll_y)
        else:
            if selected_territory is not None:
                selected_territory.drawCurrent(self.territoryHighlightSurfScreen, 'r', scroll_x, scroll_y)
                selected_territory.drawRoutes(self.territoryHighlightSurfScreen, self.cols.brightCrimson, scroll_x, scroll_y)

        s.blit(self.territoryHighlightSurfScreen, (0, 0))

    def drawClouds(self, s, mx_map, my_map, mouseSize, playerObj, scroll_pos, screen_dims, player_visible_territory_ids: set):
        if not self.cloud_surf_initialized:
            self._initialize_full_map_cloud_surface_state()
        if not self.cloudSurfFullMap:
            return

        tiles_to_redraw = set()
        opacities_this_frame = {}

        mouse_vision_radius_options = [50, 150, 300, 600]
        mouse_vision_radius = mouse_vision_radius_options[mouseSize % len(mouse_vision_radius_options)]

        self._calculate_fov_opacities_for_source((mx_map, my_map), mouse_vision_radius, opacities_this_frame)

        if hasattr(playerObj, 'ships') and isinstance(playerObj.ships, list):
            for ship in playerObj.ships:
                if hasattr(ship, 'pos') and ship.pos and hasattr(ship, 'currentVision'):
                    ship_vision_px = ship.currentVision * self.size
                    self._calculate_fov_opacities_for_source(ship.pos, ship_vision_px, opacities_this_frame)

        for terr_id in player_visible_territory_ids:
            territory_vision_data = self.territory_vision_cache.get(terr_id)
            if territory_vision_data:
                for tile_id, precomputed_opacity in territory_vision_data:
                    tile_obj = self.tiles_by_id.get(tile_id)
                    if tile_obj:
                        current_min_opacity = opacities_this_frame.get(tile_obj, 1.0)
                        opacities_this_frame[tile_obj] = min(current_min_opacity, precomputed_opacity)

        new_active_transparent = set()
        for tile, new_op in opacities_this_frame.items():
            if abs(tile.cloudOpacity - new_op) > 0.001:
                tile.cloudOpacity = new_op
                tiles_to_redraw.add(tile)
            if tile.cloudOpacity < 0.999:
                new_active_transparent.add(tile)

        for tile in (self.active_transparent_tiles - new_active_transparent):
            if tile not in opacities_this_frame:
                if abs(tile.cloudOpacity - 1.0) > 0.001:
                    tile.cloudOpacity = 1.0
                    tiles_to_redraw.add(tile)
        self.active_transparent_tiles = new_active_transparent

        for tile in tiles_to_redraw:
            # self.cloudSurfFullMap.fill((0, 0, 0, 0), rect=tile.bounding_rect)
            tile.drawCloud(self.cloudSurfFullMap, 0, 0)

        visible_map_rect_on_full_map = pygame.Rect(-scroll_pos[0], -scroll_pos[1], screen_dims[0], screen_dims[1])
        s.blit(self.cloudSurfFullMap.subsurface(visible_map_rect_on_full_map), (0, 0))
