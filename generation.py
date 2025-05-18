import pygame
import math
import random
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from text import drawText
from calcs import distance, ang, normalize_angle, draw_arrow, linearGradient, normalize, setOpacity
from territory import Territory

try:
    from shapely.geometry import Polygon as ShapelyPolygon

    SHAPELY_AVAILABLE_FOR_HINTS = True
except ImportError:
    SHAPELY_AVAILABLE_FOR_HINTS = False


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
        self.cloudOpacity = 1.0  # Start fully opaque
        self.target_opacity_this_frame = 1.0  # Target opacity for the current frame calculation
        self.precomp_chunk_coords = None
        self.debug_cloud_color_override = None

    def prepare_for_pickling(self):
        self.adjacent_tile_ids = [adj.tile_id for adj in self.adjacent if hasattr(adj, 'tile_id')]
        self.adjacent = []
        self.territory = None
        self.region = None
        self.mountainRegion = None  # self.target_opacity_this_frame can be left as it will be reset

    def draw(self, s):
        pygame.draw.polygon(s, self.col, self.hex)

    def drawArrows(self, s, color):
        if not self.adjacent: return
        arrow_col = tuple(color) if color else (255, 0, 0)
        for adj in self.adjacent:
            try:
                angle = normalize_angle(ang((self.x, self.y), (adj.x, adj.y)))
                dist_val = distance((self.x, self.y), (adj.x, adj.y))
                factor = 0.35
                draw_arrow(s, (self.x, self.y), (self.x + dist_val * factor * math.cos(angle), self.y + dist_val * factor * math.sin(angle)), arrow_col, pygame, 2, 5, 25)
            except Exception as e:
                print(f"Error drawing arrow from ({self.grid_x},{self.grid_y}) to ({adj.grid_x},{adj.grid_y}): {e}")

    def showWaterLand(self, s, font, color):
        if font:
            try:
                text_col = tuple(color) if color else (0, 0, 0)
                drawText(s, text_col, font, self.x, self.y, f"{self.waterLand:.2f}", justify="center", centeredVertically=True)
            except Exception as e:
                print(f"Error drawing text on Hex ({self.grid_x},{self.grid_y}): {e}")


class TileHandler:
    def __init__(self, width, height, size, cols, waterThreshold=0.51, mountainThreshold=0.51, territorySize=100, font=None, font_name=None, resource_info=None, structure_info=None):
        self.screenWidth, self.screenHeight = width, height
        self.cols = cols
        self.font = font
        self.font_name = font_name
        self.resource_info = resource_info
        self.structure_info = structure_info
        self.size = size
        self.territorySize = territorySize
        self.tiles = []
        self.tiles_by_id = {}
        self.territories_by_id = {}
        self.harbors_by_id = {}
        self.contiguousTerritoryIDs = []
        self.all_territories_for_unpickling = []
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        self.waterThreshold = waterThreshold
        self.mountainThreshold = mountainThreshold
        self.borderSize = 4
        self.horizontal_distance = (3 / 2 * size)
        self.vertical_distance = (math.sqrt(3) * size)
        self.gridSizeX = int(width / self.horizontal_distance) - self.borderSize - 1
        self.gridSizeY = int(height / self.vertical_distance) - self.borderSize + 1
        self.allWaterTiles = []
        self.allLandTiles = []
        self.allCoastalTiles = []
        self.surf = None
        self.debugOverlay = None
        self.territorySurf = None
        self.playersSurf = None
        self.cloudSurf = None
        self.PRECOMP_CHUNK_SIZE = 30
        self.precomp_radii = sorted([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        self.precomputed_cloud_info = {}
        self.precomp_chunk_grid_tiles = {}
        self.precomp_grid_width = 0
        self.precomp_grid_height = 0
        # self.persistent_revealed_tiles is removed as we will iterate all tiles for drawing clouds
        # self.tiles_potentially_affected_this_frame is also removed for simplification
        self._temp_contiguous_territories_objs = None
        print("\nGenerating tiles...")
        self.generateTiles()
        print("Precomputing cloud visibility patterns...")
        self._precompute_cloud_visibility()
        self._link_adjacent_objects()
        print("Running generation cycles...")
        for _ in range(50): self.generationCycle()
        print("Setting tile colors...")
        self.setTileCols()
        print("Finding contiguous regions...")
        landRegionsRaw = self.findContiguousRegions([t for t in self.tiles if t.waterLand >= self.waterThreshold])
        print("Indexing Oceans...")
        self.indexOceans()
        print("Assigning coast land to oceans...")
        self.assignCoastTiles()
        print("Creating territories...")
        self.createTerritories(landRegionsRaw)
        print("Connecting Harbors...")
        self.allHarbors = []
        self.connectTerritoryHarbors()
        print("TileHandler Worker Initialization Complete.")

    def prepare_for_pickling(self):
        print("  TileHandler: Preparing for pickling...")
        self.surf, self.debugOverlay, self.territorySurf, self.playersSurf, self.cloudSurf = None, None, None, None, None
        self.font = None
        self.all_territories_for_unpickling = list(self.territories_by_id.values())
        self.contiguousTerritoryIDs = [[terr.id for terr in terr_list] for terr_list in self._temp_contiguous_territories_objs] if hasattr(self, '_temp_contiguous_territories_objs') and self._temp_contiguous_territories_objs else []
        if hasattr(self, '_temp_contiguous_territories_objs'): del self._temp_contiguous_territories_objs
        for tile in self.tiles: tile.prepare_for_pickling()
        for territory in self.all_territories_for_unpickling:
            territory.prepare_for_pickling()
            for harbor in territory.harbors: harbor.prepare_for_pickling()
        self.oceanTiles, self._ocean_id_map, self._ocean_water = {}, {}, {}
        self.territories_by_id, self.harbors_by_id = {}, {}
        print("  TileHandler: Pickling preparation complete.")

    def initialize_graphics_and_external_libs(self, fonts_dict):
        print("  TileHandler: Initializing graphics and external libs...")
        self.surf = pygame.Surface((self.screenWidth, self.screenHeight)).convert_alpha()
        self.debugOverlay = pygame.Surface((self.screenWidth, self.screenHeight)).convert_alpha()
        self.territorySurf = pygame.Surface((self.screenWidth, self.screenHeight), pygame.SRCALPHA)
        self.playersSurf = pygame.Surface((self.screenWidth, self.screenHeight), pygame.SRCALPHA)
        self.cloudSurf = pygame.Surface((self.screenWidth, self.screenHeight), pygame.SRCALPHA)
        base_map_fill_color = self.cols.oceanBlue if hasattr(self.cols, 'oceanBlue') else (0, 0, 100)
        self.surf.fill(base_map_fill_color)
        self.debugOverlay.fill((0, 0, 0, 0))
        self.territorySurf.fill((0, 0, 0, 0))
        self.playersSurf.fill((0, 0, 0, 0))
        self.cloudSurf.fill((0, 0, 0, 0))
        if self.font_name and self.font_name in fonts_dict:
            self.font = fonts_dict[self.font_name]
        elif self.font_name:
            print(f"  TileHandler Warning: Font name '{self.font_name}' not found.")
        self.tiles_by_id = {tile.tile_id: tile for tile in self.tiles if hasattr(tile, 'tile_id')}
        print(f"  TileHandler: Rebuilt tiles_by_id map ({len(self.tiles_by_id)} entries).")
        if not hasattr(self, 'all_territories_for_unpickling') or not self.all_territories_for_unpickling:
            print("  Error: Cannot find pickled territory objects for map rebuilding.")
            return
        print(f"  TileHandler: Rebuilding territory/harbor maps from {len(self.all_territories_for_unpickling)} territories...")
        self.territories_by_id, self.harbors_by_id = {}, {}
        all_harbors_temp = []
        for ter in self.all_territories_for_unpickling:
            if hasattr(ter, 'id'):
                self.territories_by_id[ter.id] = ter
                if hasattr(ter, 'harbors'):
                    all_harbors_temp.extend(ter.harbors)
        harbor_id_counter = 0
        for harbor in all_harbors_temp:
            new_id = harbor.harbor_id if hasattr(harbor, 'harbor_id') and harbor.harbor_id != -1 else harbor_id_counter
            self.harbors_by_id[new_id] = harbor
            harbor.harbor_id = new_id
            harbor_id_counter = max(harbor_id_counter, new_id + 1)
        print(f"  TileHandler: Rebuilt {len(self.territories_by_id)} territories, {len(self.harbors_by_id)} harbors.")
        restored_terr_links = 0
        for tile in self.tiles:
            tile.territory = None
            if hasattr(tile, 'territory_id') and tile.territory_id != -1:
                territory_obj = self.territories_by_id.get(tile.territory_id)
                if territory_obj:
                    tile.territory = territory_obj
                    restored_terr_links += 1
        print(f"  TileHandler: Restored {restored_terr_links} Hex -> Territory links.")
        restored_adj_links, total_adj = 0, 0
        for tile in self.tiles:
            tile.adjacent = []
            if hasattr(tile, 'adjacent_tile_ids'):
                total_adj += len(tile.adjacent_tile_ids)
                for neighbor_id in tile.adjacent_tile_ids:
                    neighbor_obj = self.tiles_by_id.get(neighbor_id)
                    if neighbor_obj:
                        tile.adjacent.append(neighbor_obj)
                        restored_adj_links += 1
        print(f"  TileHandler: Restored {restored_adj_links}/{total_adj} Hex -> Adjacent links.")
        print("  TileHandler: Initializing territory graphics...")
        for territory in self.territories_by_id.values():
            if hasattr(territory, 'initialize_graphics_and_external_libs'): territory.initialize_graphics_and_external_libs()
        print("  TileHandler: Initializing harbor routes...")
        for harbor in self.harbors_by_id.values():
            if hasattr(harbor, 'initialize_graphics_and_external_libs'): harbor.initialize_graphics_and_external_libs(self.tiles_by_id, self.harbors_by_id)
        print("  TileHandler: Updating territory reachableHarbors...")
        for territory in self.territories_by_id.values():
            if hasattr(territory, 'update_reachable_harbors'): territory.update_reachable_harbors()
        print("  TileHandler: Drawing static elements (terrain) to internal surfaces...")
        self.draw2InternalScreen()
        print("  TileHandler: Graphics and external libs initialized.")

    def generationCycle(self):
        shifts = {prop: [] for prop in ['waterLand', 'mountainous', 'cloudy']}
        for tile in self.tiles:
            if tile.adjacent:
                for prop in shifts: shifts[prop].append(sum(getattr(adj, prop) for adj in tile.adjacent) / len(tile.adjacent))
            else:
                for prop in shifts: shifts[prop].append(getattr(tile, prop))
        for i, tile in enumerate(self.tiles):
            for prop in shifts:
                current_val = getattr(tile, prop)
                new_val = max(0.0, min(1.0, current_val + (shifts[prop][i] - current_val) / 2))
                setattr(tile, prop, new_val)

    def generateTiles(self):
        tile_id_counter = 0
        self.tiles = []
        self.tiles_by_id = {}
        self.precomp_grid_width = math.ceil(self.screenWidth / self.PRECOMP_CHUNK_SIZE)
        self.precomp_grid_height = math.ceil(self.screenHeight / self.PRECOMP_CHUNK_SIZE)
        self.precomp_chunk_grid_tiles = {(cx, cy): [] for cx in range(self.precomp_grid_width) for cy in range(self.precomp_grid_height)}
        for x_grid_idx in range(self.gridSizeX):
            for y_grid_idx in range(self.gridSizeY):
                x_pos = self.horizontal_distance * x_grid_idx + self.size * (self.borderSize + 0.5)
                y_pos = self.vertical_distance * y_grid_idx + self.size * (self.borderSize - 0.5)
                if x_grid_idx % 2 == 1: y_pos += self.vertical_distance / 2
                hex_obj = Hex(x_grid_idx, y_grid_idx, x_pos, y_pos, self.size, tile_id_counter)
                pcx, pcy = int(hex_obj.x // self.PRECOMP_CHUNK_SIZE), int(hex_obj.y // self.PRECOMP_CHUNK_SIZE)
                hex_obj.precomp_chunk_coords = (pcx, pcy)
                if 0 <= pcx < self.precomp_grid_width and 0 <= pcy < self.precomp_grid_height:
                    self.precomp_chunk_grid_tiles[(pcx, pcy)].append(hex_obj)
                self.tiles.append(hex_obj)
                self.tiles_by_id[tile_id_counter] = hex_obj
                tile_id_counter += 1
        print(f"  Generated {len(self.tiles)} tiles. Precomp cloud grid: {self.precomp_grid_width}x{self.precomp_grid_height} chunks.")

    def _link_adjacent_objects(self):
        grid_obj_map = {(tile.grid_x, tile.grid_y): tile for tile in self.tiles}
        for tile in self.tiles:
            tile.adjacent = []
            offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1 if tile.grid_x % 2 == 0 else 1), (1, -1 if tile.grid_x % 2 == 0 else 1)]
            for dx, dy in offsets:
                neighbor_obj = grid_obj_map.get((tile.grid_x + dx, tile.grid_y + dy))
                if neighbor_obj: tile.adjacent.append(neighbor_obj)

    def getTileAtPosition(self, x, y):
        for tile in self.tiles:
            if distance(tile.center, (x, y)) < tile.size: return tile
        return None

    def setTileCols(self):
        for tile in self.tiles:
            tile.isLand = (tile.waterLand >= self.waterThreshold)
            tile.isMountain = (tile.isLand and tile.mountainous >= self.mountainThreshold)
        value_sets = {'water': [t.waterLand for t in self.tiles if not t.isLand], 'land': [t.waterLand for t in self.tiles if t.isLand and not t.isMountain], 'mountain': [t.mountainous for t in self.tiles if t.isMountain], 'cloud': [t.cloudy for t in self.tiles if hasattr(t, 'cloudy')]}
        bounds = {k: [min(v), max(v)] if v else ([0.0, 1.0] if k != 'cloud' else [0.0, 1.0]) for k, v in value_sets.items()}
        if bounds['water']: bounds['water'][1] = self.waterThreshold
        if bounds['land']: bounds['land'][0] = self.waterThreshold
        if bounds['mountain']: bounds['mountain'][0] = self.mountainThreshold
        noise_levels = {'water': 0.0035, 'land': 0.004, 'mountain': 0.0021, 'cloud': 0.008}
        dist_funcs = {'water': lambda x: (x ** 2) / 2 + (1 - (1 - x) ** 2) ** 10 / 2, 'land': lambda x: (1 - 2 ** (-3 * x)) * 8 / 7, 'cloud': lambda x: (1 - 2 ** (-3 * x)) * 8 / 7}
        self.allWaterTiles, self.allLandTiles, self.allCoastalTiles = [], [], []
        for tile in self.tiles:
            tile.isCoast = False
            cloud_noise = random.uniform(-noise_levels['cloud'], noise_levels['cloud'])
            norm_cloud = 0.5 if bounds['cloud'][1] == bounds['cloud'][0] else normalize(tile.cloudy + cloud_noise, *bounds['cloud'], True)
            tile.cloudCol = linearGradient([self.cols.cloudDark, self.cols.cloudMedium, self.cols.cloudLight], dist_funcs['cloud'](norm_cloud))
            if not tile.isLand:
                noise = random.uniform(-noise_levels['water'], noise_levels['water'])
                norm_val = normalize(tile.waterLand + noise, *bounds['water'], True)
                tile.col = linearGradient([self.cols.oceanBlue, self.cols.oceanGreen, self.cols.lightOceanGreen, self.cols.oceanFoam], dist_funcs['water'](norm_val))
                self.allWaterTiles.append(tile)
            elif tile.isMountain:
                noise = random.uniform(-noise_levels['mountain'], noise_levels['mountain'])
                norm_val = normalize(tile.mountainous + noise, *bounds['mountain'], True)
                tile.col = linearGradient([self.cols.mountainBlue, self.cols.darkMountainBlue], norm_val)
                self.allLandTiles.append(tile)
            else:
                noise = random.uniform(-noise_levels['land'], noise_levels['land'])
                norm_val = normalize(tile.waterLand + noise, *bounds['land'], True)
                tile.col = linearGradient([self.cols.oliveGreen, self.cols.darkOliveGreen], dist_funcs['land'](norm_val))
                self.allLandTiles.append(tile)
        allWaterTilesSet = set(self.allWaterTiles)
        for tile in self.allLandTiles:
            if any(adj in allWaterTilesSet for adj in tile.adjacent):
                self.allCoastalTiles.append(tile)
                tile.isCoast = True
        print(f"  Tile colors set. {len(self.allCoastalTiles)} coastal tiles identified.")

    def indexOceans(self):
        self.oceanTiles, self._ocean_id_map, self._ocean_water = {}, {}, {}
        unvisited_water_tiles = set(self.allWaterTiles)
        current_ocean_id = 0
        while unvisited_water_tiles:
            start_tile = unvisited_water_tiles.pop()
            current_ocean_set, queue = {start_tile}, deque([start_tile])
            self._ocean_id_map[start_tile], start_tile.connectedOceanID = current_ocean_id, current_ocean_id
            while queue:
                tile = queue.popleft()
                for neighbor in tile.adjacent:
                    if neighbor in unvisited_water_tiles:
                        neighbor.connectedOceanID, self._ocean_id_map[neighbor] = current_ocean_id, current_ocean_id
                        unvisited_water_tiles.remove(neighbor)
                        current_ocean_set.add(neighbor)
                        queue.append(neighbor)
            if current_ocean_set:
                self.oceanTiles[current_ocean_id], self._ocean_water[current_ocean_id] = current_ocean_set, current_ocean_set
                current_ocean_id += 1
        print(f"  Indexed {len(self.oceanTiles)} distinct water bodies.")

    def assignCoastTiles(self):
        allWaterTilesSet, assigned_count = set(self.allWaterTiles), 0
        for coastTile in self.allCoastalTiles:
            ocean_ids = [adj.connectedOceanID for adj in coastTile.adjacent if adj in allWaterTilesSet and hasattr(adj, 'connectedOceanID') and adj.connectedOceanID != -1]
            coastTile.connectedOceanID = max(ocean_ids) if ocean_ids else -1
            if coastTile.connectedOceanID != -1: assigned_count += 1
        print(f"Assigned {assigned_count}/{len(self.allCoastalTiles)} coastal tiles to their oceans.")

    @staticmethod
    def findContiguousRegions(tiles_to_check):
        visited, regions, tilesSet = set(), [], set(tiles_to_check)
        for tile in tiles_to_check:
            if tile in visited: continue
            current_region, stack = [], [tile]
            while stack:
                curr = stack.pop()
                visited.add(curr)
                current_region.append(curr)
                for adj in curr.adjacent:
                    if adj in tilesSet and adj not in visited: stack.append(adj)
            if current_region: regions.append(current_region)
        return regions

    def createTerritories(self, land_regions_list):
        self.contiguousTerritoryIDs, self.territories_by_id, self.all_territories_for_unpickling, self._temp_contiguous_territories_objs = [], {}, [], []
        tid_counter = 0
        for region in land_regions_list:
            if not region: continue
            centers = np.array([t.center for t in region])
            n_clusters = min(max(1, math.ceil(len(region) / self.territorySize)), len(region))
            kmeans = KMeans(n_clusters=n_clusters, random_state=random.randint(0, 10000), n_init='auto', init='k-means++')
            labels = kmeans.fit_predict(centers)
            grouped_tiles = [[] for _ in range(n_clusters)]
            for i, t in enumerate(region): grouped_tiles[labels[i]].append(t)
            r_ids, r_objs = [], []
            for i in range(n_clusters):
                if grouped_tiles[i]:
                    cx, cy = sum(t.x for t in grouped_tiles[i]) / len(grouped_tiles[i]), sum(t.y for t in grouped_tiles[i]) / len(grouped_tiles[i])
                    terr = Territory(self.screenWidth, self.screenHeight, [cx, cy], grouped_tiles[i], self.allWaterTiles, self.cols, self.resource_info, self.structure_info)
                    terr.id, tid_counter = tid_counter, tid_counter + 1
                    r_ids.append(terr.id)
                    r_objs.append(terr)
                    self.territories_by_id[terr.id] = terr
                    for t_in_terr in grouped_tiles[i]: t_in_terr.territory_id = terr.id
            if r_ids:
                self.contiguousTerritoryIDs.append(r_ids)
                self._temp_contiguous_territories_objs.append(r_objs)
                self.all_territories_for_unpickling.extend(r_objs)

    def connectTerritoryHarbors(self):
        self.allHarbors = [h for terr in self.territories_by_id.values() for h in terr.harbors]
        if not self.allHarbors:
            print("  No harbors to connect.")
            return 0
        hid_counter = 0
        self.harbors_by_id = {}
        for h in self.allHarbors: h.harbor_id, self.harbors_by_id[hid_counter], hid_counter = hid_counter, h, hid_counter + 1
        print(f"  Assigned IDs to {len(self.allHarbors)} harbors.")
        harbors_by_ocean, unconnected = {}, 0
        for h in self.allHarbors:
            found_ocean_for_harbor = False
            for adj_t in h.tile.adjacent:
                if adj_t in self._ocean_id_map:
                    harbors_by_ocean.setdefault(self._ocean_id_map[adj_t], []).append(h)
                    found_ocean_for_harbor = True
                    break
            if not found_ocean_for_harbor: unconnected += 1
        if unconnected: print(f"  Warning: {unconnected} harbors have no direct water tile connection.")
        print(f"  Harbors grouped into {len(harbors_by_ocean)} oceans for pathfinding.")
        routes_found = 0
        for oid, h_list in harbors_by_ocean.items():
            if len(h_list) < 2: continue
            water_set = self._ocean_water.get(oid)
            if not water_set:
                print(f"  Warning: Could not find water tile set for ocean ID {oid}.")
                continue
            for i, src_h in enumerate(h_list):
                others = h_list[i + 1:]
                if not others or not hasattr(src_h, 'generateAllRoutes'): continue
                routes_found += src_h.generateAllRoutes(others, water_set)
        print(f"  Harbor connection complete. Found {routes_found} connected harbor pairs.")
        return hid_counter

    @staticmethod
    def _get_chunk_bounds(cx, cy, cs):
        return cx * cs, cy * cs, (cx + 1) * cs, (cy + 1) * cs

    @staticmethod
    def _is_chunk_fully_within_radius(crel, rcen, rad, cs):
        min_x, min_y, max_x, max_y = crel[0] * cs, crel[1] * cs, (crel[0] + 1) * cs, (crel[1] + 1) * cs
        return all(distance(c, rcen) < (rad * 0.3) for c in [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)])

    @staticmethod
    def _does_chunk_intersect_radius(crel, rcen, rad, cs):
        cmin_x, cmin_y, cmax_x, cmax_y = crel[0] * cs, crel[1] * cs, (crel[0] + 1) * cs, (crel[1] + 1) * cs
        cx, cy = max(cmin_x, min(rcen[0], cmax_x)), max(cmin_y, min(rcen[1], cmax_y))
        return ((rcen[0] - cx) ** 2 + (rcen[1] - cy) ** 2) < (rad * rad)

    def _precompute_cloud_visibility(self):
        patterns, max_r = {}, (self.precomp_radii[-1] if self.precomp_radii else 0)
        max_dist = math.ceil((max_r + 1.5 * self.PRECOMP_CHUNK_SIZE) / self.PRECOMP_CHUNK_SIZE)
        center_rel = (0.5 * self.PRECOMP_CHUNK_SIZE, 0.5 * self.PRECOMP_CHUNK_SIZE)
        for r_idx in reversed(range(len(self.precomp_radii))):
            rad = self.precomp_radii[r_idx]
            internal, edge, candidates = set(), set(), set()
            if r_idx == len(self.precomp_radii) - 1:
                for drx in range(-max_dist, max_dist + 1):
                    for dry in range(-max_dist, max_dist + 1):
                        dc_sq = (((drx + 0.5) * self.PRECOMP_CHUNK_SIZE - center_rel[0]) ** 2 + ((dry + 0.5) * self.PRECOMP_CHUNK_SIZE - center_rel[1]) ** 2)
                        if dc_sq <= (rad + self.PRECOMP_CHUNK_SIZE * 1.5) ** 2: candidates.add((drx, dry))
            else:
                if (r_idx + 1) in patterns: candidates.update(patterns[r_idx + 1]['internal_rel'], patterns[r_idx + 1]['edge_rel'])
                candidates.add((0, 0))
            for rcx, rcy in candidates:
                if self._is_chunk_fully_within_radius((rcx, rcy), center_rel, rad, self.PRECOMP_CHUNK_SIZE):
                    internal.add((rcx, rcy))
                elif self._does_chunk_intersect_radius((rcx, rcy), center_rel, rad, self.PRECOMP_CHUNK_SIZE):
                    edge.add((rcx, rcy))
            patterns[r_idx] = {'internal_rel': internal, 'edge_rel': edge}
        self.precomputed_cloud_info = {}
        for acx in range(self.precomp_grid_width):
            for acy in range(self.precomp_grid_height):
                for r_idx in range(len(self.precomp_radii)):
                    i_abs, e_abs = [], []
                    if r_idx in patterns:
                        pat = patterns[r_idx]
                        for rcx, rcy in pat['internal_rel']:
                            ax, ay = acx + rcx, acy + rcy
                            if 0 <= ax < self.precomp_grid_width and 0 <= ay < self.precomp_grid_height: i_abs.append((ax, ay))
                        for rcx, rcy in pat['edge_rel']:
                            ax, ay = acx + rcx, acy + rcy
                            if 0 <= ax < self.precomp_grid_width and 0 <= ay < self.precomp_grid_height: e_abs.append((ax, ay))
                    self.precomputed_cloud_info[(acx, acy, r_idx)] = {'internal_chunks': i_abs, 'edge_chunks': e_abs}
        print(f"  Cloud precomputation complete. {len(self.precomputed_cloud_info)} entries.")

    @staticmethod
    def _update_tile_opacity_for_source(tile, revealCenterCoords, actual_radius, is_internal_chunk, is_edge_chunk, debug_cloud_chunks):
        """Helper to calculate target opacity for a single tile from a single source."""
        opacity_from_this_source = 1.0  # Default to opaque for this source

        if is_internal_chunk:
            opacity_from_this_source = 0.0
            if debug_cloud_chunks: tile.debug_cloud_color_override = (0, 255, 0)  # GREEN
        elif is_edge_chunk:
            d = distance(revealCenterCoords, tile.center)
            if d < actual_radius:
                opacity_from_this_source = min(1.0, max(0.0, (d / actual_radius) ** 0.9))
            if debug_cloud_chunks:
                # Preserve GREEN if it was already set by an internal chunk check from another source
                if getattr(tile, 'debug_cloud_color_override', None) != (0, 255, 0):
                    tile.debug_cloud_color_override = (255, 165, 0)  # ORANGE

        tile.target_opacity_this_frame = min(tile.target_opacity_this_frame, opacity_from_this_source)

    def _apply_chunk_based_fov(self, revealCenterCoords, desiredRevealRadius, debug_cloud_chunks):
        center_pcx = int(revealCenterCoords[0] // self.PRECOMP_CHUNK_SIZE)
        center_pcy = int(revealCenterCoords[1] // self.PRECOMP_CHUNK_SIZE)
        actual_radius, radius_idx = self.precomp_radii[-1], len(self.precomp_radii) - 1
        for i, r_val in enumerate(self.precomp_radii):
            if desiredRevealRadius <= r_val:
                actual_radius, radius_idx = r_val, i
                break

        precomp_entry = self.precomputed_cloud_info.get((center_pcx, center_pcy, radius_idx))
        if not precomp_entry: return

        # Process internal chunks
        for chunk_coord_tuple in precomp_entry['internal_chunks']:
            for tile in self.precomp_chunk_grid_tiles.get(chunk_coord_tuple, []):
                self._update_tile_opacity_for_source(tile, revealCenterCoords, actual_radius, True, False, debug_cloud_chunks)

        # Process edge chunks
        for chunk_coord_tuple in precomp_entry['edge_chunks']:
            for tile in self.precomp_chunk_grid_tiles.get(chunk_coord_tuple, []):
                self._update_tile_opacity_for_source(tile, revealCenterCoords, actual_radius, False, True, debug_cloud_chunks)

        # Explicitly process center chunk tiles if not covered by precomp (e.g., small radii)
        # This ensures the immediate vicinity of the source is always checked.
        # Most tiles here would already be processed if in internal/edge of precomp.
        center_chunk_tiles = self.precomp_chunk_grid_tiles.get((center_pcx, center_pcy), [])
        for tile in center_chunk_tiles:
            # Re-evaluate, could be internal or edge depending on exact position within center chunk
            is_internal_to_radius = distance(revealCenterCoords, tile.center) < actual_radius * 0.1  # Heuristic for "very close"
            is_edge_to_radius = not is_internal_to_radius and distance(revealCenterCoords, tile.center) < actual_radius
            self._update_tile_opacity_for_source(tile, revealCenterCoords, actual_radius, is_internal_to_radius, is_edge_to_radius, debug_cloud_chunks)

    def _finalize_all_tile_opacities(self, dt, debug_cloud_chunks):
        REVEAL_SMOOTHING_RATE = 0.6
        OPAQUE_SMOOTHING_RATE = 0.75  # Opacity units per second
        opaque_smoothing_per_frame = OPAQUE_SMOOTHING_RATE * dt

        for tile in self.tiles:  # Iterate ALL tiles
            current_op = tile.cloudOpacity
            # tile.target_opacity_this_frame was set by _apply_chunk_based_fov or defaulted to 1.0
            final_target_op = tile.target_opacity_this_frame

            if final_target_op < current_op:  # Revealing
                tile.cloudOpacity += (final_target_op - current_op) * REVEAL_SMOOTHING_RATE
            elif final_target_op > current_op:  # Fading back
                tile.cloudOpacity += opaque_smoothing_per_frame
                if tile.cloudOpacity > final_target_op:
                    tile.cloudOpacity = final_target_op

            tile.cloudOpacity = min(1.0, max(0.0, tile.cloudOpacity))

            # Set debug color for tiles that are fading back (not directly under a source's green/orange)
            if debug_cloud_chunks and tile.target_opacity_this_frame == 1.0 and tile.cloudOpacity < 0.999:
                # Check if it wasn't already colored green or orange by a source this frame
                if getattr(tile, 'debug_cloud_color_override', None) not in [(0, 255, 0), (255, 165, 0)]:
                    tile.debug_cloud_color_override = (0, 0, 255)  # BLUE

    def draw2InternalScreen(self):
        if self.surf is None or self.debugOverlay is None: return
        base_color = self.cols.dark if hasattr(self.cols, 'dark') else (20, 20, 20)
        self.surf.fill(base_color)
        for tile in self.tiles: tile.draw(self.surf)
        self.debugOverlay.fill((0, 0, 0, 0))
        for tile in self.tiles:
            if tile.territory and hasattr(tile.territory, 'territoryCol'):
                d_col = tile.territory.territoryCol
                if isinstance(d_col, (list, tuple)) and len(d_col) in [3, 4]: pygame.draw.polygon(self.debugOverlay, d_col, tile.hex)
        for id_list in self.contiguousTerritoryIDs:
            for tid in id_list:
                terr = self.territories_by_id.get(tid)
                if terr and hasattr(terr, 'drawInternalTerritoryBaseline'): terr.drawInternalTerritoryBaseline(self.surf, self.debugOverlay)
            for tid in id_list:
                terr = self.territories_by_id.get(tid)
                if terr and hasattr(terr, 'drawInternalStructures'): terr.drawInternalStructures(self.surf)

    def draw(self, s, mx, my, showArrows=False, showDebugOverlay=False, showWaterLand=False, showDebugRoutes=False):
        if self.playersSurf is None: self.playersSurf = pygame.Surface((self.screenWidth, self.screenHeight), pygame.SRCALPHA)
        self.playersSurf.fill((0, 0, 0, 0))
        if self.surf is None or self.territorySurf is None:
            print("Error: Base map surfaces not initialized.")
            return
        s.blit(self.surf, (0, 0))
        if showDebugOverlay and self.debugOverlay: s.blit(self.debugOverlay, (0, 0))
        self.territorySurf.fill((0, 0, 0, 0))
        if showArrows:
            arrow_c = getattr(self.cols, 'debugRed', (255, 0, 0))
            for tile in self.tiles: tile.drawArrows(self.territorySurf, arrow_c)
        if showWaterLand and self.font:
            text_c = getattr(self.cols, 'dark', (0, 0, 0))
            for tile in self.tiles: tile.showWaterLand(self.territorySurf, self.font, text_c)
        for id_list in self.contiguousTerritoryIDs:
            for tid in id_list:
                terr = self.territories_by_id.get(tid)
                if terr and hasattr(terr, 'drawCurrent'): terr.drawCurrent(self.territorySurf, mx, my, showDebugRoutes)
        s.blit(self.territorySurf, (0, 0))

    def drawClouds(self, s, mx, my, mouseSize, playerObj, dt, debug_cloud_chunks=False):
        # 1. Reset target_opacity_this_frame for ALL tiles and debug colors
        for tile in self.tiles:
            tile.target_opacity_this_frame = 1.0  # Default to opaque for this frame
            if debug_cloud_chunks:
                tile.debug_cloud_color_override = None

        # 2. Apply FoV from all sources, updating tile.target_opacity_this_frame
        self._apply_chunk_based_fov((mx, my), 250 * mouseSize, debug_cloud_chunks=debug_cloud_chunks)  # Mouse
        for ship in playerObj.ships:
            if hasattr(ship, 'pos') and ship.pos is not None and hasattr(ship, 'currentVision'):
                reveal_radius_pixels = ship.currentVision * self.size
                self._apply_chunk_based_fov(ship.pos, reveal_radius_pixels, debug_cloud_chunks=debug_cloud_chunks)

        # 3. Finalize actual cloudOpacity for all tiles based on their target_opacity_this_frame and smoothing
        self._finalize_all_tile_opacities(dt, debug_cloud_chunks)

        # 4. Draw the cloud layer by iterating ALL tiles
        self.cloudSurf.fill((0, 0, 0, 0))
        for tile in self.tiles:
            if tile.cloudOpacity > 0.001:  # Only draw if not fully transparent
                alpha = int(tile.cloudOpacity * 255)

                color_rgb = tile.cloudCol[:3] if isinstance(tile.cloudCol, (list, tuple)) and len(tile.cloudCol) >= 3 else (50, 50, 50)
                if debug_cloud_chunks and tile.debug_cloud_color_override:
                    color_rgb = tile.debug_cloud_color_override

                final_color_rgba = tuple(color_rgb) + (alpha,)
                pygame.draw.polygon(self.cloudSurf, final_color_rgba, tile.hex)
        s.blit(self.cloudSurf, (0, 0))
