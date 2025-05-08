import pygame
import math
import random
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from text import drawText
from calcs import distance, ang, normalize_angle, draw_arrow, linearGradient, normalize
from territory import Territory

try:
    from shapely.geometry import Polygon as ShapelyPolygon

    SHAPELY_AVAILABLE_FOR_HINTS = True
except ImportError:
    SHAPELY_AVAILABLE_FOR_HINTS = False


class Hex:
    def __init__(self, grid_x, grid_y, x, y, size, tile_id, col=(0, 0, 0)):
        self.tile_id = tile_id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = x
        self.y = y
        self.size = size
        self.col = col
        self.center = [self.x, self.y]
        self.floatHexVertices = [(self.x + self.size * math.cos(math.pi / 3 * angle), self.y + self.size * math.sin(math.pi / 3 * angle)) for angle in range(6)]
        self.hex = [(int(round(p[0])), int(round(p[1]))) for p in self.floatHexVertices]

        self.adjacent_tile_ids = []  # Populated before pickling
        self.territory_id = -1
        self.adjacent = []  # Object references used temporarily during generation
        self.territory = None
        self.region = None
        self.mountainRegion = None
        self.regionCol = None

        self.waterLand = random.random()
        self.mountainous = random.random()
        self.isLand = False
        self.isMountain = False
        self.isCoast = False

        self.connectedOceanID = -1

    def prepare_for_pickling(self):
        # Create adjacent_tile_ids before nullifying adjacent object list
        self.adjacent_tile_ids = [adj.tile_id for adj in self.adjacent if hasattr(adj, 'tile_id')]

        self.adjacent = []  # Nullify object list for pickling
        self.territory = None
        self.region = None
        self.mountainRegion = None

    def draw(self, s):
        draw_col = self.col
        pygame.draw.polygon(s, draw_col, self.hex)

    def drawArrows(self, s, color):
        if not self.adjacent:
            return
        arrow_col = tuple(color) if color else (255, 0, 0)
        for adj in self.adjacent:
            try:
                angle = normalize_angle(ang((self.x, self.y), (adj.x, adj.y)))
                dist = distance((self.x, self.y), (adj.x, adj.y))
                factor = 0.35
                draw_arrow(s, (self.x, self.y), (self.x + dist * factor * math.cos(angle), self.y + dist * factor * math.sin(angle)), arrow_col, 2, 5, 25)
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
        self.all_territories_for_unpickling = []  # Temp flat list for pickling
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        self.shifter = [0, 0]
        self.shifterMomentum = [0, 0]
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

        self._temp_contiguous_territories_objs = None

        print("\nGenerating tiles...")
        self.generateTiles()
        self._link_adjacent_objects()  # Link OBJECTS first for generation cycle

        print("Running generation cycles...")
        for _ in range(50):
            self.generationCycle()  # Uses Hex.adjacent object list

        print("Finding contiguous regions...")
        landRegionsRaw = self.findContiguousRegions([t for t in self.tiles if t.waterLand >= self.waterThreshold])
        self.setTileCols()

        print("Indexing Oceans...")
        self.indexOceans()  # Uses Hex.adjacent object list

        print("Assigning coast land to oceans...")
        self.assignCoastTiles()

        print("Creating territories...")
        self.createTerritories(landRegionsRaw)  # Uses object lists

        print("Connecting Harbors...")
        self.allHarbors = []  # List of Harbor objects
        self.connectTerritoryHarbors()  # Uses Hex.adjacent objects internally

        print("TileHandler Worker Initialization Complete.")

    def generationCycle(self):
        """Smooths waterLand and mountainous values based on adjacent Hex objects."""
        waterLandShifts = []
        mountainShifts = []
        for tile in self.tiles:
            if tile.adjacent:
                waterLandShifts.append(sum(adj.waterLand for adj in tile.adjacent) / len(tile.adjacent))
                mountainShifts.append(sum(adj.mountainous for adj in tile.adjacent) / len(tile.adjacent))
            else:
                waterLandShifts.append(tile.waterLand)
                mountainShifts.append(tile.mountainous)

        for i, tile in enumerate(self.tiles):
            tile.waterLand = max(0.0, min(1.0, tile.waterLand + (waterLandShifts[i] - tile.waterLand) / 2))
            tile.mountainous = max(0.0, min(1.0, tile.mountainous + (mountainShifts[i] - tile.mountainous) / 2))

    def prepare_for_pickling(self):
        print("  TileHandler: Preparing for pickling...")
        self.surf = None
        self.debugOverlay = None
        self.territorySurf = None
        self.font = None

        # Store territories flatly before clearing maps
        temp_territories = list(self.territories_by_id.values())
        self.all_territories_for_unpickling = temp_territories

        # Convert contiguous territory structure to IDs
        self.contiguousTerritoryIDs = []

        self.contiguousTerritoryIDs = [[terr.id for terr in terr_list] for terr_list in self._temp_contiguous_territories_objs]
        del self._temp_contiguous_territories_objs  # Clean up temp list

        # Clean individual tiles (creates adjacent_tile_ids, nullifies adjacent objs)
        for tile in self.tiles:
            tile.prepare_for_pickling()

        # Clean territories (nullifies surfaces, etc.)
        for territory in self.all_territories_for_unpickling:  # Use flat list
            territory.prepare_for_pickling()
            # Clean harbors within territories
            for harbor in territory.harbors:
                harbor.prepare_for_pickling()

        # Clear large intermediate data structures
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        # Clear object maps (will be rebuilt)
        self.territories_by_id = {}
        self.harbors_by_id = {}
        # Keep self.tiles and self.contiguousTerritoryIDs (list of lists of IDs)

        print("  TileHandler: Pickling preparation complete.")

    def initialize_graphics_and_external_libs(self, fonts_dict):
        print("  TileHandler: Initializing graphics and external libs...")
        self.surf = pygame.Surface((self.screenWidth, self.screenHeight)).convert_alpha()
        self.debugOverlay = pygame.Surface((self.screenWidth, self.screenHeight)).convert_alpha()
        self.territorySurf = pygame.Surface((self.screenWidth, self.screenHeight)).convert_alpha()
        self.surf.fill(self.cols.oceanBlue)
        self.debugOverlay.fill((0, 0, 0, 0))
        self.territorySurf.fill((0, 0, 0, 0))

        if self.font_name and self.font_name in fonts_dict:
            self.font = fonts_dict[self.font_name]
        elif self.font_name:
            print(f"  TileHandler Warning: Font name '{self.font_name}' not found.")

        # Rebuild tiles_by_id map
        self.tiles_by_id = {tile.tile_id: tile for tile in self.tiles if hasattr(tile, 'tile_id')}
        print(f"  TileHandler: Rebuilt tiles_by_id map ({len(self.tiles_by_id)} entries).")

        # Rebuild territory/harbor maps from the flat pickled list
        if not hasattr(self, 'all_territories_for_unpickling') or not self.all_territories_for_unpickling:
            print("  Error: Cannot find pickled territory objects for map rebuilding.")
            # Ensure all_territories_for_unpickling is correctly populated and pickled.
            return  # Cannot proceed safely

        print(f"  TileHandler: Rebuilding territory/harbor maps from {len(self.all_territories_for_unpickling)} territories...")
        self.territories_by_id = {}
        self.harbors_by_id = {}
        all_harbors = []
        for territory in self.all_territories_for_unpickling:
            if hasattr(territory, 'id'):
                self.territories_by_id[territory.id] = territory
                if hasattr(territory, 'harbors'):
                    all_harbors.extend(territory.harbors)

        # Assign harbor IDs consistently
        harbor_id_counter = 0
        for harbor in all_harbors:
            if hasattr(harbor, 'harbor_id'):
                if harbor.harbor_id != -1:  # Use existing valid ID
                    self.harbors_by_id[harbor.harbor_id] = harbor
                    # Ensure counter stays ahead of existing IDs if loading mixed saves
                    harbor_id_counter = max(harbor_id_counter, harbor.harbor_id + 1)
                else:  # Assign new ID if it wasn't set correctly (-1)
                    harbor.harbor_id = harbor_id_counter
                    self.harbors_by_id[harbor_id_counter] = harbor
                    harbor_id_counter += 1
            else:  # Assign new ID if attribute missing
                harbor.harbor_id = harbor_id_counter
                self.harbors_by_id[harbor_id_counter] = harbor
                harbor_id_counter += 1
        # self.all_territories_for_unpickling can be cleared or kept as self.all_territories if useful
        print(f"  TileHandler: Rebuilt {len(self.territories_by_id)} territories, {len(self.harbors_by_id)} harbors.")

        # Re-link Hex -> Territory object references
        print("  TileHandler: Re-linking Hex -> Territory references...")
        restored_terr_links = 0
        for tile in self.tiles:
            tile.territory = None  # Reset first
            if hasattr(tile, 'territory_id') and tile.territory_id != -1:
                territory_obj = self.territories_by_id.get(tile.territory_id)
                if territory_obj:
                    tile.territory = territory_obj
                    restored_terr_links += 1
        print(f"  TileHandler: Restored {restored_terr_links} Hex -> Territory links.")

        # Re-link Hex -> Adjacent Hex object references using IDs
        print("  TileHandler: Re-linking Hex -> Adjacent Hex references...")
        restored_adj_links = 0
        total_adj = 0
        for tile in self.tiles:
            tile.adjacent = []  # Clear list first
            if hasattr(tile, 'adjacent_tile_ids'):
                total_adj += len(tile.adjacent_tile_ids)
                for neighbor_id in tile.adjacent_tile_ids:
                    neighbor_obj = self.tiles_by_id.get(neighbor_id)
                    if neighbor_obj:
                        tile.adjacent.append(neighbor_obj)
                        restored_adj_links += 1
        print(f"  TileHandler: Restored {restored_adj_links}/{total_adj} Hex -> Adjacent links.")

        # Initialize territory graphics (polygons, etc.)
        print("  TileHandler: Initializing territory graphics...")
        for territory in self.territories_by_id.values():
            if hasattr(territory, 'initialize_graphics_and_external_libs'):
                territory.initialize_graphics_and_external_libs()

        # Initialize Harbor routes (rebuilds internal object maps from ID data)
        print("  TileHandler: Initializing harbor routes...")
        for harbor in self.harbors_by_id.values():
            if hasattr(harbor, 'initialize_graphics_and_external_libs'):
                harbor.initialize_graphics_and_external_libs(self.tiles_by_id, self.harbors_by_id)

        # Update territory reachableHarbors map AFTER harbor routes are initialized
        print("  TileHandler: Updating territory reachableHarbors...")
        for territory in self.territories_by_id.values():
            if hasattr(territory, 'update_reachable_harbors'):
                territory.update_reachable_harbors()

        print("  TileHandler: Drawing static elements to internal surfaces...")
        self.draw2InternalScreen()
        print("  TileHandler: Graphics and external libs initialized.")

    def generateTiles(self):
        tile_id = 0
        for x in range(self.gridSizeX):
            for y in range(self.gridSizeY):
                x_pos = self.horizontal_distance * x + self.size * (self.borderSize + 0.5)
                y_pos = self.vertical_distance * y + self.size * (self.borderSize - 0.5)
                if x % 2 == 1:
                    y_pos += self.vertical_distance / 2
                hex_obj = Hex(x, y, x_pos, y_pos, self.size, tile_id)
                self.tiles.append(hex_obj)
                self.tiles_by_id[tile_id] = hex_obj
                tile_id += 1

    def _link_adjacent_objects(self):
        """Populates Hex.adjacent object list based on grid coords. Used during generation."""
        print("  TileHandler: Linking adjacent tile objects for generation...")
        grid_obj_map = {(tile.grid_x, tile.grid_y): tile for tile in self.tiles}
        for tile in self.tiles:
            tile.adjacent = []  # Ensure it's empty first
            if tile.grid_x % 2 == 0:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, -1)]
            else:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)]
            for dx, dy in offsets:
                neighbor_coord = (tile.grid_x + dx, tile.grid_y + dy)
                neighbor_obj = grid_obj_map.get(neighbor_coord)
                if neighbor_obj:
                    tile.adjacent.append(neighbor_obj)  # Link OBJECT reference

    def getTileAtPosition(self, x, y):
        for tile in self.tiles:
            if math.dist((x, y), (tile.x, tile.y)) <= self.size:
                return tile
        return None

    def setTileCols(self):
        # Determine tile types
        for tile in self.tiles:
            tile.isLand = (tile.waterLand >= self.waterThreshold)
            tile.isMountain = (tile.isLand and tile.mountainous >= self.mountainThreshold)

        # Calculate value bounds for coloring
        water_values = [t.waterLand for t in self.tiles if not t.isLand]
        land_values = [t.waterLand for t in self.tiles if t.isLand and not t.isMountain]
        mountain_values = [t.mountainous for t in self.tiles if t.isMountain]

        waterBounds = [min(water_values), max(water_values)] if water_values else [0.0, self.waterThreshold]
        landBounds = [min(land_values), max(land_values)] if land_values else [self.waterThreshold, 1.0]
        mountainBounds = [min(mountain_values), max(mountain_values)] if mountain_values else [self.mountainThreshold, 1.0]

        waterColoringNoise = 0.0035
        landColoringNoise = 0.0035
        mountainColoringNoise = 0.005

        # Assign Colors and populate type lists
        self.allWaterTiles = []
        self.allLandTiles = []
        self.allCoastalTiles = []
        weightWaterDistribution = lambda x: (x ** 2) / 2 + (1 - (1 - x) ** 2) ** 10 / 2
        weightLandDistribution = lambda x: (1 - 2 ** (-3 * x)) * 8 / 7

        for tile in self.tiles:
            tile.isCoast = False  # Reset coast flag
            if not tile.isLand:
                noise = random.uniform(-waterColoringNoise, waterColoringNoise)
                normalized_val = normalize(tile.waterLand + noise, waterBounds[0], waterBounds[1], True)
                tile.col = linearGradient([self.cols.oceanBlue, self.cols.oceanGreen, self.cols.lightOceanGreen, self.cols.oceanFoam], weightWaterDistribution(normalized_val))
                self.allWaterTiles.append(tile)
            elif tile.isMountain:
                noise = random.uniform(-mountainColoringNoise, mountainColoringNoise)
                normalized_val = normalize(tile.mountainous + noise, mountainBounds[0], mountainBounds[1], True)
                tile.col = linearGradient([self.cols.mountainBlue, self.cols.darkMountainBlue], normalized_val)
                self.allLandTiles.append(tile)  # Mountains are also land
            else:  # Just land
                noise = random.uniform(-landColoringNoise, landColoringNoise)
                normalized_val = normalize(tile.waterLand + noise, landBounds[0], landBounds[1], True)
                tile.col = linearGradient([self.cols.oliveGreen, self.cols.darkOliveGreen], weightLandDistribution(normalized_val))
                self.allLandTiles.append(tile)

        # Identify coastal tiles (land adjacent to water)
        allWaterTilesSet = set(self.allWaterTiles)
        for tile in self.allLandTiles:
            # Check adjacent list (needs to be the object list here)
            if any(adj in allWaterTilesSet for adj in tile.adjacent):
                self.allCoastalTiles.append(tile)
                tile.isCoast = True

    def indexOceans(self):
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        unvisited = set(self.allWaterTiles)
        ocean_id = 0
        while unvisited:
            start_tile = unvisited.pop()
            current_ocean_set = {start_tile}
            queue = deque([start_tile])
            self._ocean_id_map[start_tile] = ocean_id
            while queue:
                tile = queue.popleft()
                for neighbor in tile.adjacent:  # Uses object list during generation
                    if neighbor in unvisited:
                        neighbor.connectedOceanID = ocean_id
                        unvisited.remove(neighbor)
                        current_ocean_set.add(neighbor)
                        self._ocean_id_map[neighbor] = ocean_id
                        queue.append(neighbor)
            if current_ocean_set:
                self.oceanTiles[ocean_id] = current_ocean_set
                self._ocean_water[ocean_id] = current_ocean_set
                ocean_id += 1

        print(f"  Indexed {len(self.oceanTiles)} distinct water bodies.")

    def assignCoastTiles(self):
        allWaterTilesSet = set(self.allWaterTiles)
        for coastTile in self.allCoastalTiles:
            coastTile.connectedOceanID = max([adjWater.connectedOceanID for adjWater in coastTile.adjacent if adjWater in allWaterTilesSet])
        print(f"Assigned {len(self.allCoastalTiles)} coastal tiles to their oceans")

    @staticmethod
    def findContiguousRegions(tiles_to_check):
        visited = set()
        regions = []
        tilesSet = set(tiles_to_check)
        for tile in tiles_to_check:
            if tile in visited:
                continue
            region = []
            stack = [tile]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                region.append(current)
                for adj in current.adjacent:  # Uses object list during generation
                    if adj in tilesSet and adj not in visited:
                        stack.append(adj)
            if region:
                regions.append(region)
        return regions

    def createTerritories(self, land_regions_list):
        self.contiguousTerritoryIDs = []
        self.territories_by_id = {}
        self.all_territories_for_unpickling = []
        self._temp_contiguous_territories_objs = []  # Store object structure temporarily for pickling prep

        territory_id_counter = 0
        for region in land_regions_list:
            if not region:
                continue

            tile_centers = np.array([tile.center for tile in region])
            num_tiles = len(region)
            # Ensure n_clusters is at least 1 and not more than the number of samples
            n_clusters = min(max(1, math.ceil(num_tiles / self.territorySize)), num_tiles)

            # Use k-means++ initialization and suppress future warning about n_init
            kmeans = KMeans(n_clusters=n_clusters, random_state=random.randint(0, 10000), n_init='auto', init='k-means++')
            labels = kmeans.fit_predict(tile_centers)
            calculated_centers = kmeans.cluster_centers_

            territory_tiles_grouped = [[] for _ in range(n_clusters)]
            for i, tile in enumerate(region):
                territory_tiles_grouped[labels[i]].append(tile)

            region_territory_ids = []
            region_territories_objs = []
            for i in range(n_clusters):
                if territory_tiles_grouped[i]:
                    center_pos = calculated_centers[i].tolist()
                    # Pass required dependencies like resource_info, structure_info
                    territory = Territory(self.screenWidth, self.screenHeight, center_pos, territory_tiles_grouped[i], self.allWaterTiles, self.cols, resource_info=self.resource_info, structure_info=self.structure_info)
                    current_id = territory_id_counter
                    territory.id = current_id
                    territory_id_counter += 1
                    region_territory_ids.append(current_id)
                    region_territories_objs.append(territory)
                    self.territories_by_id[current_id] = territory
                    # Link tiles back to their territory ID
                    for tile in territory_tiles_grouped[i]:
                        tile.territory_id = current_id

            if region_territory_ids:
                self.contiguousTerritoryIDs.append(region_territory_ids)  # Store ID list (kept after pickling)
                self._temp_contiguous_territories_objs.append(region_territories_objs)  # Store object list temporarily
                self.all_territories_for_unpickling.extend(region_territories_objs)  # Add to flat list for pickling

    def connectTerritoryHarbors(self):
        all_territories = self.territories_by_id.values()
        self.allHarbors = [h for terr in all_territories for h in terr.harbors]
        if not self.allHarbors:
            return 0

        # Assign IDs to harbors before grouping them
        harbor_id = 0
        self.harbors_by_id = {}
        for h in self.allHarbors:
            h.harbor_id = harbor_id
            self.harbors_by_id[harbor_id] = h
            harbor_id += 1
        print(f"  Assigned IDs to {len(self.allHarbors)} harbors.")

        # Group harbors by the ocean ID they are connected to
        harbors_by_ocean = {}
        unconnected_harbors = 0
        for h in self.allHarbors:
            found_ocean = False
            # Check adjacent tiles (must be object list during generation/connection)
            for w in h.tile.adjacent:
                if w in self._ocean_id_map:
                    oid = self._ocean_id_map[w]
                    harbors_by_ocean.setdefault(oid, []).append(h)
                    found_ocean = True
                    break  # Found the ocean connection
            if not found_ocean:
                unconnected_harbors += 1

        if unconnected_harbors > 0:
            print(f"  Warning: {unconnected_harbors} harbors have no direct water tile connection.")
        print(f"  Harbors grouped into {len(harbors_by_ocean)} oceans for pathfinding.")

        # Generate routes between harbors within the same ocean
        total_routes_found = 0
        for oid, harbors_in_ocean in harbors_by_ocean.items():
            if len(harbors_in_ocean) < 2:
                continue  # Need at least two harbors to connect

            water_set_for_ocean = self._ocean_water.get(oid)
            if not water_set_for_ocean:
                print(f"  Warning: Could not find water tile set for ocean ID {oid}.")
                continue

            # Iterate through pairs of harbors in this ocean
            for i, src_harbor in enumerate(harbors_in_ocean):
                other_harbors = harbors_in_ocean[i + 1:]
                if not other_harbors:
                    continue  # No more pairs for this source harbor

                # Check if the Harbor object has the pathfinding method
                if hasattr(src_harbor, 'generateAllRoutes'):
                    routes = src_harbor.generateAllRoutes(other_harbors, water_set_for_ocean)
                    total_routes_found += routes
                else:
                    print(f"  Warning: Harbor object (ID {src_harbor.harbor_id}) lacks 'generateAllRoutes' method.")

        print(f"  Harbor connection complete. Found {total_routes_found} connected harbor pairs.")
        return harbor_id  # Return the next available ID

    def draw2InternalScreen(self):
        """Draws static elements (tiles, territory borders, resources) onto internal surfaces."""
        if self.surf is None or self.debugOverlay is None:
            print("  Warning: Attempting draw2InternalScreen before graphics initialization.")
            return

        # Draw base tiles to the main static surface
        self.surf.fill(self.cols.dark)  # Background color
        for tile in self.tiles:
            tile.draw(self.surf)

        # Draw static debug info (like territory color overlays) to the debug surface
        self.debugOverlay.fill((0, 0, 0, 0))  # Clear debug overlay
        for tile in self.tiles:
            # Use territory object link (restored in initialize_graphics_and_external_libs)
            if tile.territory and hasattr(tile.territory, 'territoryCol'):
                debug_col = tile.territory.territoryCol
                # Ensure color is a valid format for pygame
                if isinstance(debug_col, (list, tuple)) and len(debug_col) in [3, 4]:
                    pygame.draw.polygon(self.debugOverlay, debug_col, tile.hex)  # else: # Optional warning for invalid colors  #     print(f"Warning: territoryCol {debug_col} for territory {tile.territory.id} is not a valid color tuple.")

        # Call each territory's method to draw its static elements (borders, resources)
        # onto the appropriate surfaces (surf or debugOverlay)
        for terr_id_list in self.contiguousTerritoryIDs:  # Iterate using the ID structure
            for terr_id in terr_id_list:
                territory = self.territories_by_id.get(terr_id)  # Get object via ID
                if territory and hasattr(territory, 'drawInternal'):
                    # Pass the TileHandler's surfaces to the territory
                    territory.drawInternal(self.surf, self.debugOverlay)

    def draw(self, s, mx, my, click, showArrows=False, showDebugOverlay=False, showWaterLand=False):
        """Draws the current map state onto the target surface `s`."""
        if self.surf is None or self.territorySurf is None:
            # Avoid drawing if graphics aren't ready (e.g., before initialization)
            return

        # 1. Blit the pre-rendered base map surface (tiles, static territory elements)
        s.blit(self.surf, (0, 0))

        # 2. Blit the pre-rendered debug overlay surface if enabled
        if showDebugOverlay and self.debugOverlay:
            s.blit(self.debugOverlay, (0, 0))

        # 3. Draw dynamic elements onto the temporary territorySurf
        self.territorySurf.fill((0, 0, 0, 0))  # Clear the dynamic surface each frame

        if showArrows:
            arrow_color = getattr(self.cols, 'debugRed', (255, 0, 0))
            # Consider optimizations here if performance is an issue
            for tile in self.tiles:
                # Uses the adjacent object list restored during initialization
                tile.drawArrows(self.territorySurf, arrow_color)

        if showWaterLand and self.font:
            text_color = getattr(self.cols, 'dark', (0, 0, 0))
            # Consider optimizations here if performance is an issue
            for tile in self.tiles:
                tile.showWaterLand(self.territorySurf, self.font, text_color)

        # Draw dynamic territory elements (like hover effects)
        for terr_id_list in self.contiguousTerritoryIDs:
            for terr_id in terr_id_list:
                territory = self.territories_by_id.get(terr_id)
                # Call territory's method to draw dynamic effects (e.g., hover)
                territory.drawCurrent(self.territorySurf, mx, my, click)

        # 4. Blit the dynamic canvas (with arrows, text, hover effects) onto the main target surface 's'
        s.blit(self.territorySurf, (0, 0))
