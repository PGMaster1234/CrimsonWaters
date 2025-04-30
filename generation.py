import pygame
import math
import random
import time
from collections import deque
from text import drawText
from calcs import distance, ang, normalize_angle, draw_arrow, linearGradient, normalize, randomCol
from territory import Territory


class TileHandler:
    def __init__(self, width, height, size, cols, waterThreshold=0.51, mountainThreshold=0.51, territorySize=100, font=None):
        self.surf = pygame.Surface((width, height)).convert_alpha()
        self.debugOverlay = pygame.Surface((width, height)).convert_alpha()
        self.territorySurf = pygame.Surface((width, height)).convert_alpha()
        self.surf.fill(cols.dark)
        self.debugOverlay.fill(cols.dark)
        self.territorySurf.fill((0, 0, 0, 0))

        self.size = size
        self.tiles = []
        self.territorySize = territorySize
        self.contiguousTerritories = []
        self.font = font
        self.cols = cols
        self.shifter = [0, 0]
        self.shifterMomentum = [0, 0]
        self.waterThreshold = waterThreshold
        self.mountainThreshold = mountainThreshold

        self.borderSize = 4
        self.horizontal_distance = (3 / 2 * size)
        self.vertical_distance = (math.sqrt(3) * size)
        self.gridSizeX = int(width / self.horizontal_distance) - self.borderSize - 1
        self.gridSizeY = int(height / self.vertical_distance) - self.borderSize

        self.allLandTiles = []
        self.allWaterTiles = []
        self.allCoastalTiles = []

        t = time.time()
        print("")
        print("Generating tiles...")
        self.generateTiles()
        self.assignAdjacentTiles()
        for _ in range(50):
            self.generationCycle()

        print(f"{round(time.time() - t, 3)}s")
        print("")
        t = time.time()
        print("Finding contiguous regions...")
        self.landRegions = self.findContiguousRegions([t for t in self.tiles if t.waterLand >= self.waterThreshold])
        self.mountainRegions = self.findContiguousRegions([t for t in self.tiles if (t.waterLand >= self.waterThreshold and t.mountainous > self.mountainThreshold)])
        self.setRegionReferences()

        self.setTileCols()

        print(f"{round(time.time() - t, 3)}s")
        print("")
        t = time.time()
        print("Indexing Oceans...")
        self.oceanTiles = {}
        self._ocean_id_map = {}
        self._ocean_water = {}
        self.indexOceans()  # populates self._ocean_id_map

        print(f"{round(time.time() - t, 3)}s")
        print("")
        t = time.time()
        print("Creating territories...")
        self.createTerritories()

        print(f"{round(time.time() - t, 3)}s")
        print("")
        t = time.time()
        print("Connecting Harbors...")
        self.allHarbors = []
        self.connectTerritoryHarbors()

        print(f"{round(time.time() - t, 3)}s")
        print("")
        t = time.time()
        print("Drawing to surf...")
        self.draw2InternalScreen()

        print(f"{round(time.time() - t, 3)}s")

    def generateTiles(self):
        for x in range(self.gridSizeX):
            for y in range(self.gridSizeY):
                x_pos = self.horizontal_distance * x + self.size * self.borderSize
                y_pos = self.vertical_distance * y + self.size * self.borderSize
                if x % 2 == 1:
                    y_pos += self.vertical_distance / 2
                self.tiles.append(Hex(x, y, x_pos, y_pos, self.size))

    def assignAdjacentTiles(self):
        grid_dict = {(tile.grid_x, tile.grid_y): tile for tile in self.tiles}

        for tile in self.tiles:
            tile.adjacent = []

            if tile.grid_x % 2 == 0:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, -1)]
            else:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1)]

            for dx, dy in offsets:
                neighbor_grid_x = tile.grid_x + dx
                neighbor_grid_y = tile.grid_y + dy

                if (neighbor_grid_x, neighbor_grid_y) in grid_dict:
                    tile.adjacent.append(grid_dict[(neighbor_grid_x, neighbor_grid_y)])

    def getTileAtPosition(self, x, y):
        for tile in self.tiles:
            if math.isclose(tile.x, x, abs_tol=self.horizontal_distance / 2) and math.isclose(tile.y, y, abs_tol=self.vertical_distance / 2):
                return tile
        return None

    def generationCycle(self):
        waterLandShifts = [sum([adj.waterLand for adj in tile.adjacent]) / len(tile.adjacent) for tile in self.tiles]
        mountainShifts = [sum([adj.mountainous for adj in tile.adjacent]) / len(tile.adjacent) for tile in self.tiles]
        for i, tile in enumerate(self.tiles):
            tile.waterLand += (waterLandShifts[i] - tile.waterLand) / 2
            tile.mountainous += (mountainShifts[i] - tile.mountainous) / 2

    def setTileCols(self):
        waterTiles = [(tile.waterLand if tile.waterLand < self.waterThreshold else 0.5) for tile in self.tiles]
        landTiles = [(tile.waterLand if tile.waterLand >= self.waterThreshold else 0.5) for tile in self.tiles]
        mountainTiles = [(tile.mountainous if (tile.waterLand >= self.waterThreshold and tile.mountainous >= self.mountainThreshold) else 0.5) for tile in self.tiles]
        waterBounds = [min(waterTiles), max(waterTiles)]
        landBounds = [min(landTiles), max(landTiles)]
        mountainBounds = [min(mountainTiles), max(mountainTiles)]
        waterColoringNoise = 0.006
        landColoringNoise = 0.004
        mountainColoringNoise = 0.01

        for i, tile in enumerate(self.tiles):
            weightWaterDistribution = lambda x: (x ** 2) / 2 + (1 - (1 - x) ** 2) ** 10 / 2
            if tile.waterLand < self.waterThreshold:
                tile.col = linearGradient([self.cols.oceanBlue, self.cols.oceanGreen, self.cols.lightOceanGreen, self.cols.oceanFoam], weightWaterDistribution(normalize(tile.waterLand + random.uniform(-waterColoringNoise, waterColoringNoise), waterBounds[0], waterBounds[1], True)))
                self.allWaterTiles.append(tile)
            elif tile.waterLand >= self.waterThreshold:
                weightLandDistribution = lambda x: (1 - 2 ** (-3 * x)) * 8 / 7
                tile.isLand = True
                self.allLandTiles.append(tile)
                if tile.mountainous < self.mountainThreshold:
                    tile.col = linearGradient([self.cols.oliveGreen, self.cols.darkOliveGreen], weightLandDistribution(normalize(tile.waterLand + random.uniform(-landColoringNoise, landColoringNoise), landBounds[0], landBounds[1], True)))
                else:
                    tile.isMountain = True
                    tile.col = linearGradient([self.cols.mountainBlue, self.cols.darkMountainBlue], normalize(tile.mountainous + random.uniform(-mountainColoringNoise, mountainColoringNoise), mountainBounds[0], mountainBounds[1], True))
        for tile in self.tiles:
            if any(adj in self.allWaterTiles for adj in tile.adjacent):
                self.allCoastalTiles.append(tile)
                tile.isCoast = True

    def indexOceans(self):
        """Identifies separate bodies of water (oceans, lakes) using flood fill (BFS)."""
        self.oceanTiles.clear()
        self._ocean_id_map.clear()
        self._ocean_water.clear()
        unvisited = set(self.allWaterTiles)  # Work on a copy of the water tile set
        ocean_id = 0
        while unvisited:  # While there are water tiles not yet assigned to an ocean
            start_tile = unvisited.pop()  # Pick an arbitrary unvisited water tile
            current_ocean_set = {start_tile}  # Start a new ocean set
            queue = deque([start_tile])  # Initialize BFS queue
            self._ocean_id_map[start_tile] = ocean_id  # Assign ocean ID to the starting tile

            # Breadth-First Search
            while queue:
                tile = queue.popleft()
                for neighbor in tile.adjacent:
                    # Check if neighbor is water AND hasn't been visited (is in unvisited set)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)  # Mark as visited by removing from set
                        current_ocean_set.add(neighbor)  # Add to the current ocean
                        self._ocean_id_map[neighbor] = ocean_id  # Assign the same ocean ID
                        queue.append(neighbor)  # Add to queue for further exploration

            # Store the results for this ocean if it contains tiles
            if current_ocean_set:
                self.oceanTiles[ocean_id] = current_ocean_set
                self._ocean_water[ocean_id] = current_ocean_set  # Store set reference here too
                ocean_id += 1  # Increment ID for the next ocean
        print(f"  Indexed {len(self.oceanTiles)} distinct water bodies.")

    @staticmethod
    def findContiguousRegions(tiles):
        visited = set()
        regions = []

        for tile in tiles:
            if tile in visited:
                continue
            stack = [tile]
            region = []

            # DFS
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                region.append(current)
                for adj in current.adjacent:
                    if (adj in tiles) and (adj not in visited):
                        stack.append(adj)
            regions.append(region)
        print(f"Found {len(regions)} contiguous regions of sizes {[len(reg) for reg in regions]}")
        return regions

    def setRegionReferences(self):
        for region in self.landRegions:
            col = randomCol('g')
            for tile in region:
                tile.region = region
                tile.regionCol = col
        for region in self.mountainRegions:
            col = randomCol('r')
            for tile in region:
                tile.mountainRegion = region
                tile.regionCol = col

    @staticmethod
    def iterate(points, centers):
        # p = [x, y, distance, parent, tileReference]
        # c = [x, y, points, tileReferences]
        for p in points:
            p[2] = 1e9
            p[3] = None

        for c in centers:
            c[2] = []
            for p in points:
                d = distance((p[0], p[1]), (c[0], c[1]))
                if d < p[2]:
                    c[2].append(p)
                    if p[3] is not None:
                        p[3][2].remove(p)
                    p[3] = c
                    p[2] = d
        return points, centers

    @staticmethod
    def shiftCenters(centers):
        # p = [x, y, distance, parent, tileReference]
        # c = [x, y, points, tileReferences]
        for c in centers:
            if len(c[2]) > 0:
                total_x = sum(p[0] for p in c[2])
                total_y = sum(p[1] for p in c[2])
                c[0] = total_x / len(c[2])
                c[1] = total_y / len(c[2])
        return centers

    def createTerritories(self):
        # p = [x, y, distance, parent, tileReference]
        # c = [x, y, points, tileReferences]
        self.contiguousTerritories = []
        for region in self.landRegions:
            points = [[tile.center[0], tile.center[1], 1e9, None, tile] for tile in region]
            centers = [[p[0], p[1], [], []] for p in random.sample(points, math.ceil(len(region) / self.territorySize))]
            for _ in range(5):
                points, centers = self.iterate(points, centers)
                centers = self.shiftCenters(centers)
            for p in points:
                p[3][3].append(p[4])  # store tile reference
            self.contiguousTerritories.append([Territory([c[0], c[1]], c[3], self.allWaterTiles, self.cols) for c in centers])

    def connectTerritoryHarbors(self):
        """Finds and stores water routes between all harbors on the map."""
        # 1) Gather all harbors from all territories into a single list
        self.allHarbors = [h for terr_list in self.contiguousTerritories for terr in terr_list for h in terr.harbors]
        if not self.allHarbors:
            print("  No harbors found to connect.")
            return
        print(f"  Connecting {len(self.allHarbors)} harbors...")

        # 2) Group harbors by the ocean_id of their adjacent water tiles
        harbors_by_ocean = {}
        unconnected_harbors = 0
        for h in self.allHarbors:
            found_ocean = False
            # Check adjacent tiles to find which ocean this harbor borders
            for w in h.tile.adjacent:
                if w in self._ocean_id_map:  # Check if neighbor is mapped water
                    oid = self._ocean_id_map[w]
                    # Add harbor to the list for this ocean ID
                    harbors_by_ocean.setdefault(oid, []).append(h)
                    found_ocean = True
                    break  # Found one ocean, that's enough to group
            if not found_ocean:
                unconnected_harbors += 1
        if unconnected_harbors > 0:
            print(f"  Warning: {unconnected_harbors} harbors have no water connection.")
        print(f"  Harbors grouped into {len(harbors_by_ocean)} oceans for pathfinding.")

        # 3) For each ocean group, run pathfinding from each harbor to others in the same group
        total_routes_found = 0  # Count A->B routes
        processed_harbor_count = 0
        # Iterate through each ocean group
        for oid, harbors_in_ocean in harbors_by_ocean.items():
            # Skip if only one harbor (or none) in this ocean
            if len(harbors_in_ocean) < 2: continue

            # Get the set of water tiles corresponding to this ocean ID
            water_set_for_ocean = self._ocean_water.get(oid)
            if not water_set_for_ocean:
                print(f"Warning: No water tile set found for ocean {oid}, skipping connections.")
                continue

            # print(f"  Processing ocean {oid} with {len(harbors_in_ocean)} harbors...")
            # Run the multi-target search (generateAllRoutes) from each harbor in this ocean
            for src_harbor in harbors_in_ocean:
                processed_harbor_count += 1
                # Define the list of 'other' harbors in the same ocean
                other_harbors = [h for h in harbors_in_ocean if h is not src_harbor]
                # Calculate paths and store results directly in src_harbor.tradeRoutes
                src_harbor.generateAllRoutes(other_harbors, water_set_for_ocean)
                # Add the number of routes found *from* this harbor
                total_routes_found += len(src_harbor.tradeRoutes)  # Simple progress indicator:  # if processed_harbor_count % 50 == 0: print(f"    Processed {processed_harbor_count}/{len(self.allHarbors)} harbors...")

        # This count includes A->B and B->A, divide by 2 for unique pairs
        print(f"  Pathfinding complete. Found {total_routes_found // 2} connected harbor pairs ({total_routes_found} directed routes).")

        # 4) Populate each Territory's reachableHarbors dictionary for drawing/logic
        for terr_list in self.contiguousTerritories:
            for terr in terr_list:
                terr.reachableHarbors.clear()  # Ensure it's empty before populating
                # For each harbor physically located IN THIS territory...
                for harbor_in_terr in terr.harbors:
                    # ...get the list of other harbors it found routes TO (which are the keys of its tradeRoutes dict)
                    # The values in reachableHarbors are lists of *Harbor objects*
                    terr.reachableHarbors[harbor_in_terr] = list(harbor_in_terr.tradeRoutes.keys())

    def draw2InternalScreen(self):
        for tile in self.tiles:
            tile.draw(self.surf, False)
            tile.draw(self.debugOverlay, True)
        for contiguousTerritoryList in self.contiguousTerritories:
            for territory in contiguousTerritoryList:
                territory.draw(self.surf, self.debugOverlay)

    def draw(self, s, mx, my, showArrows=False, showDebugOverlay=False, showWaterLand=False):
        self.territorySurf.fill((0, 0, 0, 0))
        s.blit(self.surf, (0, 0))
        if showDebugOverlay:
            s.blit(self.debugOverlay, (0, 0))
        if showArrows:
            for tile in self.tiles:
                tile.drawArrows(s)
        if showWaterLand:
            for tile in self.tiles:
                tile.showWaterLand(s, self.font)
        for contiguousTerritoryList in self.contiguousTerritories:
            for territory in contiguousTerritoryList:
                territory.drawCurrent(self.territorySurf, mx, my)
        s.blit(self.territorySurf, (0, 0))


class Hex:
    def __init__(self, grid_x, grid_y, x, y, size, col=(0, 0, 0)):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = x
        self.y = y
        self.size = size
        self.col = col
        self.center = [self.x, self.y]
        self.hex = [[int(self.x + self.size * math.cos(math.pi / 3 * angle)), int(self.y + self.size * math.sin(math.pi / 3 * angle))] for angle in range(6)]
        self.adjacent = []
        self.waterLand = random.randint(0, 1)
        self.mountainous = random.randint(0, 1)
        self.isLand = False
        self.isMountain = False
        self.isCoast = False
        self.region = None
        self.mountainRegion = None
        self.regionCol = None

    def draw(self, s, showRegions=False):
        pygame.draw.polygon(s, self.regionCol if self.isLand and showRegions else self.col, self.hex)

    def drawArrows(self, s):
        for adj in self.adjacent:
            angle = normalize_angle(ang((self.x, self.y), (adj.x, adj.y)))
            dist = distance((self.x, self.y), (adj.x, adj.y))
            factor = 0.35
            draw_arrow(s, (self.x, self.y), (self.x + dist * factor * math.cos(angle), self.y + dist * factor * math.sin(angle)), self.col.debugRed, pygame, 2, 5, 25)

    def showWaterLand(self, s, font):
        drawText(s, (18, 22, 27), font, self.x - self.size / 2, self.y - self.size / 2, str(round(self.waterLand, 2)))
