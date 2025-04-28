import pygame
import math
import random
from text import drawText
from calcs import distance, ang, normalize_angle, draw_arrow, linearGradient, normalize, randomCol
from territory import Territory


class TileHandler:
    def __init__(self, width, height, size, cols, waterThreshold=0.51, mountainThreshold=0.51, territorySize=100, font=None):
        self.surf = pygame.Surface((width, height)).convert_alpha()
        self.debugOverlay = pygame.Surface((width, height)).convert_alpha()
        self.transparentSurf = pygame.Surface((width, height)).convert_alpha()
        self.surf.fill(cols.dark)
        self.debugOverlay.fill(cols.dark)
        self.transparentSurf.fill((0, 0, 0, 0))

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

        self.generateTiles()
        self.assignAdjacentTiles()
        for _ in range(50):
            self.generationCycle()
        self.landRegions = self.findContiguousRegions([t for t in self.tiles if t.waterLand >= self.waterThreshold])
        self.mountainRegions = self.findContiguousRegions([t for t in self.tiles if (t.waterLand >= self.waterThreshold and t.mountainous > self.mountainThreshold)])
        self.setRegionReferences()
        self.createTerritories()

        self.setTileCols()
        self.draw2InternalScreen()

    def generateTiles(self):
        for x in range(self.gridSizeX):
            for y in range(self.gridSizeY):
                x_pos = self.horizontal_distance * x + self.size * (self.borderSize + 0.5)
                y_pos = self.vertical_distance * y + self.size * (self.borderSize + 0.5)
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
            elif tile.waterLand >= self.waterThreshold:
                weightLandDistribution = lambda x: (1 - 2 ** (-3 * x)) * 8 / 7
                tile.isLand = True
                if tile.mountainous < self.mountainThreshold:
                    tile.col = linearGradient([self.cols.oliveGreen, self.cols.darkOliveGreen], weightLandDistribution(normalize(tile.waterLand + random.uniform(-landColoringNoise, landColoringNoise), landBounds[0], landBounds[1], True)))
                else:
                    tile.isMountain = True
                    tile.col = linearGradient([self.cols.mountainBlue, self.cols.darkMountainBlue], normalize(tile.mountainous + random.uniform(-mountainColoringNoise, mountainColoringNoise), mountainBounds[0], mountainBounds[1], True))

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
            for _ in range(10):
                points, centers = self.iterate(points, centers)
                centers = self.shiftCenters(centers)
            for p in points:
                p[3][3].append(p[4]) # store tile reference
            self.contiguousTerritories.append([Territory([c[0], c[1]], c[3], self.cols) for c in centers])

    def draw2InternalScreen(self):
        for tile in self.tiles:
            tile.draw(self.surf, False)
            tile.draw(self.debugOverlay, True)
        for contiguousTerritoryList in self.contiguousTerritories:
            for territory in contiguousTerritoryList:
                territory.draw(self.surf, self.transparentSurf, self.debugOverlay)

    def draw(self, s, showArrows=False, showDebugOverlay=False, showWaterLand=False):
        s.blit(self.surf, (0, 0))
        if showDebugOverlay:
            s.blit(self.debugOverlay, (0, 0))
        if showArrows:
            for tile in self.tiles:
                tile.drawArrows(s)
        if showWaterLand:
            for tile in self.tiles:
                tile.showWaterLand(s, self.font)
        s.blit(self.transparentSurf, (0, 0))


class Hex:
    def __init__(self, grid_x, grid_y, x, y, size, col=(0, 0, 0)):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = x
        self.y = y
        self.size = size
        self.col = col
        self.center = [self.x, self.y]
        self.hex = [(self.x + self.size * math.cos(math.pi / 3 * angle), self.y + self.size * math.sin(math.pi / 3 * angle)) for angle in range(6)]
        self.adjacent = []
        self.waterLand = random.randint(0, 1)
        self.mountainous = random.randint(0, 1)
        self.isLand = False
        self.isMountain = False
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
