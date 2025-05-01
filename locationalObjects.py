import pygame
import heapq
import itertools
import numpy as np


# Helper function for NumPy vector normalization
def normalize_vector_np(v):
    norm = np.linalg.norm(v)
    # Use snake_case here as it's a global function, not part of the request scope
    return v if norm == 0 else v / norm


class Resource:
    # wood: any land : all ships, all structures
    # stone: prefers coast : all structures
    # iron: prefers mountains : warships, defensePosts
    # pine (gets converted to tar): further inland : ship repairs, tanky ships like Carrack and Galleon
    # amber: prefers mountains, rare : specific upgrades that give buffs
    #                                       LightHouse and DefensePost gives morale (speed) boost to nearby ships
    #                                       Upgrade harbors to increase resource unload speed
    #                                       Specific ship upgrades that increase MS or ATK

    def __init__(self, tile, resourceType, img):
        self.tile = tile
        self.resourceType = resourceType
        self.img = img
        self.resourceRate = 0
        if self.resourceType == 'wood':
            self.resourceRate = 5

    def draw(self, s):
        # Draw resource tile based on type
        colors = {'wood': (115, 80, 32), 'stone': (123, 133, 150), 'iron': (106, 85, 125), 'pine': (63, 105, 51), 'amber': (184, 102, 48)}
        color = colors.get(self.resourceType)
        if color:
            pygame.draw.polygon(s, color, self.tile.hex)
        # s.blit(self.img, self.tile.center)


class LightHouse:
    # spawned on land, provides large vision radius
    pass


class Harbor:
    # lets trade ships steal loot
    # controlled harbors can spawn/rebuild ships and collect resources from returning ones
    # most territories don't spawn with a harbor, so players must take adjacent tiles and land invade to reach them

    # can only build one ship at a time
    #    build speed can be upgraded
    def __init__(self, tile):
        self.tile = tile
        self.tradeRoutes = {} # Stores calculated paths {targetHarbor: [tile, tile, ...]}

    def generateAllRoutes(self, otherHarbors, waterTilesInOcean):
        """
        Single-source multi-target Dijkstra using NumPy for vector math.
        Finds the shortest paths through water tiles, adjusting cost based on turns.
        Uses camelCase internally.
        """
        self.tradeRoutes = {}
        # Negative value encourages turns (makes path cost lower). Positive discourages. Zero ignores.
        turnCostFactor = -0.1
        counter = itertools.count()  # Tie-breaker for heapq

        startWaterNeighbors = {w for w in self.tile.adjacent if w in waterTilesInOcean}
        if not startWaterNeighbors:
            return

        # Map water tiles adjacent to targets back to the target harbor object
        targetWaterMap = {}
        targetHarborSet = set()
        for h in otherHarbors:
            if h == self: continue
            isTarget = False
            for w in h.tile.adjacent:
                if w in waterTilesInOcean:
                    targetWaterMap[w] = h
                    isTarget = True
            if isTarget:
                targetHarborSet.add(h)

        if not targetHarborSet:
            return

        # Dijkstra Initialization
        frontier = []
        cameFrom = {} # Stores {tile: predecessor_tile_in_path}
        gScore = {w: float('inf') for w in waterTilesInOcean} # Stores cost to reach tile

        for startNeighbor in startWaterNeighbors:
            initialCost = 1.0
            gScore[startNeighbor] = initialCost
            cameFrom[startNeighbor] = self.tile # Conceptual start from harbor land tile
            heapq.heappush(frontier, (initialCost, next(counter), startNeighbor))

        # Dijkstra Loop
        while frontier and targetHarborSet:
            gCurr, _, currentWaterTile = heapq.heappop(frontier)

            if gCurr > gScore.get(currentWaterTile, float('inf')):
                continue # Already found a shorter path

            # Check if a target harbor is reached
            if currentWaterTile in targetWaterMap:
                targetHarbor = targetWaterMap[currentWaterTile]
                if targetHarbor in targetHarborSet:
                    # Reconstruct path
                    path = []
                    temp = currentWaterTile
                    while temp != self.tile: # Trace back
                        path.append(temp)
                        if temp not in cameFrom:
                            path = None
                            break
                        prevTemp = cameFrom.get(temp)
                        if prevTemp == temp:
                            path = None
                            break # Avoid infinite loops
                        temp = prevTemp

                    if path:
                        self.tradeRoutes[targetHarbor] = path[::-1] # Store reversed path

                    targetHarborSet.remove(targetHarbor)
                    if not targetHarborSet: break # Optimization: exit if all targets found

            # Explore Neighbors
            prevTile = cameFrom.get(currentWaterTile)
            if prevTile is None: continue

            currentCenterNp = np.array(currentWaterTile.center)
            prevCenterNp = np.array(prevTile.center)

            for neighbor in currentWaterTile.adjacent:
                if neighbor not in waterTilesInOcean: continue

                neighborCenterNp = np.array(neighbor.center)

                # Calculate Cost, adjusting for turns
                baseCost = 1.0
                turnAdjustment = 0.0

                # Apply turn cost adjustment only after the first step
                if prevTile != self.tile:
                    vec1 = currentCenterNp - prevCenterNp
                    vec2 = neighborCenterNp - currentCenterNp

                    normVec1 = normalize_vector_np(vec1)
                    normVec2 = normalize_vector_np(vec2)

                    if np.any(normVec1) and np.any(normVec2): # Avoid dot product with zero vectors
                        dot = np.dot(normVec1, normVec2)
                        dot = np.clip(dot, -1.0, 1.0)
                        # Adjust cost based on turn angle: factor * (1 - cos(angle))
                        turnAdjustment = turnCostFactor * (1.0 - dot)

                tentativeG = gCurr + baseCost + turnAdjustment

                # Update path if this route is shorter
                if tentativeG < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = currentWaterTile
                    gScore[neighbor] = tentativeG
                    heapq.heappush(frontier, (tentativeG, next(counter), neighbor))

    @staticmethod
    def heuristic(a, b): # Parameters a, b kept short as is conventional
        # A* heuristic (Euclidean distance) - not used in current Dijkstra
        return np.linalg.norm(np.array(a.center) - np.array(b.center))

    def draw(self, s):
        # Draw the harbor tile itself
        pygame.draw.polygon(s, (255, 0, 0), self.tile.hex)

    def drawRoute(self, s, otherHarbor):
        # Draws a specific trade route line
        if otherHarbor in self.tradeRoutes:
            pathTiles = self.tradeRoutes[otherHarbor]
            if len(pathTiles) > 1:
                points = [tile.center for tile in pathTiles]
                # Draw lines with some transparency
                pygame.draw.lines(s, (127, 63, 63, 180), False, points, 3)


class DefensePost:
    # acts like a land-based, stationary warship
    # best placed near choke points and near the coast
    # land invasion defense buff
    pass
