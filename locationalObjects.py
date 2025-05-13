import pygame
import heapq
import itertools
import numpy as np
import os
from calcs import isAngleNearMultiple, catmullRomCentripetal


def normalize_vector_np(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class Resource:
    def __init__(self, tile, resourceType):
        self.tile = tile
        self.resourceType = resourceType
        self.resourceRate = 0
        self.img = None
        self.imgDims = None
        if self.resourceType == 'wood': self.resourceRate = 5

    def initializeImg(self):
        filename = f"assets/structures/{self.resourceType}Icon.png"
        if os.path.exists(filename):
            imgScalar = 5
            self.img = pygame.transform.scale(pygame.image.load(filename).convert_alpha(), [self.tile.size * imgScalar] * 2)
            self.imgDims = self.img.get_width(), self.img.get_height()

    def draw(self, s):
        s.blit(self.img, (self.tile.x - self.imgDims[0] / 2, self.tile.y - self.imgDims[1] / 2))


class LightHouse: pass


class DefensePost: pass


class Harbor:
    def __init__(self, tile, isUsable=False):
        self.tile = tile  # Direct reference to the tile it sits on
        self.harbor_id = -1  # Assigned by TileHandler
        # --- Data stored using IDs for pickling ---
        self.tradeRoutesData = {}  # {target_harbor_id: List[tile_id]}
        self.tradeRoutesPoints = {}  # {target_harbor_id: List[xi, yi]}
        # --- Data reconstructed after pickling ---
        self.tradeRouteObjects = {}  # {target_Harbor_object: List[Hex_object]}
        self.isUsable = isUsable

        self.prunedPathPoints = []

    def prepare_for_pickling(self):
        """Ensure only ID-based data is kept for pickling."""
        # Nullify the object-based route dictionary
        self.tradeRouteObjects = {}  # self.tile reference is kept, assuming Hex pickling is handled

    def initialize_graphics_and_external_libs(self, tiles_by_id_map, harbors_by_id_map):
        """Reconstructs the tradeRouteObjects map from tradeRoutesData using IDs."""
        self.tradeRouteObjects = {}  # Clear existing (should be empty anyway)
        for target_hid, path_tile_ids in self.tradeRoutesData.items():
            target_harbor = harbors_by_id_map.get(target_hid)
            if target_harbor:
                path_objects = []
                points = []
                valid_path = True
                for tile_id in path_tile_ids:
                    tile_obj = tiles_by_id_map.get(tile_id)
                    if tile_obj:
                        path_objects.append(tile_obj)
                        points.append(tile_obj.center)
                    else:
                        print(f"Warning: Tile ID {tile_id} not found during route reconstruction for Harbor {self.harbor_id}.")
                        valid_path = False
                        break  # Stop reconstructing this path
                if valid_path:
                    self.tradeRouteObjects[target_harbor] = path_objects

                    # prune path for redundant points
                    popping = []
                    self.prunedPathPoints = []
                    for i in range(len(points)):
                        if len(points) > i + 2:
                            if isAngleNearMultiple(points[i], points[i + 2], 60, 1):
                                popping.append(i + 1)
                                self.prunedPathPoints.append(points[i + 1])
                                continue
                    for pop in reversed(popping):
                        points.pop(pop)
                    self.tradeRoutesPoints[target_harbor] = catmullRomCentripetal([self.tile.center] + points + [target_harbor.tile.center], 20)[0::2]

    def generateAllRoutes(self, other_harbors_in_ocean, waterTilesInOcean):
        """
        Finds the shortest paths to other harbors using IDs. Stores results in self.tradeRoutesData
        and targetHarbor.tradeRoutesData. Returns the number of PAIRS connected.
        """
        routes_found_count = 0
        if not other_harbors_in_ocean: return 0

        turnCostFactor = -0.001
        counter = itertools.count()

        startWaterNeighborTiles = {w for w in self.tile.adjacent if w in waterTilesInOcean}
        if not startWaterNeighborTiles: return 0

        # Map water tiles adjacent to targets back to the target harbor object ID
        targetWaterMap = {}  # {water_tile_object: target_harbor_id}
        targetHarborIdSet = set()
        for h in other_harbors_in_ocean:
            if h == self or h.harbor_id == -1: continue  # Skip self or unassigned IDs
            isTarget = False
            for w in h.tile.adjacent:
                if w in waterTilesInOcean:
                    targetWaterMap[w] = h.harbor_id  # Map water tile to target ID
                    isTarget = True
            if isTarget: targetHarborIdSet.add(h.harbor_id)

        if not targetHarborIdSet: return 0

        # Dijkstra Initialization (operates on tile objects internally for convenience)
        frontier = []
        cameFrom = {}  # {tile_object: predecessor_tile_object}
        gScore = {w: float('inf') for w in waterTilesInOcean}

        for startNeighbor in startWaterNeighborTiles:
            initialCost = 1.0
            gScore[startNeighbor] = initialCost
            cameFrom[startNeighbor] = self.tile
            heapq.heappush(frontier, (initialCost, next(counter), startNeighbor))

        targets_remaining = targetHarborIdSet.copy()

        pathLength = {w: 0 for w in waterTilesInOcean}

        while frontier and targets_remaining:
            gCurr, _, currentWaterTile = heapq.heappop(frontier)

            if gCurr > gScore.get(currentWaterTile, float('inf')): continue

            # Check if this water tile is adjacent to one of our remaining targets
            targetHarborId = targetWaterMap.get(currentWaterTile)
            if targetHarborId is not None and targetHarborId in targets_remaining:
                # Reconstruct path (list of tile objects)
                path_objects = []
                temp = currentWaterTile
                possible = True
                while temp != self.tile:
                    path_objects.append(temp)
                    prev_temp = cameFrom.get(temp)
                    if prev_temp is None or prev_temp == temp:
                        path_objects = None
                        possible = False
                        break
                    temp = prev_temp

                if possible and path_objects is not None:
                    final_path_objects = path_objects[::-1]
                    # Convert path objects to path IDs
                    final_path_ids = [t.tile_id for t in final_path_objects if hasattr(t, 'tile_id')]

                    if len(final_path_ids) == len(final_path_objects):  # Ensure all tiles had IDs
                        # Store path IDs in BOTH directions using IDs
                        self.tradeRoutesData[targetHarborId] = final_path_ids

                        # Find the target harbor object to store the reverse path
                        target_harbor_object = None
                        for h in other_harbors_in_ocean:  # Inefficient lookup, better if map passed
                            if h.harbor_id == targetHarborId:
                                target_harbor_object = h
                                break
                        if target_harbor_object:
                            if not hasattr(target_harbor_object, 'tradeRoutesData'):
                                target_harbor_object.tradeRoutesData = {}
                            target_harbor_object.tradeRoutesData[self.harbor_id] = final_path_ids[::-1]  # Store reversed IDs
                            routes_found_count += 1  # Increment pair count  # else: print(f"Warning: Could not find target harbor object for ID {targetHarborId} to store reverse path.")  # else: print(f"Warning: Could not get IDs for all tiles in path for harbor {self.harbor_id} -> {targetHarborId}")

                targets_remaining.remove(targetHarborId)
                if not targets_remaining: break

            # Explore Neighbors (using tile objects)
            prevTile = cameFrom.get(currentWaterTile)
            if prevTile is None: continue
            currentCenterNp = np.array(currentWaterTile.center)
            prevCenterNp = np.array(prevTile.center)

            for neighbor in currentWaterTile.adjacent:
                if neighbor not in waterTilesInOcean: continue

                neighborCenterNp = np.array(neighbor.center)
                baseCost = 1.0
                turnAdjustment = 0.0

                if prevTile != self.tile:
                    vec1 = currentCenterNp - prevCenterNp
                    vec2 = neighborCenterNp - currentCenterNp
                    normVec1 = normalize_vector_np(vec1)
                    normVec2 = normalize_vector_np(vec2)
                    if np.any(normVec1) and np.any(normVec2):
                        dot = np.clip(np.dot(normVec1, normVec2), -1.0, 1.0)
                        turnAdjustment = turnCostFactor * (1.0 - dot)

                tentativeG = gCurr + baseCost + turnAdjustment
                if tentativeG < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = currentWaterTile
                    gScore[neighbor] = tentativeG

                    # skip this path, it's too long
                    pathLength[neighbor] = pathLength[currentWaterTile] + 1
                    if pathLength[neighbor] > 20:
                        continue
                    
                    heapq.heappush(frontier, (tentativeG, next(counter), neighbor))

        return routes_found_count

    def draw(self, s):
        pygame.draw.polygon(s, ((200, 30, 30) if self.isUsable else (100, 10, 10)), self.tile.hex)

    def drawRoute(self, s, otherHarbor, color=(127, 63, 63, 180), debug=False):
        # Use the reconstructed object path map
        pathObjects = self.tradeRouteObjects[otherHarbor]
        if pathObjects and len(pathObjects) >= 1:
            points = self.tradeRoutesPoints[otherHarbor]
            draw_color = tuple(color) if len(color) == 4 and (s.get_flags() & pygame.SRCALPHA) else tuple(color[:3])
            if len(points) > 1:
                pygame.draw.lines(s, draw_color, False, points, 2)
            if debug:
                if len(points) > 1:
                    for p in points:
                        pygame.draw.circle(s, (0, 0, 255), p, 3)
                    pygame.draw.lines(s, (0, 0, 255), False, points, 1)
                if len(self.prunedPathPoints) > 1:
                    for p in self.prunedPathPoints:
                        pygame.draw.circle(s, (0, 255, 0), p, 3)
