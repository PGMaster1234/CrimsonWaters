import pygame
import heapq
import itertools
import numpy as np


# Import Hex for type hinting if desired, but avoid circular import dependency if possible
# from generation import Hex # Careful with circular imports

def normalize_vector_np(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


class Resource:
    def __init__(self, tile, resourceType, img):
        self.tile = tile  # Keep direct tile reference for location
        self.resourceType = resourceType
        self.img = img
        self.resourceRate = 0
        if self.resourceType == 'wood': self.resourceRate = 5  # Example rate

    def draw(self, s):
        colors = {'wood': (115, 80, 32), 'stone': (123, 133, 150), 'iron': (106, 85, 125), 'pine': (63, 105, 51), 'amber': (184, 102, 48)}
        color = colors.get(self.resourceType)
        if color and self.tile and hasattr(self.tile, 'hex'):  # Check tile and hex exist
            try:
                pygame.draw.polygon(s, color, self.tile.hex)
            except TypeError:
                if hasattr(self.tile, 'center') and hasattr(self.tile, 'size'):
                    pygame.draw.circle(s, color, self.tile.center, self.tile.size * 0.4)

    # No pickling prep needed if it only holds reference to one tile


class LightHouse: pass


class DefensePost: pass


class Harbor:
    def __init__(self, tile):
        self.tile = tile  # Direct reference to the tile it sits on
        self.harbor_id = -1  # Assigned by TileHandler
        # --- Data stored using IDs for pickling ---
        self.tradeRoutesData = {}  # {target_harbor_id: List[tile_id]}
        # --- Data reconstructed after pickling ---
        self.tradeRouteObjects = {}  # {target_Harbor_object: List[Hex_object]}

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
                valid_path = True
                for tile_id in path_tile_ids:
                    tile_obj = tiles_by_id_map.get(tile_id)
                    if tile_obj:
                        path_objects.append(tile_obj)
                    else:
                        # print(f"Warning: Tile ID {tile_id} not found during route reconstruction for Harbor {self.harbor_id}.")
                        valid_path = False
                        break  # Stop reconstructing this path
                if valid_path:
                    self.tradeRouteObjects[target_harbor] = path_objects  # else: print(f"Warning: Target Harbor ID {target_hid} not found during route reconstruction for Harbor {self.harbor_id}.")

    def generateAllRoutes(self, other_harbors_in_ocean, waterTilesInOcean, tiles_by_id_map):
        """
        Finds the shortest paths to other harbors using IDs. Stores results in self.tradeRoutesData
        and targetHarbor.tradeRoutesData. Returns the number of PAIRS connected.
        Requires access to tiles_by_id_map to convert between objects and IDs.
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
                    heapq.heappush(frontier, (tentativeG, next(counter), neighbor))

        return routes_found_count

    def draw(self, s):
        if self.tile and hasattr(self.tile, 'hex'):
            try:
                pygame.draw.polygon(s, (200, 30, 30), self.tile.hex)
                pygame.draw.polygon(s, (50, 0, 0), self.tile.hex, 2)
            except TypeError:
                if hasattr(self.tile, 'center') and hasattr(self.tile, 'size'):
                    pygame.draw.circle(s, (200, 30, 30), self.tile.center, self.tile.size * 0.5)

    def drawRoute(self, s, otherHarbor, color=(127, 63, 63, 180)):
        """Draws a specific trade route path using the reconstructed object path."""
        # Use the reconstructed object path map
        pathObjects = self.tradeRouteObjects.get(otherHarbor)
        if pathObjects and len(pathObjects) >= 1:
            # Add start/end harbor tile centers
            points = [self.tile.center] + [tile.center for tile in pathObjects] + [otherHarbor.tile.center]
            if len(points) > 1:
                try:
                    draw_color = tuple(color) if len(color) == 4 and (s.get_flags() & pygame.SRCALPHA) else tuple(color[:3])
                    pygame.draw.lines(s, draw_color, False, points, 2)
                except (ValueError, TypeError) as e:
                    print(f"Error drawing route line: {e}")
