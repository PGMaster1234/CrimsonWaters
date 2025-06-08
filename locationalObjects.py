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

    def draw(self, s, scroll_x, scroll_y): # scroll_x, scroll_y are added but expected to be 0 for static map elements
        if self.img and self.imgDims:
            s.blit(self.img, (self.tile.x - self.imgDims[0] / 2, self.tile.y - self.imgDims[1] / 2))


class LightHouse: pass


class DefensePost: pass


class Harbor:
    def __init__(self, tile, isUsable=False):
        self.parentTerritory = None
        self.tile = tile
        self.harbor_id = -1
        self.tradeRoutesData = {}
        self.tradeRoutesPoints = {}
        self.tradeRouteObjects = {}
        self.isUsable = isUsable

        self.prunedPathPoints = []

    def assignHarborParentReference(self, parentTerritory):
        self.parentTerritory = parentTerritory

    def prepare_for_pickling(self):
        self.tradeRouteObjects = {}

    def initialize_graphics_and_external_libs(self, tiles_by_id_map, harbors_by_id_map):
        self.tradeRouteObjects = {}
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
                        break
                if valid_path:
                    self.tradeRouteObjects[target_harbor] = path_objects

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

    def generateAllRoutes(self, other_harbors_in_ocean, waterTilesInOcean, ocean_harbors_by_id_map):
        routes_found_count = 0
        if not other_harbors_in_ocean: return 0

        turnCostFactor = -0.001
        counter = itertools.count()

        startWaterNeighborTiles = {w for w in self.tile.adjacent if w in waterTilesInOcean}
        if not startWaterNeighborTiles: return 0

        targetWaterMap = {}
        targetHarborIdSet = set()
        for h in other_harbors_in_ocean:
            if h == self or h.harbor_id == -1: continue
            isTarget = False
            for w in h.tile.adjacent:
                if w in waterTilesInOcean:
                    targetWaterMap[w] = h.harbor_id
                    isTarget = True
            if isTarget: targetHarborIdSet.add(h.harbor_id)

        if not targetHarborIdSet: return 0

        frontier = []
        cameFrom = {}
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

            targetHarborId = targetWaterMap.get(currentWaterTile)
            if targetHarborId is not None and targetHarborId in targets_remaining:
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
                    final_path_ids = [t.tile_id for t in final_path_objects if hasattr(t, 'tile_id')]

                    if len(final_path_ids) == len(final_path_objects):
                        self.tradeRoutesData[targetHarborId] = final_path_ids

                        target_harbor_object = ocean_harbors_by_id_map.get(targetHarborId)
                        if target_harbor_object:
                            if not hasattr(target_harbor_object, 'tradeRoutesData'):
                                target_harbor_object.tradeRoutesData = {}
                            target_harbor_object.tradeRoutesData[self.harbor_id] = final_path_ids[::-1]
                            routes_found_count += 1

                targets_remaining.remove(targetHarborId)
                if not targets_remaining: break

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

                    pathLength[neighbor] = pathLength[currentWaterTile] + 1
                    if pathLength[neighbor] > 20 and (routes_found_count > 0):
                        continue

                    heapq.heappush(frontier, (tentativeG, next(counter), neighbor))

        return routes_found_count

    def draw(self, s, scroll_x, scroll_y): # scroll_x, scroll_y are added but expected to be 0 for static map elements
        shifted_hex = [(p[0] + scroll_x, p[1] + scroll_y) for p in self.tile.hex]
        pygame.draw.polygon(s, ((200, 30, 30) if self.isUsable else (100, 10, 10)), shifted_hex)

    def drawRoute(self, s, otherHarbor, color=(94, 32, 32), debug=False, scroll_x=0, scroll_y=0):
        if otherHarbor not in self.tradeRouteObjects or not self.tradeRouteObjects[otherHarbor]:
            return

        points = self.tradeRoutesPoints.get(otherHarbor)
        if points is None:
            return

        draw_color = tuple(color) if len(color) == 4 and (s.get_flags() & pygame.SRCALPHA) else tuple(color[:3])

        shifted_points = [(p[0] + scroll_x, p[1] + scroll_y) for p in points]

        if len(shifted_points) > 1:
            pygame.draw.lines(s, draw_color, False, shifted_points, 3)
        if debug:
            if len(shifted_points) > 1:
                for p in shifted_points:
                    pygame.draw.circle(s, (0, 0, 255), p, 3)
                pygame.draw.lines(s, (0, 0, 255), False, shifted_points, 2)
            if len(self.prunedPathPoints) > 1:
                shifted_pruned_points = [(p[0] + scroll_x, p[1] + scroll_y) for p in self.prunedPathPoints]
                for p in shifted_pruned_points:
                    pygame.draw.circle(s, (0, 255, 0), p, 4)
