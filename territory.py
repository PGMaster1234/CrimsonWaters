import math
import random
import pygame

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.ops import unary_union

    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon, MultiPolygon, Point = None, None, None
    unary_union = None
    SHAPELY_AVAILABLE = False

from calcs import randomCol, setOpacity
from locationalObjects import Resource, Harbor


class Territory:
    def __init__(self, screenWidth, screenHeight, centerPos, tiles, allWaterTiles, cols, resource_info=None, structure_info=None):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.centerPos = centerPos
        self.tiles = tiles
        self.allWaterTiles = allWaterTiles
        self.size = len(self.tiles)
        self.cols = cols
        self.resource_info = resource_info
        self.structure_info = structure_info

        num_res = getattr(resource_info, 'numResources', 0)
        self.resourceStorages = [0] * num_res
        self.containedResources = []
        self.harbors = []

        self.reachableHarbors = {}
        self.shortestPathToReachableTerritories = {}

        self.territoryCol = randomCol('r')
        self.selectedTerritoryCol = randomCol('b')
        self.claimed = None
        self.id = -1
        self.coastlines = []
        self.surf = None
        self.debugSurf = None

        self.polygon = None
        self.exteriors = []
        self.interiors = []

        if SHAPELY_AVAILABLE:
            self.exteriors, self.interiors, self.polygon = self.territoryBorders(self.tiles)

        self.landTiles = [t for t in self.tiles if t.isLand]
        self.mountainTiles = [t for t in self.tiles if t.isMountain]
        self.coastTiles = [t for t in self.tiles if t.isCoast]
        self.unusedSpawningTiles = list(self.tiles)

        self.spawnResources(self.resource_info)
        self.spawnHarbors(self.structure_info)
        for harbor in self.harbors:
            harbor.assignHarborParentReference(self)

    def prepare_for_pickling(self):
        self.surf = None
        self.debugSurf = None
        self.polygon = None
        self.reachableHarbors = {}

    def initialize_graphics_and_external_libs(self, tiles_by_id_map, harbors_by_id_map):
        self.surf = None
        self.debugSurf = None

        if SHAPELY_AVAILABLE:
            if self.tiles:
                _, _, self.polygon = self.territoryBorders(self.tiles)
            else:
                self.polygon = None

        for resource in self.containedResources:
            if hasattr(resource, 'initializeImg'):
                resource.initializeImg()

        for harbor in self.harbors:
            if hasattr(harbor, 'initialize_graphics_and_external_libs'):
                harbor.initialize_graphics_and_external_libs(tiles_by_id_map, harbors_by_id_map)

    def territoryBorders(self, tiles):
        if not SHAPELY_AVAILABLE or not tiles or Polygon is None:
            return [], [], None

        polys = []
        PRECISION = 8
        for tile in tiles:
            if not hasattr(tile, 'floatHexVertices'):
                continue
            pts = [(round(p[0], PRECISION), round(p[1], PRECISION)) for p in tile.floatHexVertices]
            if len(pts) < 3:
                continue
            polys.append(Polygon(pts))

        if not polys:
            return [], [], None

        merged = unary_union(polys)
        return self.extractRings(merged)

    @staticmethod
    def extractRings(merged):
        ext, inter, poly_obj = [], [], None
        if not SHAPELY_AVAILABLE or not merged or merged.is_empty or Polygon is None or MultiPolygon is None:
            return [], [], None

        def _extract(polygon_geom):
            extracted = [(int(round(p[0])), int(round(p[1]))) for p in polygon_geom.exterior.coords]
            ind = []
            for r_in in polygon_geom.interiors:
                if len(r_in.coords) > 2:
                    ind.append([(int(round(p[0])), int(round(p[1]))) for p in r_in.coords])
            return extracted, ind

        if isinstance(merged, Polygon):
            if merged.exterior:
                e, i = _extract(merged)
                poly_obj = merged
                ext.append(e)
                inter.extend(i)
        elif isinstance(merged, MultiPolygon):
            valid_polys = []
            for poly_item in merged.geoms:
                if isinstance(poly_item, Polygon) and poly_item.exterior:
                    e, i = _extract(poly_item)
                    ext.append(e)
                    inter.extend(i)
                    valid_polys.append(poly_item)
            if valid_polys:
                poly_obj = MultiPolygon(valid_polys) if len(valid_polys) > 1 else valid_polys[0]
        return ext, inter, poly_obj

    def spawnResources(self, info):
        if info is None: return
        for res_type in getattr(info, 'resourceTypes', []):
            spawnable_tiles_func = getattr(info, 'getSpawnableTiles', None)
            if not spawnable_tiles_func: continue

            spawnable_tiles = spawnable_tiles_func(res_type, self.unusedSpawningTiles)
            spawn_rate = getattr(info, 'spawnRates', {}).get(res_type, 0.0)
            num_to_spawn = int((len(spawnable_tiles) * spawn_rate + random.random()) ** 0.5)

            if spawnable_tiles and num_to_spawn > 0:
                k = min(num_to_spawn, len(spawnable_tiles))
                try:
                    selected_tiles = random.sample(spawnable_tiles, k)
                    for tile in selected_tiles:
                        if tile in self.unusedSpawningTiles:
                            self.containedResources.append(Resource(tile, res_type))
                            self.unusedSpawningTiles.remove(tile)
                except ValueError:
                    pass

    def spawnHarbors(self, info):
        if info is None: return
        possibleTiles = [t for t in self.coastTiles if not t.isMountain]
        if not possibleTiles: return

        spawnRate = getattr(info, 'harborSpawnRate', 0.01)
        num_potential_harbors = math.ceil(len(possibleTiles) * spawnRate * 2)
        num_potential_harbors = min(num_potential_harbors, len(possibleTiles))
        num_potential_harbors = max(1, num_potential_harbors) if possibleTiles else 0

        if not possibleTiles or num_potential_harbors == 0: return

        k_harbors = min(len(possibleTiles), num_potential_harbors)
        selected_harbor_tiles = random.sample(possibleTiles, k_harbors)

        for tile in selected_harbor_tiles:
            is_active_harbor = random.random() < 0.7
            new_harbor = Harbor(tile, is_active_harbor)
            self.harbors.append(new_harbor)
            if tile in self.unusedSpawningTiles:
                self.unusedSpawningTiles.remove(tile)

    def update_reachable_harbors(self):
        self.reachableHarbors.clear()
        self.shortestPathToReachableTerritories.clear()
        for local_harbor in self.harbors:
            if not hasattr(local_harbor, 'tradeRouteObjects') or not hasattr(local_harbor, 'tradeRoutesPoints'):
                continue

            current_reachable = []
            for targetHarbor, route_tiles in local_harbor.tradeRouteObjects.items():
                if not hasattr(targetHarbor, 'parentTerritory'): continue

                current_reachable.append(targetHarbor)
                routeLength = len(route_tiles)
                destinationTerritory = targetHarbor.parentTerritory
                routePoints = local_harbor.tradeRoutesPoints.get(targetHarbor)

                if routePoints is None: continue

                if destinationTerritory not in self.shortestPathToReachableTerritories or routeLength < self.shortestPathToReachableTerritories[destinationTerritory][2]:
                    self.shortestPathToReachableTerritories[destinationTerritory] = [local_harbor, targetHarbor, routeLength, routePoints]

            if current_reachable:
                self.reachableHarbors[local_harbor] = current_reachable

    def drawInternalTerritoryBaseline(self, target_surf, target_debug_surf):
        if target_surf is None or target_debug_surf is None: return

        if hasattr(self.cols, 'dark'):
            pygame.draw.circle(target_debug_surf, self.cols.dark, self.centerPos, 5, 2)

        borderCol = setOpacity(self.cols.dark, 180)
        borderWidth = 3
        for border in self.exteriors:
            if len(border) > 1:
                pygame.draw.lines(target_surf, borderCol, True, border, width=borderWidth)
        for border in self.interiors:
            if len(border) > 1:
                pygame.draw.lines(target_surf, borderCol, True, border, width=borderWidth)

    def drawInternalStructures(self, target_surf):
        for resource in self.containedResources:
            resource.draw(target_surf)
        for harbor in self.harbors:
            harbor.draw(target_surf)

    def drawHover(self, s):
        fill_color = setOpacity(self.territoryCol, 60)
        line_color = setOpacity(self.territoryCol, 200)
        width = 4

        for border in self.exteriors:
            if len(border) > 2: pygame.draw.polygon(s, fill_color, border)
            if len(border) > 1: pygame.draw.lines(s, line_color, True, border, width=width)
        for border in self.interiors:
            if len(border) > 1: pygame.draw.lines(s, line_color, True, border, width=width)

    def drawSelected(self, s):
        fill_color = setOpacity(self.selectedTerritoryCol, 60)
        line_color = setOpacity(self.selectedTerritoryCol, 200)
        width = 6

        for border in self.exteriors:
            if len(border) > 2: pygame.draw.polygon(s, fill_color, border)
            if len(border) > 1: pygame.draw.lines(s, line_color, True, border, width=width)
        for border in self.interiors:
            if len(border) > 1: pygame.draw.lines(s, line_color, True, border, width=width)

    def drawRoutes(self, s, color):
        for src_harbor, reachable_target_harbors in self.reachableHarbors.items():
            for target_harbor in reachable_target_harbors:
                if hasattr(src_harbor, 'drawRoute'):
                    src_harbor.drawRoute(s, target_harbor, color, False)
