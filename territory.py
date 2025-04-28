import pygame
import math
from calcs import randomCol, setOpacity
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


class Territory:
    def __init__(self, centerPos, tiles, cols):
        self.centerPos = centerPos
        self.tiles = tiles
        self.size = len(self.tiles)
        self.cols = cols

        self.exteriors, self.interiors = self.territoryBorders(self.tiles)

    @staticmethod
    def territoryBorders(tiles):
        # 1) build a Shapely Polygon for each tile
        polys = []
        for tile in tiles:
            # derive the 6 corner points of this hex
            pts = [(tile.x + tile.size * math.cos(math.pi / 3 * i), tile.y + tile.size * math.sin(math.pi / 3 * i)) for i in range(6)]
            polys.append(Polygon(pts))

        # 2) union them all
        merged = unary_union(polys)

        # merged may be a Polygon or a MultiPolygon, so we define a helper to extract boundaries from a single Polygon
        def extract(polygon):
            exterior = list(polygon.exterior.coords)
            interiorsList = [list(ring.coords) for ring in polygon.interiors]
            return exterior, interiorsList

        # 3) collect all exterior/interior rings
        exteriors = []
        interiors = []
        if isinstance(merged, Polygon):
            ext, ints = extract(merged)
            exteriors.append(ext)
            interiors.extend(ints)
        elif isinstance(merged, MultiPolygon):
            for poly in merged.geoms:
                ext, ints = extract(poly)
                exteriors.append(ext)
                interiors.extend(ints)
        else:
            # unlikely, but handle
            raise ValueError(f"unexpected geometry type: {type(merged)}")
        return exteriors, interiors

    def draw(self, s, transparentS, debugS):
        for border in self.exteriors:
            territoryCol = randomCol('red')
            pygame.draw.polygon(transparentS, setOpacity(territoryCol, 50), border)
            pygame.draw.lines(transparentS, setOpacity(territoryCol, 100), True, border, width=3)

        pygame.draw.circle(s, self.cols.dark, self.centerPos, 4, 2)
        pygame.draw.circle(debugS, self.cols.dark, self.centerPos, 4, 2)
