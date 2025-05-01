import pygame
import random

from calcs import randomCol, setOpacity
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point
from shapely.ops import unary_union, snap
from shapely.validation import make_valid
from controlPanel import ResourceInfo, StructureInfo
from locationalObjects import Resource, Harbor


class Territory:
    def __init__(self, centerPos, tiles, allWaterTiles, cols):
        self.centerPos = centerPos  # Calculated center from K-means
        self.tiles = tiles  # List of Hex objects belonging to this territory
        self.allWaterTiles = allWaterTiles  # Reference needed for spawning coastal harbors
        self.size = len(self.tiles)
        self.cols = cols
        self.resourceStorages = [0] * ResourceInfo.numResources  # Inventory placeholder
        self.containedResources = []  # List of Resource objects within territory
        self.harbors = []  # List of Harbor objects within territory
        # This dict maps harbors IN THIS territory to a list of OTHER harbors they can reach.
        # Populated by TileHandler after pathfinding.
        self.reachableHarbors = {}  # {harbor_object_in_this_territory: list_of_reachable_other_harbor_objects}

        self.territoryCol = randomCol('r')

        # Generate border polygon using Shapely if available
        self.exteriors, self.interiors, self.polygon = self.territoryBorders(self.tiles)

        # find tile types
        self.landTiles = [t for t in self.tiles if t.isLand]
        self.mountainTiles = [t for t in self.tiles if t.isMountain]
        self.coastTiles = [t for t in self.tiles if t.isCoast]
        self.unusedSpawningTiles = [t for t in self.tiles]

        self.spawnResources(ResourceInfo)
        self.spawnHarbors(StructureInfo)

    def territoryBorders(self, tiles):
        """Generates territory borders using Shapely."""
        polys = []
        all_corners_coords_rounded = []
        PRECISION = 8 # Precision for rounding float coordinates

        if not tiles:
            print("Warning: No tiles provided to territoryBorders.")
            return [], [], None

        for tile in tiles:
            # Round float vertices to fixed precision before creating polygon
            pts_float_rounded = [(round(p[0], PRECISION), round(p[1], PRECISION)) for p in tile.floatHexVertices]

            poly = Polygon(pts_float_rounded)

            if poly.is_valid:
                polys.append(poly)
                all_corners_coords_rounded.extend(pts_float_rounded)
            else:
                # Attempt to fix invalid polygons
                fixed_poly = make_valid(poly)
                if isinstance(fixed_poly, Polygon) and fixed_poly.is_valid:
                    polys.append(fixed_poly)
                    all_corners_coords_rounded.extend([(round(p[0], PRECISION), round(p[1], PRECISION)) for p in fixed_poly.exterior.coords])

        if not polys:
            print("Warning: No valid polygons to merge for territory.")
            return [], [], None

        # Merge the tile polygons
        merged = unary_union(polys)

        # --- Post-processing merged polygon ---
        merged = merged.buffer(0) # Clean up boundaries

        # Snap vertices with a very small tolerance
        snap_tolerance = 1e-9
        if all_corners_coords_rounded:
            unique_corners = set(all_corners_coords_rounded)
            if unique_corners:
                grid_pts = MultiPoint(list(unique_corners))
                merged = snap(merged, grid_pts, snap_tolerance)

        # Simplify to remove redundant vertices
        merged = merged.simplify(0.0, preserve_topology=True)

        # Ensure final polygon is valid
        if not merged.is_valid:
            merged = make_valid(merged)

        # Extract coordinate rings for drawing
        return self.extractRings(merged)

    @staticmethod
    def extractRings(merged):
        """Extracts exterior/interior rings (int coords) and the Shapely object."""
        exteriors, interiors = [], []
        polygon_obj_for_hover = None

        if not merged or merged.is_empty:
            return [], [], None # Invalid input geometry

        def extract(polygon):
            """Helper to extract int coords from a polygon's rings."""
            # Round before int conversion
            ext = [(int(round(p[0])), int(round(p[1]))) for p in polygon.exterior.coords]
            ints = [[(int(round(p[0])), int(round(p[1]))) for p in r.coords] for r in polygon.interiors]
            return ext, ints

        # Handle Polygon or MultiPolygon results
        if isinstance(merged, Polygon):
            if merged.is_valid and merged.exterior:
                e, i = extract(merged)
                if e: exteriors.append(e)
                if i: interiors.extend(i)
                polygon_obj_for_hover = merged
        elif isinstance(merged, MultiPolygon):
            valid_polygons = []
            for poly in merged.geoms:
                if isinstance(poly, Polygon) and poly.is_valid and poly.exterior:
                    e, i = extract(poly)
                    if e: exteriors.append(e)
                    if i: interiors.extend(i)
                    valid_polygons.append(poly)
            if valid_polygons:
                polygon_obj_for_hover = merged # Store the MultiPolygon itself

        return exteriors, interiors, polygon_obj_for_hover

    def spawnResources(self, info):
        for resource in info.resourceTypes:
            spawnableTiles = info.getSpawnableTiles(resource, self.unusedSpawningTiles)
            count = len(spawnableTiles) * info.spawnRates[resource]
            numResource = int(count) + int((random.random() < (count % 1)) or (int(count) == 0 and resource == 'wood'))
            for _ in range(numResource):
                if not len(spawnableTiles):
                    return
                sampledTile = random.choice(self.unusedSpawningTiles)
                self.containedResources.append(Resource(sampledTile, resource, None))
                self.unusedSpawningTiles.remove(sampledTile)

    def spawnHarbors(self, spawnRates):
        if random.random() < spawnRates.harborSpawnRate * self.size:
            possibleSpawningLocations = [tile for tile in self.tiles if (tile.isCoast and not tile.isMountain)]
            if possibleSpawningLocations:
                self.harbors.append(Harbor(random.choice(possibleSpawningLocations)))

    def draw(self, s, debugS):
        pygame.draw.circle(debugS, self.cols.dark, self.centerPos, 5, 2)
        border_col = setOpacity(self.cols.dark, 180)
        for border in self.exteriors:
            pygame.draw.lines(s, border_col, True, border, width=3)
        for border in self.interiors:
            pygame.draw.lines(s, border_col, True, border, width=3)

    def drawCurrent(self, s, mx, my):
        for resource in self.containedResources:
            resource.draw(s)

        for harbor in self.harbors:
            harbor.draw(s)
            if harbor in self.reachableHarbors:
                for otherHarbor in self.reachableHarbors[harbor]:
                    harbor.drawRoute(s, otherHarbor)

        hover = self.polygon.contains(Point(mx, my))

        if hover:
            fill_color = setOpacity(self.territoryCol, 60)
            line_color = setOpacity(self.territoryCol, 200)
            for border in self.exteriors:
                pygame.draw.polygon(s, fill_color, border)
                pygame.draw.lines(s, line_color, True, border, width=4)
            for border in self.interiors:
                pygame.draw.lines(s, line_color, True, border, width=4)
