import pygame
import random
from calcs import randomCol, setOpacity
from shapely.geometry import Polygon, MultiPoint, Point
from shapely.ops import unary_union, snap
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

        # Spawn initial features within the territory
        self.spawnResources(ResourceInfo)
        self.spawnHarbors(StructureInfo)

    def territoryBorders(self, tiles):
        polys = []
        all_corners_cords = []  # Collect INT cords for snapping
        for tile in tiles:
            # --- Key Change: Use integer coordinates directly from tile.hex ---
            # Assuming tile.hex already stores integer vertex coordinates

            pts = [(int(p[0]), int(p[1])) for p in tile.hex]

            try:
                # Create polygon from integer coordinates
                poly = Polygon(pts)
                if poly.is_valid:
                    polys.append(poly)
                    # Add integer exterior coordinates for snapping
                    all_corners_cords.extend(poly.exterior.coords)  # These are already int tuples from pts  # else:  # print(f"Warning: Skipping invalid integer tile polygon at {tile.grid_x},{tile.grid_y}")
            except Exception as e:
                print(f"Warning: Error creating Shapely polygon (int cords) for tile {tile.grid_x},{tile.grid_y}: {e}")

        if not polys:
            print("Warning: No valid integer polygons to merge for territory.")
            return [], [], Polygon([])

        merged = unary_union(polys)
        merged = merged.buffer(0.1)

        if all_corners_cords:
            unique_corners = set(all_corners_cords)
            grid_pts = MultiPoint(list(unique_corners))

            merged = snap(merged, grid_pts, 0.1)

        merged = merged.simplify(0.1, preserve_topology=True)

        # Extract coordinate rings (will convert back to int)
        return self.extractRings(merged)

    @staticmethod
    def extractRings(merged):
        """Extracts exterior and interior coordinate rings from Shapely geometry for Pygame."""
        exteriors, interiors = [], []

        def extract(polygon):
            # Convert shapely float cords back to lists of ints for pygame drawing
            ext = [(int(p[0]), int(p[1])) for p in polygon.exterior.coords]
            ints = [[(int(p[0]), int(p[1])) for p in r.coords] for r in polygon.interiors]
            return ext, ints

        if isinstance(merged, Polygon):
            if merged.exterior:  # Check if the polygon is valid
                e, i = extract(merged)
                if e: exteriors.append(e)  # Only add if valid
                if i: interiors.extend(i)
        elif hasattr(merged, 'geoms'):  # Treat as MultiPolygon or GeometryCollection
            for poly in merged.geoms:
                if isinstance(poly, Polygon) and poly.exterior:
                    e, i = extract(poly)
                    if e: exteriors.append(e)
                    if i: interiors.extend(i)

        # Store the main polygon object for hover checks (can be None if extraction failed badly)
        polygon_obj = merged if isinstance(merged, Polygon) else None
        if not polygon_obj and hasattr(merged, 'geoms') and merged.geoms:
            # Try to get the first valid polygon from a collection
            polygon_obj = next((p for p in merged.geoms if isinstance(p, Polygon)), None)

        return exteriors, interiors, polygon_obj

    def spawnResources(self, spawnRates):
        for tile in self.tiles:
            if tile.isLand and not tile.isMountain:
                if random.random() < spawnRates.woodSpawnRate:
                    self.containedResources.append(Resource(tile, "wood", None))
                    continue
                if random.random() < spawnRates.pineSpawnRate:
                    self.containedResources.append(Resource(tile, "pine", None))
            if tile.isLand:
                if random.random() < spawnRates.stoneSpawnRate:
                    self.containedResources.append(Resource(tile, "stone", None))
            if tile.isLand and tile.isMountain:
                if random.random() < spawnRates.ironSpawnRate:
                    self.containedResources.append(Resource(tile, "iron", None))
            if tile.isLand and tile.isMountain and tile.isCoast:
                if random.random() < spawnRates.amberSpawnRate:
                    self.containedResources.append(Resource(tile, "amber", None))

    def spawnHarbors(self, spawnRates):
        for tile in self.tiles:
            if not tile.isLand or tile.isMountain:
                continue
            is_coastal = any(adj in self.allWaterTiles for adj in tile.adjacent)
            if not is_coastal:
                continue

            if random.random() < spawnRates.harborSpawnRate:
                self.harbors.append(Harbor(tile))

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
