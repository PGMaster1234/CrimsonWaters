import random

import pygame

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

from calcs import randomCol, setOpacity

SHAPELY_AVAILABLE = True

from locationalObjects import Resource, Harbor

from ships import Ship
from controlPanel import ShipInfo, ResourceInfo


class Territory:
    def __init__(self, screenWidth, screenHeight, centerPos, tiles, allWaterTiles, cols, resource_info=None, structure_info=None):
        self.screenWidth, self.screenHeight = screenWidth, screenHeight
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
        self.reachableHarbors = {}  # Stores {local_harbor_obj: [reachable_harbor_objs]}
        self.territoryCol = randomCol('r')
        self.claimed = None  # Placeholder for ownership
        self.id = -1  # Unique identifier assigned by TileHandler

        self.ships = []

        # Surfaces are managed by TileHandler; these are effectively null
        self.surf = None
        self.debugSurf = None

        self.polygon: Polygon | MultiPolygon | None = None  # Shapely object for geometry checks
        self.exteriors = []  # List of exterior border coordinate lists
        self.interiors = []  # List of interior border (hole) coordinate lists

        # Generate borders using Shapely if available
        if SHAPELY_AVAILABLE:
            self.exteriors, self.interiors, self.polygon = self.territoryBorders(self.tiles)
        else:
            self.exteriors, self.interiors = [], []  # Default if Shapely is missing

        # Categorize tiles within the territory
        self.landTiles = [t for t in self.tiles if t.isLand]
        self.mountainTiles = [t for t in self.tiles if t.isMountain]
        self.coastTiles = [t for t in self.tiles if t.isCoast]
        self.unusedSpawningTiles = list(self.tiles)  # Tiles available for spawning objects

        # Spawn initial resources and structures
        self.spawnResources(self.resource_info)
        self.spawnHarbors(self.structure_info)

    def prepare_for_pickling(self):
        """Prepare territory data for saving (serialization)."""
        self.surf = None
        self.debugSurf = None
        self.polygon = None  # Shapely objects cannot be pickled directly
        self.reachableHarbors = {}  # Rebuilt after loading

    def initialize_graphics_and_external_libs(self):
        """Initialize graphics-related attributes and regenerate Shapely polygon after loading."""
        self.surf = None  # Not used by Territory directly
        self.debugSurf = None  # Not used by Territory directly
        if SHAPELY_AVAILABLE:
            # Regenerate the polygon using the loaded tiles
            _, _, self.polygon = self.territoryBorders(self.tiles)

    def territoryBorders(self, tiles):
        """Calculates the exterior and interior borders of the territory using Shapely."""
        if not SHAPELY_AVAILABLE or not tiles:
            return [], [], None

        polys = []
        PRECISION = 8  # Precision for rounding float coordinates
        for tile in tiles:
            if not hasattr(tile, 'floatHexVertices'):
                continue
            # Round vertices to avoid floating point inaccuracies with Shapely
            pts = [(round(p[0], PRECISION), round(p[1], PRECISION)) for p in tile.floatHexVertices]
            if len(pts) < 3:  # Need at least 3 points for a polygon
                continue
            polys.append(Polygon(pts))

        if not polys:
            return [], [], None  # No valid polygons created

        # Merge all tile polygons into one (potentially multi-)polygon
        merged = unary_union(polys)
        return self.extractRings(merged)

    @staticmethod
    def extractRings(merged):
        """Extracts exterior and interior coordinate rings from a Shapely Polygon or MultiPolygon."""
        ext, inter, poly_obj = [], [], None
        if not merged or merged.is_empty:
            return [], [], None

        def _extract(polygon):
            # Extracts coordinates, rounding to integers for drawing
            extracted = [(int(round(p[0])), int(round(p[1]))) for p in polygon.exterior.coords]
            ind = []
            for r in polygon.interiors:
                if len(r.coords) > 2:  # Ensure interior ring is valid
                    ind.append([(int(round(p[0])), int(round(p[1]))) for p in r.coords])
            return extracted, ind

        if isinstance(merged, Polygon):
            if merged.exterior:
                e, i = _extract(merged)
                poly_obj = merged  # Keep the original Shapely object
                ext.append(e)
                inter.extend(i)
        elif isinstance(merged, MultiPolygon):
            valid_polys = []
            for poly in merged.geoms:
                if isinstance(poly, Polygon) and poly.exterior:
                    e, i = _extract(poly)
                    ext.append(e)
                    inter.extend(i)
                    valid_polys.append(poly)  # Collect valid Shapely polygons
            if valid_polys:
                # Store as MultiPolygon if multiple, otherwise single Polygon
                poly_obj = MultiPolygon(valid_polys) if len(valid_polys) > 1 else valid_polys[0]

        return ext, inter, poly_obj

    def spawnResources(self, info):
        """Spawns resources within the territory based on configuration."""
        if not info or not hasattr(info, 'resourceTypes'):
            return

        for res_type in getattr(info, 'resourceTypes', []):
            spawnable_tiles = []
            spawn_rate = 0.0
            # Get tiles suitable for this resource type
            if hasattr(info, 'getSpawnableTiles'):
                spawnable_tiles = info.getSpawnableTiles(res_type, self.unusedSpawningTiles)
            # Get the spawn rate for this resource
            if hasattr(info, 'spawnRates'):
                spawn_rate = info.spawnRates.get(res_type, 0.0)

            # Calculate number to spawn based on rate and available tiles
            num_to_spawn = int(len(spawnable_tiles) * spawn_rate + random.random())

            if spawnable_tiles and num_to_spawn > 0:
                # Ensure we don't try to sample more tiles than available
                k = min(num_to_spawn, len(spawnable_tiles))
                try:
                    # Randomly select tiles for spawning
                    selected_tiles = random.sample(spawnable_tiles, k)
                    for tile in selected_tiles:
                        # Double-check tile is still available before spawning
                        if tile in self.unusedSpawningTiles:
                            self.containedResources.append(Resource(tile, res_type, None))  # Add Resource object
                            self.unusedSpawningTiles.remove(tile)  # Mark tile as used
                except ValueError:
                    # random.sample throws ValueError if k > len(spawnable) - should be prevented by min() but handle defensively
                    pass

    def spawnHarbors(self, info):
        possible_tiles = [t for t in self.coastTiles if t in self.unusedSpawningTiles and not t.isMountain]
        spawn_chance = info.harborSpawnRate * len(possible_tiles)

        print("Starting coastal search")
        coastLines = []
        currentCoastID = 0
        while possible_tiles:
            queue = [random.choice(possible_tiles)]
            coastLines.append([queue[0]])
            while queue:
                for adj in queue[0].adjacent:
                    if adj in possible_tiles:
                        coastLines[currentCoastID].append(adj)
                        possible_tiles.remove(adj)
                        queue.append(adj)
                queue.pop(0)
            currentCoastID += 1

        justUsedTiles = []
        print(f"Found {len(coastLines)} coastlines")
        for coastLine in coastLines:
            for oceanID in set([tile.connectedOceanID for tile in coastLine]):
                correspondingUsableTiles = [t for t in coastLine if (t.connectedOceanID == oceanID and t not in justUsedTiles)]
                chosenTile = random.choice(correspondingUsableTiles)
                self.harbors.append(Harbor(chosenTile, (random.random() < spawn_chance)))
                self.unusedSpawningTiles.remove(chosenTile)
                justUsedTiles.append(chosenTile)

    def update_reachable_harbors(self):
        """Updates the dictionary mapping local harbors to harbors reachable via trade routes."""
        self.reachableHarbors.clear()  # Reset the map
        for local_harbor in self.harbors:
            self.reachableHarbors[local_harbor] = list(local_harbor.tradeRouteObjects.keys())

    def drawInternal(self, target_surf, target_debug_surf):
        """Draws static territory elements (borders, resources, harbors) onto TileHandler's surfaces."""
        if target_surf is None or target_debug_surf is None:
            # Don't draw if surfaces aren't provided (e.g., during init)
            return

        # Draw a small circle at the territory's logical center on the debug overlay
        if hasattr(self.cols, 'dark'):
            pygame.draw.circle(target_debug_surf, self.cols.dark, self.centerPos, 5, 2)

        # Draw territory borders on the main static surface
        border_col = setOpacity(self.cols.dark, 180)  # Semi-transparent dark color
        width = 3  # Border thickness

        # Draw exterior borders
        for border in self.exteriors:
            if len(border) > 1:  # Need at least 2 points to draw lines
                pygame.draw.lines(target_surf, border_col, True, border, width=width)  # True = closed loop

        # Draw interior borders (holes)
        for border in self.interiors:
            if len(border) > 1:
                pygame.draw.lines(target_surf, border_col, True, border, width=width)

        # Draw static appearance of resources and harbors onto the main surface
        # Assumes Resource and Harbor classes have their own draw methods
        for resource in self.containedResources:
            resource.draw(target_surf)
        for harbor in self.harbors:
            harbor.draw(target_surf)

    def drawCurrent(self, s, mx, my, click):
        """Draws dynamic elements (like hover effects) onto the provided surface `s`."""
        hover = False
        # Check for hover using Shapely if available
        if SHAPELY_AVAILABLE and self.polygon:
            # Create a Shapely Point for the mouse position
            mouse_point = Point(mx, my)
            # Check if the territory's polygon contains the mouse point
            hover = self.polygon.contains(mouse_point)

        # If hovering, draw the highlight effect
        if hover:
            fill_color = setOpacity(self.territoryCol, 60)  # Semi-transparent territory color
            line_color = setOpacity(self.territoryCol, 200)  # More opaque border color
            width = 4  # Highlight border thickness

            # Draw filled polygon and thicker lines for exterior borders
            for border in self.exteriors:
                if len(border) > 2:  # Need at least 3 points for a polygon
                    pygame.draw.polygon(s, fill_color, border)
                if len(border) > 1:
                    pygame.draw.lines(s, line_color, True, border, width=width)

            # Draw thicker lines for interior borders (holes)
            for border in self.interiors:
                if len(border) > 1:
                    pygame.draw.lines(s, line_color, True, border, width=width)

            for src_harbor, reachable_harbors in self.reachableHarbors.items():
                for target_harbor in reachable_harbors:
                    src_harbor.drawRoute(s, target_harbor, self.cols.debugRed)

            if click and len(self.ships) == 0:
                self.ships.append(Ship(self.harbors[0].tile, "fluyt", ShipInfo, ResourceInfo))
                self.ships[0].beginVoyage(self.harbors[0].tradeRoutesPoints[list(self.harbors[0].tradeRoutesPoints.keys())[0]])

        if len(self.ships) != 0:
            self.ships[0].draw(s)
            self.ships[0].move()
