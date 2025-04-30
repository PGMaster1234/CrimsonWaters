import pygame
import math
import heapq
import itertools


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
        if self.resourceType == 'wood':
            pygame.draw.polygon(s, (115, 80, 32), self.tile.hex)
        if self.resourceType == 'stone':
            pygame.draw.polygon(s, (123, 133, 150), self.tile.hex)
        if self.resourceType == 'iron':
            pygame.draw.polygon(s, (106, 85, 125), self.tile.hex)
        if self.resourceType == 'pine':
            pygame.draw.polygon(s, (63, 105, 51), self.tile.hex)
        if self.resourceType == 'amber':
            pygame.draw.polygon(s, (184, 102, 48), self.tile.hex)

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
        self.tradeRoutes = {}
        # add a counter for tie-breaking in the heap
        self._heap_counter = itertools.count()

    def generateAllRoutes(self, other_harbors, waterTilesInOcean):
        """
        Single-source multi-target Dijkstra to find the shortest paths through water
        from this harbor to all other specified harbors within the same ocean.
        Stores results in self.tradeRoutes.
        """
        self.tradeRoutes = {}  # Reset routes for this search
        counter = itertools.count()  # Tie-breaker for heap queue

        # Find water tiles adjacent to this harbor (the starting points)
        start_water_neighbors = {w for w in self.tile.adjacent if w in waterTilesInOcean}
        if not start_water_neighbors:
            # print(f"Debug: Harbor {self.tile.grid_x},{self.tile.grid_y} has no adjacent water.")
            return  # Cannot pathfind if no water access

        # Map water tiles adjacent to target harbors back to the harbor object
        target_water_map = {}
        target_harbor_set = set()  # Track which target harbors we still need to find
        for h in other_harbors:
            is_target = False
            for w in h.tile.adjacent:
                if w in waterTilesInOcean:
                    target_water_map[w] = h  # Map water tile -> target harbor
                    is_target = True
            if is_target:
                target_harbor_set.add(h)  # Add harbor to the set of targets

        if not target_harbor_set:
            # print(f"Debug: No reachable target harbors for {self.tile.grid_x},{self.tile.grid_y}.")
            return  # No targets to pathfind to

        # Dijkstra Initialization
        frontier = []  # Min-heap: (g_score, tie_breaker, current_water_tile)
        came_from = {}  # {water_tile: previous_water_tile_or_source_harbor_tile}
        g_score = {w: float('inf') for w in waterTilesInOcean}  # Cost from start

        # Initialize frontier with all starting water neighbors
        for start_neighbor in start_water_neighbors:
            g_score[start_neighbor] = 0  # Cost to enter water is 0
            heapq.heappush(frontier, (0, next(counter), start_neighbor))
            # Mark that these water tiles came directly from the starting harbor's land tile (conceptual parent)
            came_from[start_neighbor] = self.tile

        # Dijkstra Loop
        while frontier and target_harbor_set:  # Continue while targets remain
            g_curr, _, current_water_tile = heapq.heappop(frontier)

            # Skip if we found a shorter path already to this specific water tile
            if g_curr > g_score.get(current_water_tile, float('inf')):
                continue

            # Check if we reached a water tile adjacent to a target harbor we haven't finished finding
            if current_water_tile in target_water_map:
                target_harbor = target_water_map[current_water_tile]
                if target_harbor in target_harbor_set:  # Check if we still need this target
                    # --- Path Found to target_harbor ---
                    path = []
                    temp = current_water_tile
                    # Reconstruct path back to the source harbor's conceptual land tile
                    while temp != self.tile:  # Stop when we trace back to the land tile
                        path.append(temp)
                        if temp not in came_from:  # Error check for reconstruction
                            print(f"Error reconstructing path for {target_harbor}")
                            path = None  # Indicate error
                            break
                        temp = came_from[temp]  # Move to the previous tile in the path

                    if path:  # If reconstruction succeeded
                        # Path is currently water tiles from target adj -> source adj. Reverse it.
                        self.tradeRoutes[target_harbor] = path[::-1]  # print(f"Found path: {self.tile.grid_x},{self.tile.grid_y} -> {target_harbor.tile.grid_x},{target_harbor.tile.grid_y} len={len(path)}")

                    target_harbor_set.remove(target_harbor)  # Mark this harbor as found

            # Explore Neighbors of the current water tile
            for neighbor in current_water_tile.adjacent:
                # Must be a valid water tile within the current ocean set for pathfinding
                if neighbor not in waterTilesInOcean:
                    continue

                # Cost to move from current to neighbor (assume cost = 1 for adjacent tiles)
                tentative_g = g_curr + 1

                # If this path to the neighbor is shorter than any previous path found
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_water_tile  # Record the path
                    g_score[neighbor] = tentative_g
                    # Add neighbor to the frontier (priority queue)
                    heapq.heappush(frontier, (tentative_g, next(counter), neighbor))

        # print(f"Finished search for {self.tile.grid_x},{self.tile.grid_y}. Found routes to {len(self.tradeRoutes)} harbors.")

    @staticmethod
    def heuristic(a, b):
        # straight‐line (Euclidean) distance
        return math.hypot(a.x - b.x, a.y - b.y)

    def draw(self, s):
        pygame.draw.polygon(s, (255, 0, 0), self.tile.hex)

    def drawRoute(self, s, otherHarbor):
        if len(self.tradeRoutes[otherHarbor]) > 1:
            pygame.draw.lines(s, (127, 63, 63), False, [tile.center for tile in self.tradeRoutes[otherHarbor]], 3)


class DefensePost:
    # acts like a land-based, stationary warship
    # best placed near choke points and near the coast
    # land invasion defense buff
    pass
