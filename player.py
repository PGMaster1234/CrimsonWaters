import pygame
import random
from controlPanel import ShipInfo, ResourceInfo
from ships import Ship
from text import drawText


class PlayerHandler:
    def __init__(self):
        self.players = []

    def add_player(self, player):
        self.players.append(player)


class Player:
    def __init__(self, ip, runtimePort, territory, screenDims, fonts, cols):
        self.ip = ip
        self.runtimePort = runtimePort
        self.territory = territory
        self.screenDims = screenDims
        self.fonts = fonts
        self.cols = cols

        self.surf = pygame.Surface(self.screenDims).convert_alpha()

        self.ships = []
        self.harbors = []
        self.resources = []

        self.resourceStorages = {r: 0 for r in ResourceInfo.resourceTypes}

        self.selectedTerritory = None
        self.selectedTerritoryResetTimer = 0
        self.clickedOnInvalidTerritory = False
        self.visibleTerritoryIDs = set()

    def handleClick(self, click, dt, hovered_territory):
        if self.selectedTerritoryResetTimer > 0:
            self.clickedOnInvalidTerritory = False
        self.selectedTerritoryResetTimer += dt

        clicked_on_a_territory = False

        if hovered_territory and click:
            clicked_on_a_territory = True
            territory = hovered_territory

            if self.selectedTerritory is None:
                if self.selectedTerritoryResetTimer > 0:
                    self.selectedTerritory = territory
                    if self.selectedTerritory:
                        self.visibleTerritoryIDs.add(self.selectedTerritory.id)
                    self.selectedTerritoryResetTimer = -30
            elif territory != self.selectedTerritory:
                if territory in self.selectedTerritory.shortestPathToReachableTerritories:
                    self.selectedTerritoryResetTimer = -30
                    s = Ship(self.selectedTerritory.shortestPathToReachableTerritories[territory][0].tile, "fluyt", ShipInfo, ResourceInfo)
                    s.beginVoyage(self.selectedTerritory.shortestPathToReachableTerritories[territory][3])
                    self.ships.append(s)
                    self.selectedTerritory = None
                    self.visibleTerritoryIDs.clear()
                else:
                    self.selectedTerritoryResetTimer = -30
                    self.clickedOnInvalidTerritory = True
                    self.selectedTerritory = None
                    self.visibleTerritoryIDs.clear()

        if click and not clicked_on_a_territory:
            self.selectedTerritory = None
            self.visibleTerritoryIDs.clear()

    def update(self, dt):
        for ship in self.ships:
            ship.move(dt)

    def draw(self, s, screenUI, debug):
        self.surf.fill((0, 0, 0, 0))
        for ship in self.ships:
            ship.draw(self.surf, debug)
        if self.clickedOnInvalidTerritory:
            shakeStrength = 2
            shake = (random.randint(-shakeStrength, shakeStrength), random.randint(-shakeStrength, shakeStrength))
            drawText(screenUI, self.cols.crimson, self.fonts['150'], self.screenDims[0] / 2 + shake[0], self.screenDims[1] / 2 + shake[1], "INVALID ORDER", self.cols.dark, 3, antiAliasing=False, justify='center', centeredVertically=True)
        s.blit(self.surf, (0, 0))
