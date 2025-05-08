import pygame
import math
from calcs import ang, normalize_angle, distance
from text import drawText


class Ship:
    # ships never regen
    # ships can be repaired at harbors for resources (small amounts) proportional to their damage, so it's always cheaper than building a new one
    #    however, repairing also requires tar

    # trading: MS, HP
    # Fluyt: fast, weak
    # Carrack: slow, tanky

    # war: MS, HP, ATK
    # Cutter: fast, weak, medium attack
    # Corsair: fast, medium, high attack

    # trade + war: MS, HP, CARGO, ATK
    # Long Ship: medium, weak, small trade, weak attack
    # Galleon: slow, tanky, large trade, medium attack

    def __init__(self, startingTile, shipType, shipInfo, resourceInfo):
        self.startingTile = startingTile
        self.shipType = shipType

        self.hp = shipInfo.shipHPs[shipType]
        self.currentHP = shipInfo.shipHPs[shipType]
        self.ms = shipInfo.shipMSs[shipType]
        self.currentMS = shipInfo.shipMSs[shipType]
        self.vision = shipInfo.shipVisionRanges[shipType]
        self.currentVision = shipInfo.shipVisionRanges[shipType]
        self.dmg = shipInfo.shipDMGs[shipType]
        self.range = shipInfo.shipAttackRanges[shipType]
        self.cargoCapacity = shipInfo.shipStorageCapacities[shipType]

        self.currentCargo = {resource: 0 for resource in resourceInfo.resourceTypes}

        self.img = pygame.transform.scale(pygame.image.load("assets/UI/Fluyt.png").convert_alpha(), (40, 40))
        self.imgDims = [self.img.get_width(), self.img.get_height()]

        self.a = 0
        self.path = None
        self.currentInd = None
        self.pos = None

        self.points = None

        # from fontDict import fonts
        # self.font =

    def beginVoyage(self, path):
        self.path = path
        self.currentInd = 1
        self.pos = list(path[0])

    def move(self):
        angDiff = normalize_angle(ang(self.pos, self.path[self.currentInd])) - self.a
        self.a += ((angDiff + math.pi) % (2 * math.pi) - math.pi) / 5
        self.pos[0] += math.cos(self.a) * self.currentMS
        self.pos[1] += math.sin(self.a) * self.currentMS

        if distance(self.pos, self.path[self.currentInd]) < 3:
            self.currentInd += 1

    def draw(self, s):
        s.blit(pygame.transform.rotate(self.img, self.a), (self.pos[0] - self.imgDims[0], self.pos[1] - self.imgDims[1]))
        # drawText(s, (255, 200, 200), )


class TradeShip(Ship):
    # opens a mini window when sending out a ship
    # player specifies a priority order for resources
    #       wood    stone    iron   (selects wood)
    #       wood    stone    iron   (wood is greyed out, selects stone)
    #       wood    stone    iron   (wood and stone greyed out, iron is auto-selected)

    # distress signal when being attacked, player can either
    #   scuttle cargo (drop into the sea, enemy claims emtpy trade ship. do this when you think you can claim it back soon)
    #   sink ship (know it's undefendable, don't want them to claim it)
    #   intentionally let it be taken (the riskiest option, if you don't win the fight and reclaim it they get free cargo)
    pass


class Warship(Ship):
    # set center of patrolling area
    # moves towards and trade ships once in range
    #    prefers attacking other warships (use a smaller sensing distance for trade ships than what's shown as visible)
    #    won't finish trade ship if it already started attacking it so it doesn't lose the warship battle
    # retreats to repair at a harbor when reaching a threshold HP
    #    can still attack and be attacked while retreating
    pass


class LongShip(Warship, TradeShip):
    # warship that also carries resources
    pass


class ScoutShip(Ship):
    # set center of patrolling area
    # acts like a lighthouse in the water (and it can be relocated)
    # very weak, but you can see enemy warships early-on and run
    pass
