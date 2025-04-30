import pygame
import math
import time
from text import drawText
from fontDict import fonts
from generation import TileHandler

pygame.init()

# ---------------- Setting up the screen, assigning some global variables, and loading text fonts
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
clock = pygame.time.Clock()
fps = 60
scaleDownFactor = 1
screen_width = int(screen.get_width() / scaleDownFactor)
screen_height = int(screen.get_height() / scaleDownFactor)
screen_center = [screen_width / 2, screen_height / 2]
screen2 = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT = pygame.Surface((screen_width, screen_height)).convert_alpha()
screenT.set_alpha(100)
screenUI = pygame.Surface((screen_width, screen_height)).convert_alpha()
timer = 0
shake = [0, 0]
shake_strength = 3
montserratRegularAdaptive = fonts[f"regular{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratExtralightAdaptive = fonts[f"extralight{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratBoldAdaptive = fonts[f"bold{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]
montserratThinAdaptive = fonts[f"thin{int(25 / (scaleDownFactor ** (1 / 1.5)))}"]

montserratRegular15 = fonts[f"regular{int(15 / (scaleDownFactor ** (1 / 1.5)))}"]


class Cols:
    oceanBlue = [59, 95, 111]
    oceanGreen = [73, 120, 122]
    lightOceanGreen = [86, 142, 143]
    oceanFoam = [148, 182, 180]

    sandyBrown = [172, 146, 95]
    darkSandyBrown = [159, 143, 91]

    oliveGreen = [95, 115, 84]
    darkOliveGreen = [54, 64, 57]

    mountainBlue = (83, 78, 90)
    darkMountainBlue = [42, 40, 52]

    light = [220, 216, 201]
    dark = [18, 22, 27]
    accentOrange = [155, 110, 83]

    debugRed = [255, 96, 141]


# Defining some more variables to use in the game loop
oscillating_random_thing = 0
ShakeCounter = 0
toggle = True
click = False

tileSize = 6
TH = TileHandler(screen_width, screen_height, tileSize, Cols, 0.51, 0.54, 100, font=montserratRegular15)

# ---------------- Main Game Loop
last_time = time.time()
running = True
while running:

    # ---------------- Reset Variables and Clear screens
    mx, my = pygame.mouse.get_pos()
    mx, my = mx / scaleDownFactor, my / scaleDownFactor
    screen.fill(Cols.oceanBlue)
    screen2.fill(Cols.oceanBlue)
    screenT.fill((0, 0, 0, 0))
    screenUI.fill((0, 0, 0, 0))
    dt = time.time() - last_time
    dt *= fps
    last_time = time.time()
    timer -= 1 * dt
    shake = [0, 0]
    oscillating_random_thing += math.pi / fps * dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                toggle = not toggle
        if event.type == pygame.KEYUP:
            pass

    TH.draw(screen2, mx, my, showArrows=False, showDebugOverlay=False, showWaterLand=False)

    # ---------------- Updating Screen
    if toggle:
        items = {round(clock.get_fps()): None, }
        for _, label in enumerate(items.keys()):
            string = str(label)
            if items[label] is not None:
                string = f"{items[label]}: " + string
            drawText(screenUI, Cols.debugRed, montserratRegularAdaptive, 5, screen_height - (30 + 25 * _) / (scaleDownFactor ** (1 / 1.8)), string, Cols.dark, int(3 / scaleDownFactor) + int(3 / scaleDownFactor) < 1, antiAliasing=False)
        pygame.mouse.set_visible(False)
        pygame.draw.circle(screenUI, Cols.dark, (mx + 2, my + 2), 7, 2)
        pygame.draw.circle(screenUI, Cols.light, (mx, my), 7, 2)
    screen.blit(pygame.transform.scale(screen2, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenT, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (shake[0], shake[1]))
    screen.blit(pygame.transform.scale(screenUI, (screen_width * scaleDownFactor, screen_height * scaleDownFactor)), (0, 0))
    pygame.display.update()
    clock.tick(fps)
