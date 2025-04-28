import pygame
import math
from calcs import distance
from calcs import normalize_angle
from calcs import brightness
from calcs import linearGradient
import random
import copy


class Shockwave:
    # Function for initializing the shockwave object with its parameters
    def __init__(self, sx, sy, duration, size, max_size, width, color, color2, shadow):
        self.x = sx
        self.y = sy
        self.duration = duration
        self.size = size
        self.max_size = max_size
        self.width = width
        self.color = color
        self.color2 = color2
        self.shadow = shadow

    # This function is run each frame and increases the size of the shockwave but decreases the width
    def expand(self):
        self.size += (self.max_size - self.size) / (10 * self.duration)
        if self.size / self.max_size < 0.8:
            self.width -= 0.03
        else:
            self.width -= 0.1

    # This function draws the shockwave (and its shadow) on the screen
    def draw(self, s):
        pygame.draw.circle(s, self.color2, (self.x - self.shadow, self.y + self.shadow), self.size, int(self.width))
        pygame.draw.circle(s, self.color, (self.x, self.y), self.size, int(self.width))


class Particle:
    # Function for initializing the particle object with its parameters
    def __init__(self, px, py, x_vel, y_vel, color, color2, size, timer, decay=0, gravity=0, bounciness=0):
        self.x = px
        self.y = py
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.color = color
        self.color2 = color2
        self.size = size
        self.timer = timer
        self.decay = decay
        self.gravity = gravity
        self.bounciness = bounciness

    # Function for drawing the particle on the screen
    def draw(self, s, scr):
        pygame.draw.circle(s, self.color2, (self.x - self.size / 4 - scr[0], self.y + self.size / 4 - scr[1]), self.size)
        pygame.draw.circle(s, self.color, (self.x - scr[0], self.y - scr[1]), self.size)

    # Function that is called every frame for moving the particle
    def move(self, dt):
        self.x += self.x_vel * dt
        self.y += self.y_vel * dt
        self.y_vel += self.gravity * dt
        self.size -= self.decay * dt
        self.timer -= dt


class Bullet:
    def __init__(self, x, y, vel, angle, dmg, size, col, t, sample_rate=60):
        self.x = x
        self.y = y
        self.v = vel
        self.a = normalize_angle(angle)
        self.dmg = dmg
        self.s = size
        self.c = col
        self.timer = t
        self.sample_counter = 0
        self.sample_rate = sample_rate
        self.walls = []

    def move(self, dt):
        self.x += self.v * math.cos(self.a) * dt
        self.y += self.v * math.sin(self.a) * dt

    def update(self, ts, tile_s, dt):
        self.timer -= 1 * dt
        self.sample_counter += 1
        if self.sample_counter > self.sample_rate:
            self.walls = []
            for ti in ts:
                if distance((ti.centerx, ti.centery), (self.x, self.y)) / (self.v / 1.5) < tile_s:
                    self.walls.append(ti)
            self.sample_counter = 0

    def collideWithWalls(self):
        for w in self.walls:
            if w.rect.collidepoint((self.x, self.y)):
                return True

    def draw(self, s, scr):
        pygame.draw.circle(s, self.c, (self.x - scr[0], self.y - scr[1]), self.s)


class PhysicsParticle:
    # This initializes the bullet object with its parameters
    def __init__(self, r, x, y, vel, angle, ricochet, col, dmg, t):
        self.r = r
        self.rect = pygame.rect.Rect(x - self.r, y - self.r, self.r * 2, self.r * 2)
        self.vel = vel
        self.a = normalize_angle(angle)
        self.ric = ricochet
        self.has_bounced = False
        self.col = col
        self.dmg = dmg
        self.sample_counter = 0
        self.sample_rate = 2
        self.walls = []
        self.movement = [1, 0]
        self.dir = [1, 1]
        self.timer = t * 60

    def update(self, ts, tile_s, delta_t):
        self.has_bounced = False
        self.timer -= 1 * delta_t
        self.sample_counter += 1
        if self.sample_counter > self.sample_rate:
            self.walls = []
            for ti in ts:
                if distance((ti.centerx, ti.centery), (self.rect.centerx, self.rect.centery)) / (self.vel / 1.5) < tile_s:
                    self.walls.append(ti)
            self.sample_counter = 0

    def move(self, delta_t):
        if self.movement[0] == 1:
            self.rect.x += self.vel * math.cos(self.a) * delta_t * self.dir[0]
            for w in self.walls:
                if w[0].colliderect(self.rect):
                    if not self.has_bounced:
                        self.ric -= 1
                        self.has_bounced = True
                    if math.cos(self.a) * self.dir[0] > 0:
                        self.dir[0] *= -1
                        self.rect.right = (w.left - self.vel)
                    elif math.cos(self.a) * self.dir[0] < 0:
                        self.dir[0] *= -1
                        self.rect.left = (w.right + self.vel)
            self.rect.y += self.vel * math.sin(self.a) * delta_t * self.dir[1]
            for w in self.walls:
                if w[0].colliderect(self.rect):
                    if not self.has_bounced:
                        self.ric -= 1
                        self.has_bounced = True
                    if math.sin(self.a) * self.dir[1] > 0:
                        self.dir[1] *= -1
                        self.rect.bottom = (w.top - self.vel)
                    elif math.sin(self.a) * self.dir[1] < 0:
                        self.dir[1] *= -1
                        self.rect.top = (w.bottom + self.vel)
        if self.ric < 0 or self.timer <= 0:
            return True

    def draw(self, s, scr, show_rects, show_hitbox):
        if show_rects:
            for w in self.walls:
                pygame.draw.rect(s, (200, 100, 100), (w.x - scr[0], w.y - scr[1], w.w, w.h))
        if show_hitbox:
            pygame.draw.rect(s, self.col, (self.rect.x - scr[0], self.rect.y - scr[1], self.rect.w, self.rect.h), 1)
        pygame.draw.circle(s, self.col, (self.rect.centerx - scr[0], self.rect.centery - scr[1]), self.r)


class Glow:
    # Function for initializing the particle object with its parameters
    def __init__(self, px, py, x_vel, y_vel, color, color2, glow_col, size, min_size, decay, gravity, flicker):
        self.x = px
        self.y = py
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.color = color
        self.color2 = color2
        self.glow_col = glow_col
        self.size = size
        self.min_size = min_size
        self.decay = decay
        self.gravity = gravity
        self.offset = random.uniform(0, 2 * math.pi)
        self.flicker = flicker

    # Function for drawing the particle on the screen
    def draw(self, s, oscillating_thing):
        if self.color2 is not None:
            pygame.draw.circle(s, self.color2, (self.x - self.size / 4, self.y + self.size / 4), self.size)
        pygame.draw.circle(s, self.color, (self.x, self.y), self.size)

        surf = pygame.Surface((self.size * 4 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker * 2, self.size * 4 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker * 2))
        if self.glow_col is None:
            pygame.draw.circle(surf, brightness(self.color, 0.1), (self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker, self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker), self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker)
        else:
            pygame.draw.circle(surf, self.glow_col, (self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker, self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker), self.size * 2 + (math.sin(oscillating_thing + self.offset) + 1) * self.flicker)
        surf.set_colorkey((0, 0, 0))
        s.blit(surf, (self.x - self.size * 2 - (math.sin(oscillating_thing + self.offset) + 1) * self.flicker, self.y - self.size * 2 - (math.sin(oscillating_thing + self.offset) + 1) * self.flicker), special_flags=pygame.BLEND_RGB_ADD)

    # Function that is called every frame for moving the particle
    def move(self, dt):
        self.x += self.x_vel * dt
        self.y += self.y_vel * dt
        self.y_vel += self.gravity * dt
        self.size -= self.decay * dt
        if self.size < self.min_size:
            return True


class Torch:
    def __init__(self, x, y, torchImg, flameColsDark2Light, smoke_cols_Dark2Light, ember_cols_Dark2Light, torchImgScale=(14, 22), numFlames=6, innerFlameSize=10, flameSizeMaxSpacing=30, flameGrowShrink=10, flameFreq=0.1, smoke_rate=0.5, smoke_grow=0.1, smoke_rad=2, smoke_rise=0.01, ember_rate=0.4, ember_shrink=0.08, ember_rad=3, emberStartingVels=(2, 3), glowingEmbers=True, gravity=0.1, wind=0.05, wind_favor=1.2):
        self.x = x
        self.y = y

        self.numFlames = numFlames
        self.innerFlameSize = innerFlameSize
        self.flameSizeMaxSpacing = flameSizeMaxSpacing
        self.flameGrowShrink = flameGrowShrink
        self.flameFreq = flameFreq

        self.smoke_rate = smoke_rate
        self.smoke_grow = smoke_grow
        self.smoke_rad = smoke_rad
        self.smoke_rise = smoke_rise

        self.ember_rate = ember_rate
        self.ember_shrink = ember_shrink
        self.ember_rad = ember_rad
        self.emberStartingVels = emberStartingVels
        self.glowingEmbers = glowingEmbers

        self.gravity = gravity
        self.wind = wind
        self.wind_favor = wind_favor

        self.smoke_cols = smoke_cols_Dark2Light
        self.ember_cols = ember_cols_Dark2Light
        self.flameColsDark2Light = flameColsDark2Light

        self.smokes = []
        self.embers = []
        self.flames = []
        self.createFlames()

        self.torchImg = pygame.transform.scale(torchImg.convert_alpha(), torchImgScale)
        self.torchSur = pygame.Surface((self.torchImg.get_width(), self.torchImg.get_height()), pygame.SRCALPHA)
        self.torchSur.blit(self.torchImg, (0, 0))
        self.torchSurCenteringOffset = [-int(self.torchSur.get_width() / 1.7), int(self.torchSur.get_height() / 16)]

    def createFlames(self):
        startingOffset = random.uniform(0, math.pi * 2)
        for i in range(self.numFlames):
            # Fancy scaling
            quadratic_scale = ((i / (self.numFlames - 1)) ** 4 + (i / (self.numFlames - 1)) ** 1.2) / 2
            size = self.innerFlameSize + quadratic_scale * self.flameSizeMaxSpacing
            # Pass the flame's size and index, so each flame can manage its own size and behavior
            self.flames.append(Flame(self.x, self.y, size, self.flameGrowShrink, self.flameFreq, self.flameColsDark2Light, i, self.numFlames, startingOffset))

    def spawn(self):
        if random.uniform(0, 1) < self.smoke_rate:
            self.smokes.append(Smoke(self.x, self.y, random.uniform(self.smoke_rad * 0.5, self.smoke_rad * 1.2), self.smoke_grow, self.smoke_cols, self.smoke_rise, self.wind, self.wind_favor, random.uniform(1, 2)))
        if random.uniform(0, 1) < self.ember_rate:
            newEmber = Ember(self.x, self.y, random.uniform(self.ember_rad * 0.5, self.ember_rad * 1.2), self.ember_shrink, self.ember_cols, self.gravity, self.wind, self.wind_favor, self.emberStartingVels, random.uniform(1, 2))
            if self.glowingEmbers:
                newEmber.flame = Flame(newEmber.x, newEmber.y, newEmber.rad * 2, newEmber.rad, self.flameFreq, self.flameColsDark2Light, 0, 1, random.uniform(0, math.pi * 2))
            self.embers.append(newEmber)

    def update(self, dt, fps):
        for sm in reversed(self.smokes):
            if sm.update(dt, fps):
                self.smokes.remove(sm)
        for em in reversed(self.embers):
            if em.update(dt, fps):
                self.embers.remove(em)
        for flm in self.flames:
            flm.x = self.x
            flm.y = self.y
            flm.update(dt)

    def draw(self, s, scr):
        s.blit(self.torchSur, (self.x + self.torchSurCenteringOffset[0], self.y + self.torchSurCenteringOffset[1]))
        for flm in reversed(self.flames):
            flm.draw(s, scr)
        for sm in self.smokes:
            sm.draw(s, scr)
        for em in self.embers:
            em.draw(s, scr)


class Flame:
    def __init__(self, x, y, size, growShrink, freq, colRange, index, totalFlameCount, startingOffset):
        self.x = x
        self.y = y
        self.size = size
        self.growShrink = growShrink
        self.freq = freq
        self.colRange = colRange
        self.index = index  # track which ring this is (to adjust transparency, etc.)
        self.totalFlameCount = totalFlameCount
        self.t = random.uniform(0, (math.pi * 2) / freq / 400) + startingOffset

        # Calculate opacity of the ring
        maxOpacity = 50
        minOpacity = 5
        self.alpha = maxOpacity - (maxOpacity - minOpacity) * (self.index / self.totalFlameCount)

    def update(self, dt):
        self.t += dt / (math.pi * 2) * self.freq

    def draw(self, s, scr):
        func = (math.sin(3 * self.t) + math.sin(7 * self.t)) / 4

        color_weight = ((func + 0.5) + (self.index / self.totalFlameCount)) / 2 # Normalize "func" to [0, 1] to get the interpolation factor, also adjust hue based on ring index
        color = linearGradient(self.colRange, color_weight)

        # Calculate the oscillating size of the flame
        oscillating_size = self.size + self.growShrink * func

        # Create a surface with the appropriate transparency (with enough space for the full circle)
        surface_size = int(oscillating_size * 2)
        layer_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)

        # Draw the circle in the center of the surface
        pygame.draw.circle(layer_surface, (color[0], color[1], color[2], self.alpha), (surface_size // 2, surface_size // 2), oscillating_size)

        # Blit the layer onto the main screen surface with the adjusted rect
        s.blit(layer_surface, layer_surface.get_rect(center=(self.x - scr[0], self.y - scr[1])))


class Smoke:
    def __init__(self, x, y, rad, grow, cols, rise, wind, wind_favor, timer):
        self.x = x
        self.y = y
        self.vel = [0, 0]
        self.rad = rad
        self.grow = grow
        self.cols = cols
        self.rise = rise
        self.wind = wind
        self.wind_favor = wind_favor
        self.timer = timer
        self.maxTimer = copy.deepcopy(timer)

    def update(self, dt, fps):
        self.vel[0] += random.uniform(-self.wind, self.wind_favor * self.wind) * dt
        self.vel[1] += random.uniform(-4 * self.rise, self.rise) * dt
        self.x += self.vel[0] * dt
        self.y += self.vel[1] * dt
        self.rad += self.grow * dt
        self.timer -= dt / fps
        if self.timer < 0:
            return True

    def draw(self, s, scr):
        pygame.draw.circle(s, linearGradient(self.cols, self.timer / self.maxTimer), [self.x - scr[0], self.y - scr[1]], self.rad + math.sin(self.timer))


class Ember:
    def __init__(self, x, y, rad, shrink, cols, grav, wind, wind_favor, emberStartingVels, timer, flame=None):
        self.x = x
        self.y = y
        self.vel = [random.uniform(-emberStartingVels[0], emberStartingVels[0]), random.uniform(-emberStartingVels[1], emberStartingVels[1] / 3)]
        self.rad = rad
        self.shrink = shrink
        self.cols = cols
        self.grav = grav
        self.wind = wind
        self.wind_favor = wind_favor
        self.emberStartingVels = emberStartingVels
        self.timer = timer
        self.maxTimer = copy.deepcopy(timer)
        self.flame = flame

    def update(self, dt, fps):
        self.vel[0] += random.uniform(-self.wind, self.wind_favor * self.wind) * dt
        self.vel[1] += random.uniform(-self.grav, 4 * self.grav) * dt
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95
        self.x += self.vel[0]
        self.y += self.vel[1]
        self.rad -= self.shrink * dt
        self.timer -= dt / fps
        if self.flame is not None:
            self.flame.update(dt)
            self.flame.x, self.flame.y = self.x, self.y
            self.flame.size = int(self.rad * 2 + 2)
        if self.timer < 0 or self.rad < 0:
            return True

    def draw(self, s, scr):
        pygame.draw.circle(s, linearGradient(self.cols, self.timer / self.maxTimer), [self.x - scr[0], self.y - scr[1]], self.rad + math.sin(self.timer))
        if self.flame is not None:
            self.flame.draw(s, scr)


class Spark:
    # This function initializes the spark object by taking in parameters
    def __init__(self, sx, sy, vel, color, color2, size, angle, decay, speed_decay, rotation, gravity, length):
        self.x = sx
        self.y = sy
        self.vel = vel
        self.color = color
        self.color2 = color2
        self.size = size
        self.angle = angle
        self.decay = decay
        self.gravity = gravity
        self.rotation = rotation
        self.speed_decay = speed_decay
        self.length = length

    # This function is used for creating the vertices of the spark, in the shape of a diamond, using parameters given
    # when the object was initialized such as the x, y, size, length, and angle
    # This function then uses those points to draw the polygon on the screen
    # The function also draws a shadow below the diamond using the points and the "shadow" parameter
    def draw(self, s, shifted_scroll):
        points = [(self.x + (self.size * self.length / 3 * math.cos(self.angle)) - self.size - shifted_scroll[0], self.y + (self.size * 2 * math.sin(self.angle)) + self.size - shifted_scroll[1]), (self.x + (self.size * math.cos(self.angle + math.pi / 2)) - self.size - shifted_scroll[0], self.y + (self.size * math.sin(self.angle + math.pi / 2)) + self.size - shifted_scroll[1]),
                  (self.x + (self.size * 2 * self.length / 3 * math.cos(self.angle + math.pi)) - self.size - shifted_scroll[0], self.y + (self.size * 3 * math.sin(self.angle + math.pi)) + self.size - shifted_scroll[1]), (self.x + (self.size * math.cos(self.angle - math.pi / 2)) - self.size - shifted_scroll[0], self.y + (self.size * math.sin(self.angle - math.pi / 2)) + self.size - shifted_scroll[1])]
        pygame.draw.polygon(s, self.color2, points)
        points = [(self.x + (self.size * self.length / 3 * math.cos(self.angle)) - shifted_scroll[0], self.y + (self.size * 2 * math.sin(self.angle)) - shifted_scroll[1]), (self.x + (self.size * math.cos(self.angle + math.pi / 2)) - shifted_scroll[0], self.y + (self.size * math.sin(self.angle + math.pi / 2)) - shifted_scroll[1]),
                  (self.x + (self.size * 2 * self.length / 3 * math.cos(self.angle + math.pi)) - shifted_scroll[0], self.y + (self.size * 3 * math.sin(self.angle + math.pi)) - shifted_scroll[1]), (self.x + (self.size * math.cos(self.angle - math.pi / 2)) - shifted_scroll[0], self.y + (self.size * math.sin(self.angle - math.pi / 2)) - shifted_scroll[1])]
        pygame.draw.polygon(s, self.color, points)

    # This function uses the angle, decay, gravity, rotation, and speed_decay parameters to move the points
    def move(self, t):
        self.x += self.vel * math.cos(self.angle) * t
        self.y += self.gravity * t
        self.y += self.vel * math.sin(self.angle) * t
        self.size -= self.decay * t
        self.rotation = 1 / (self.size * 20)
        if self.vel > 0:
            self.vel -= self.speed_decay * t
        if 3 * math.pi / 2 > self.angle > math.pi / 2:
            self.angle -= self.rotation * t
        if 5 * math.pi / 2 > self.angle > 3 * math.pi / 2:
            self.angle += self.rotation * t
        if math.pi / 2 > self.angle > 0:
            self.angle += self.rotation * t
