import pygame
import random
import sys
import torch
import torch.nn as nn


# Settings
WIDTH, HEIGHT    = 400, 600
FPS              = 60

GRAVITY          = 0.4
JUMP_FORCE       = -13

MOVE_SPEED       = 10.0
ACCELERATION     = 0.9
FRICTION         = 0.92

SCROLL_THRESH    = HEIGHT // 2

PLATFORM_W       = 70
PLATFORM_H       = 12
PLATFORM_COUNT   = 10
GRID_SIZE        = 20

POPULATION_SIZE  = 30
GENERATIONS      = 10
FRAMES_PER_AGENT = 4000
ELITE_COUNT      = 8
MUTATION_RATE    = 0.12
MUTATION_STRENGTH= 0.25

TRAINING_EPSILON = 0.05

SHOW_BEST_AGENT  = True

# Colors
BG_COLOR        = (255, 255, 255)
GRID_COLOR      = (220, 220, 220)
PLATFORM_COLOR  = (214, 214, 214)
PLAYER_COLOR    = (255, 236, 0)
SCORE_COLOR     = (60, 60, 60)
DEAD_COLOR      = (220, 50, 50)
SPRING_COLOR    = (255, 200, 0)
MOVING_COLOR    = (80, 160, 230)
BREAKING_COLOR  = (180, 120, 60)
OUTLINE_COLOR   = (0, 0, 0)


# Helpers

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_dx(target_x, px):
    dx = target_x - px
    if dx > WIDTH / 2:
        dx -= WIDTH
    elif dx < -WIDTH / 2:
        dx += WIDTH
    return dx


def draw_rounded_rect(surf, color, rect, radius=6):
    pygame.draw.rect(surf, color, rect, border_radius=radius)


def draw_grid(surf, camera_y):
    surf.fill(BG_COLOR)

    offset_y = int(camera_y) % GRID_SIZE

    x = 0
    while x <= WIDTH:
        pygame.draw.line(surf, GRID_COLOR, (x, 0), (x, HEIGHT))
        x += GRID_SIZE

    y = -offset_y
    while y <= HEIGHT:
        pygame.draw.line(surf, GRID_COLOR, (0, y), (WIDTH, y))
        y += GRID_SIZE

# N.N.
#
# Inputs:
#   0  signed horizontal distance to closest-by-x platform above [-1, 1]
#   1  vertical distance to closest-by-x platform above [-1, 1]
#   2  signed horizontal distance to closest-by-x platform below [-1, 1]
#   3  vertical distance to closest-by-x platform below [-1, 1]
#   4  signed horizontal distance to closest platform by vertical distance [-1, 1]
#   5  vertical distance to closest platform by vertical distance [-1, 1]
#   6  signed horizontal distance to second closest platform by vertical distance [-1, 1]
#   7  vertical distance to second closest platform by vertical distance [-1, 1]
#   8  player x position [0, 1]
#   9  player horizontal velocity [-1, 1]
#   10 player vertical velocity [-1, 1]
#   11 has platform above flag
#   12 has platform below flag
#   13 falling flag: 1 if vy > 0
#   14 repeated-platform warning flag: 1 if same platform has been hit at least 3 times in a row
#   15 relative horizontal distance between above-x and below-x targets [-1, 1]
#   16 relative horizontal distance between closest and second-closest targets [-1, 1]
#   17 repeated jump count, normalized to [0, 1]
# Outputs:
#   0 = accelerate left
#   1 = no horizontal acceleration
#   2 = accelerate right

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(18, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 3),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)


class Agent:
    def __init__(self):
        self.net     = Net()
        self.fitness = 0.0

    def copy(self):
        a = Agent()
        a.net.load_state_dict(self.net.state_dict())
        a.fitness = self.fitness
        return a


# Evolutionary

def mutate(agent):
    with torch.no_grad():
        for p in agent.net.parameters():
            mask  = torch.rand_like(p) < MUTATION_RATE
            noise = torch.randn_like(p) * MUTATION_STRENGTH
            p += mask * noise


def crossover(a, b):
    child = a.copy()

    with torch.no_grad():
        pa = dict(a.net.named_parameters())
        pb = dict(b.net.named_parameters())

        for name, param in child.net.named_parameters():
            mask = torch.rand_like(param) > 0.5
            param.copy_(torch.where(mask, pa[name], pb[name]))

    return child


# Platforms

KIND_NORMAL   = 0
KIND_MOVING   = 1
KIND_BREAKING = 2
KIND_SPRING   = 3


class Platform:
    _next_id = 0

    def __init__(self, x, y, kind=KIND_NORMAL):
        self.id = Platform._next_id
        Platform._next_id += 1

        self.x = float(x)
        self.y = float(y)
        self.w = PLATFORM_W
        self.h = PLATFORM_H
        self.kind = kind

        self.broken = False
        self.break_timer = 0

        self.dir   = random.choice([-1, 1])
        self.speed = 1.5

        self.spring_bounced = False

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.w, self.h)

    @property
    def centre_x(self):
        return self.x + self.w / 2

    def update(self):
        if self.kind == KIND_MOVING:
            self.x += self.dir * self.speed

            if self.x < 0 or self.x + self.w > WIDTH:
                self.dir *= -1

        if self.kind == KIND_BREAKING and self.broken:
            self.break_timer += 1

    def draw(self, surf, camera_y):
        dy = self.y - camera_y

        if dy > HEIGHT + 20 or dy < -20:
            return

        color = {
            KIND_NORMAL:   PLATFORM_COLOR,
            KIND_MOVING:   MOVING_COLOR,
            KIND_BREAKING: (200, 100, 50) if self.broken else BREAKING_COLOR,
            KIND_SPRING:   PLATFORM_COLOR,
        }.get(self.kind, PLATFORM_COLOR)

        r = pygame.Rect(int(self.x), int(dy), self.w, self.h)
        draw_rounded_rect(surf, color, r, 4)

        pygame.draw.rect(
            surf,
            tuple(min(c + 60, 255) for c in color),
            pygame.Rect(int(self.x) + 4, int(dy) + 2, self.w - 8, 3),
            border_radius=2,
        )

        if self.kind == KIND_SPRING and not self.spring_bounced:
            draw_rounded_rect(
                surf,
                SPRING_COLOR,
                pygame.Rect(int(self.x) + self.w // 2 - 5, int(dy) - 10, 10, 10),
                3,
            )

        if self.kind == KIND_BREAKING and self.broken:
            cx, cy2 = int(self.x) + self.w // 2, int(dy) + self.h // 2
            pygame.draw.line(surf, (100, 50, 20), (cx - 10, cy2 - 2), (cx + 5, cy2 + 3), 2)
            pygame.draw.line(surf, (100, 50, 20), (cx, cy2 - 3), (cx - 5, cy2 + 4), 2)

        pygame.draw.rect(surf, OUTLINE_COLOR, r, width=2, border_radius=4)


# Platform Gen

def gen_platforms(top_y, difficulty=0):
    x  = random.randint(0, WIDTH - PLATFORM_W)

    lo = 55 + min(difficulty * 5, 45)
    hi = 90 + min(difficulty * 10, 60)

    y  = top_y - random.randint(int(lo), int(hi))

    if difficulty < 3:
        kind = random.choices(
            [KIND_NORMAL, KIND_SPRING],
            weights=[80, 20],
        )[0]
    elif difficulty < 6:
        kind = random.choices(
            [KIND_NORMAL, KIND_MOVING, KIND_SPRING],
            weights=[75, 20, 5],
        )[0]
    else:
        kind = random.choices(
            [KIND_NORMAL, KIND_MOVING, KIND_SPRING],
            weights=[30, 65, 5],
        )[0]

    return [Platform(x, y, kind)]


def gen_breaking(top_y, difficulty=0):
    if difficulty < 3 or random.random() > 0.25:
        return []

    return [
        Platform(
            random.randint(0, WIDTH - PLATFORM_W),
            top_y - random.randint(-40, 40),
            KIND_BREAKING,
        )
    ]


def initial_platforms():
    ps = [Platform(WIDTH // 2 - PLATFORM_W // 2, HEIGHT - 80)]

    for i in range(PLATFORM_COUNT - 1):
        ps.append(
            Platform(
                random.randint(0, WIDTH - PLATFORM_W),
                HEIGHT - 80 - (i + 1) * random.randint(55, 80),
            )
        )

    return ps


# Targeting
def closest_by_x_platforms(px, py, platforms):

    candidates = [p for p in platforms if not p.broken and p.kind != KIND_BREAKING]

    if not candidates:
        dummy_above = Platform(px - PLATFORM_W / 2, py - 80)
        dummy_below = Platform(px - PLATFORM_W / 2, py + 80)
        return dummy_above, dummy_below, False, False

    above = [p for p in candidates if p.y < py]
    below = [p for p in candidates if p.y >= py]

    if above:
        above_target = min(above, key=lambda p: abs(wrap_dx(p.centre_x, px)))
        has_above = True
    else:
        above_target = min(candidates, key=lambda p: abs(p.y - py))
        has_above = False

    if below:
        below_target = min(below, key=lambda p: abs(wrap_dx(p.centre_x, px)))
        has_below = True
    else:
        below_target = min(candidates, key=lambda p: abs(p.y - py))
        has_below = False

    return above_target, below_target, has_above, has_below


def closest_vertical_platforms(px, py, platforms):

    candidates = [p for p in platforms if not p.broken and p.kind != KIND_BREAKING]

    if not candidates:
        dummy1 = Platform(px - PLATFORM_W / 2, py - 80)
        dummy2 = Platform(px - PLATFORM_W / 2, py + 80)
        return dummy1, dummy2

    sorted_candidates = sorted(candidates, key=lambda p: abs(p.y - py))

    closest = sorted_candidates[0]
    second = sorted_candidates[1] if len(sorted_candidates) > 1 else closest

    return closest, second


def find_target_platform_below(px, py, platforms):
    candidates = [p for p in platforms if not p.broken and p.kind != KIND_BREAKING and p.y > py]
    candidates.sort(key=lambda p: p.y)

    if not candidates:
        return None
    else:
        return candidates[0]


def platform_features(platform, px, py):
    dx_raw = wrap_dx(platform.centre_x, px)

    dx = clamp(dx_raw / (WIDTH / 2), -1.0, 1.0)
    dy = clamp((platform.y - py) / HEIGHT, -1.0, 1.0)

    return dx, dy


# Build
def build_inputs(px, py, vx, vy, platforms, repeated_platform_jumps=0):
    above_target, below_target, has_above, has_below = closest_by_x_platforms(px, py, platforms)
    closest_target, second_closest_target = closest_vertical_platforms(px, py, platforms)

    above_dx, above_dy = platform_features(above_target, px, py)
    below_dx, below_dy = platform_features(below_target, px, py)

    closest_dx, closest_dy = platform_features(closest_target, px, py)
    second_dx, second_dy = platform_features(second_closest_target, px, py)

    norm_x  = px / WIDTH
    norm_vx = clamp(vx / MOVE_SPEED, -1.0, 1.0)
    norm_vy = clamp(vy / 12.0, -1.0, 1.0)

    has_above_flag = 1.0 if has_above else 0.0
    has_below_flag = 1.0 if has_below else 0.0
    is_falling     = 1.0 if vy > 0 else 0.0

    repeated_platform_flag = 1.0 if repeated_platform_jumps >= 3 else 0.0
    repeated_platform_norm = clamp(repeated_platform_jumps / 6.0, 0.0, 1.0)

    above_below_gap_dx = clamp(
        wrap_dx(above_target.centre_x, below_target.centre_x) / (WIDTH / 2),
        -1.0,
        1.0,
    )

    closest_second_gap_dx = clamp(
        wrap_dx(closest_target.centre_x, second_closest_target.centre_x) / (WIDTH / 2),
        -1.0,
        1.0,
    )

    return [
        above_dx,
        above_dy,
        below_dx,
        below_dy,
        closest_dx,
        closest_dy,
        second_dx,
        second_dy,
        norm_x,
        norm_vx,
        norm_vy,
        has_above_flag,
        has_below_flag,
        is_falling,
        repeated_platform_flag,
        above_below_gap_dx,
        closest_second_gap_dx,
        repeated_platform_norm,
    ]


def choose_action(agent, inputs, epsilon=0.0):
    if epsilon > 0 and random.random() < epsilon:
        return random.randint(0, 2)

    x = torch.tensor(inputs, dtype=torch.float32)

    with torch.no_grad():
        out = agent.net(x)

    return torch.argmax(out).item()

def apply_horizontal_action(vx, action):

    if action == 0:
        vx -= ACCELERATION
    elif action == 2:
        vx += ACCELERATION

    vx *= FRICTION
    vx = clamp(vx, -MOVE_SPEED, MOVE_SPEED)

    return vx


def rule_based_action(px, py, vx, vy, platforms):
    aim_target = find_target_platform_below(px, py, platforms)

    if aim_target is None:
        return 1

    dx = wrap_dx(aim_target.centre_x, px)

    if abs(dx) < 5:
        return 1
    elif dx < 0:
        return 0 
    else:
        return 2 

# Sim
def evaluate_agent(agent):
    px, py   = float(WIDTH // 2), float(HEIGHT - 100)
    vx, vy   = 0.0, float(JUMP_FORCE)
    camera_y = 0.0

    best_height = 0
    platforms   = initial_platforms()

    fitness = 0.0

    action_counts = [0, 0, 0]
    previous_action = None
    same_action_streak = 0
    longest_same_action_streak = 0

    stall_frames  = 0
    last_camera_y = 0.0

    last_landed_platform_id = None
    repeated_platform_jumps = 0

    for _frame in range(FRAMES_PER_AGENT):
        action = rule_based_action(px, py, vx, vy, platforms)
        aim_target = find_target_platform_below(px, py, platforms)

        action_counts[action] += 1

        if action == previous_action:
            same_action_streak += 1
        else:
            same_action_streak = 0

        longest_same_action_streak = max(longest_same_action_streak, same_action_streak)
        previous_action = action

        if same_action_streak > 45 and action != 1:
            fitness -= 0.25 * (same_action_streak - 45)

        vx = apply_horizontal_action(vx, action)

        vy += GRAVITY
        py += vy
        px  = (px + vx) % WIDTH
        
        if aim_target is not None:
            dx_to_center = abs(wrap_dx(aim_target.centre_x, px))
            center_reward = max(0.0, 1.0 - dx_to_center / 120.0)
            fitness += center_reward * 2.0

            if dx_to_center > PLATFORM_W:
                fitness -= 0.05

        screen_y = py - camera_y

        if screen_y < SCROLL_THRESH:
            delta = SCROLL_THRESH - screen_y
            camera_y -= delta

            stall_frames = 0
            fitness += 1.0 + delta * 0.05

            best_height = max(best_height, int(-camera_y // 5))

        if camera_y < last_camera_y:
            last_camera_y = camera_y
            stall_frames = 0
        else:
            stall_frames += 1

        if stall_frames > 300:
            fitness -= 4.0

        if stall_frames > 1200:
            break

        difficulty = best_height // 200

        # Platform generation + updates
        solid_platforms = [p for p in platforms if p.kind != KIND_BREAKING]
        top_y = min(p.y for p in solid_platforms) if solid_platforms else camera_y

        while top_y > camera_y - HEIGHT:
            new_ps = gen_platforms(top_y, difficulty)
            platforms.extend(new_ps)

            platforms.extend(gen_breaking(top_y, difficulty))
            top_y = new_ps[0].y

        platforms = [
            p for p in platforms
            if p.y < camera_y + HEIGHT + 100
            and not (p.kind == KIND_BREAKING and p.break_timer > 30)
        ]

        for p in platforms:
            p.update()

        # Collision
        if vy > 0:
            player_left   = px - 18
            player_right  = px + 18
            player_bottom = py

            for p in platforms:
                if p.broken:
                    continue

                if (
                    player_bottom >= p.y
                    and player_bottom <= p.y + 18
                    and player_right > p.x
                    and player_left < p.x + p.w
                ):
                    landing_dx = abs(wrap_dx(p.centre_x, px))
                    center_bonus = max(0.0, 1.0 - landing_dx / (PLATFORM_W / 2))

                    if p.id == last_landed_platform_id:
                        repeated_platform_jumps += 1
                    else:
                        last_landed_platform_id = p.id
                        repeated_platform_jumps = 1

                    fitness += 25.0 + 50.0 * center_bonus

                    if repeated_platform_jumps > 3:
                        fitness -= 80.0 * (repeated_platform_jumps - 3)

                    if p.kind == KIND_BREAKING:
                        p.broken = True
                    elif p.kind == KIND_SPRING and not p.spring_bounced:
                        vy = JUMP_FORCE * 1.8
                        p.spring_bounced = True
                    else:
                        vy = float(JUMP_FORCE)

                    break

        # Death
        if py - camera_y > HEIGHT + 50:
            break

    total_actions = sum(action_counts)

    if total_actions > 0:
        left_frac  = action_counts[0] / total_actions
        stay_frac  = action_counts[1] / total_actions
        right_frac = action_counts[2] / total_actions

        max_frac = max(left_frac, stay_frac, right_frac)

        if max_frac > 0.75:
            fitness -= 800.0 * (max_frac - 0.75)

        if right_frac > 0.70 or left_frac > 0.70:
            fitness -= 500.0 * max(left_frac, right_frac)

        balance = 1.0 - abs(left_frac - right_frac)
        fitness += 100.0 * balance

        if longest_same_action_streak > 120:
            fitness -= 2.0 * (longest_same_action_streak - 120)

    if repeated_platform_jumps > 3:
        fitness -= 150.0 * (repeated_platform_jumps - 3)

    agent.fitness = fitness
    return agent.fitness


# Evolution

def evolve(population):
    for agent in population:
        evaluate_agent(agent)

    population.sort(key=lambda a: a.fitness, reverse=True)

    best_f = population[0].fitness
    avg_f  = sum(a.fitness for a in population) / len(population)

    print(f"  Best: {best_f:>9.1f}    Avg: {avg_f:>9.1f}")

    new_pop = [population[i].copy() for i in range(ELITE_COUNT)]

    pool_size   = max(ELITE_COUNT, int(POPULATION_SIZE * 0.4))
    parent_pool = population[:pool_size]

    while len(new_pop) < POPULATION_SIZE:
        if random.random() < 0.7:
            p1, p2 = random.sample(parent_pool, 2)
            child  = crossover(p1, p2)
        else:
            child = random.choice(parent_pool).copy()

        mutate(child)
        new_pop.append(child)

    return new_pop, population[0]

def train():
    pygame.init()

    population = [Agent() for _ in range(POPULATION_SIZE)]

    best_ever = None
    best_fitness_ever = float("-inf")

    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen + 1}/{GENERATIONS}")

        population, best = evolve(population)

        if best.fitness > best_fitness_ever:
            best_fitness_ever = best.fitness
            best_ever = best.copy()

    print(f"\nTraining complete — best fitness: {best_fitness_ever:.1f}")

    return best_ever


# Visualization

def draw_player(surf, cx, cy, facing):
    cx, cy = int(cx), int(cy)

    draw_rounded_rect(surf, (0, 0, 0), pygame.Rect(cx - 14, cy - 27, 30, 28), 6)
    pygame.draw.circle(surf, (0, 0, 0), (cx + 1, cy - 17), 15)

    tube_length, tube_height = 19, 8

    if facing >= 0:
        pygame.draw.rect(
            surf,
            (0, 0, 0),
            pygame.Rect(cx + 9, cy - 19, tube_length, tube_height + 4),
            border_radius=4,
        )
    else:
        pygame.draw.rect(
            surf,
            (0, 0, 0),
            pygame.Rect(cx - 9 - tube_length, cy - 19, tube_length, tube_height + 4),
            border_radius=4,
        )

    for xo in (-10, -4, 4, 10):
        pygame.draw.rect(surf, (0, 0, 0), (cx + xo, cy - 7, 2, 12), border_radius=2)

    draw_rounded_rect(surf, PLAYER_COLOR, pygame.Rect(cx - 12, cy - 25, 26, 24), 6)
    pygame.draw.circle(surf, PLAYER_COLOR, (cx + 1, cy - 17), 13)

    if facing >= 0:
        pygame.draw.rect(
            surf,
            PLAYER_COLOR,
            pygame.Rect(cx + 7, cy - 17, tube_length, tube_height),
            border_radius=4,
        )
    else:
        pygame.draw.rect(
            surf,
            PLAYER_COLOR,
            pygame.Rect(cx - 7 - tube_length, cy - 17, tube_length, tube_height),
            border_radius=4,
        )

    pygame.draw.circle(surf, (10, 10, 80), (cx - 4, cy - 15), 2)
    pygame.draw.circle(surf, (10, 10, 80), (cx + 7, cy - 15), 2)
    
def watch_agent(agent):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neuroevolution Doodle Jump - Improved")
    clock = pygame.time.Clock()

    font_med   = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 18)
    font_tiny  = pygame.font.SysFont("Arial", 14)

    def reset():
        return (
            float(WIDTH // 2),
            float(HEIGHT - 100),
            0.0,
            float(JUMP_FORCE),
            0.0,
            0,
            True,
            initial_platforms(),
            1,
        )

    px, py, vx, vy, camera_y, score, alive, platforms, facing = reset()
    best_score = 0
    
    last_landed_platform_id = None
    repeated_platform_jumps = 0
    aim_target = None

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_r:
                    best_score = max(best_score, score)
                    px, py, vx, vy, camera_y, score, alive, platforms, facing = reset()
                    last_landed_platform_id = None
                    repeated_platform_jumps = 0
                    aim_target = None

        if alive:
            action = rule_based_action(px, py, vx, vy, platforms)
            aim_target = find_target_platform_below(px, py, platforms)

            vx = apply_horizontal_action(vx, action)

            if vx < -0.2:
                facing = -1
            elif vx > 0.2:
                facing = 1

            vy += GRAVITY
            py += vy
            px = (px + vx) % WIDTH

            screen_y = py - camera_y

            if screen_y < SCROLL_THRESH:
                camera_y -= SCROLL_THRESH - screen_y
                score = max(score, int(-camera_y // 5))

            difficulty = score // 200

            solid = [p for p in platforms if p.kind != KIND_BREAKING]
            top_y = min(p.y for p in solid) if solid else camera_y

            while top_y > camera_y - HEIGHT:
                new_ps = gen_platforms(top_y, difficulty)
                platforms.extend(new_ps)
                platforms.extend(gen_breaking(top_y, difficulty))
                top_y = new_ps[0].y

            platforms = [
                p for p in platforms
                if p.y < camera_y + HEIGHT + 100
                and not (p.kind == KIND_BREAKING and p.break_timer > 30)
            ]

            for p in platforms:
                p.update()

            if vy > 0:
                pr_rect = pygame.Rect(int(px) - 18, int(py) - 36, 36, 36)

                for p in platforms:
                    if p.broken:
                        continue

                    pr = p.rect

                    if (
                        pr_rect.bottom >= pr.top
                        and pr_rect.bottom <= pr.top + 18
                        and pr_rect.right > pr.left
                        and pr_rect.left < pr.right
                    ):
                        if p.id == last_landed_platform_id:
                            repeated_platform_jumps += 1
                        else:
                            last_landed_platform_id = p.id
                            repeated_platform_jumps = 1

                        if p.kind == KIND_BREAKING:
                            p.broken = True
                        elif p.kind == KIND_SPRING and not p.spring_bounced:
                            vy = JUMP_FORCE * 1.8
                            p.spring_bounced = True
                        else:
                            vy = float(JUMP_FORCE)

                        break

            if py - camera_y > HEIGHT + 50:
                alive = False
                best_score = max(best_score, score)

        draw_grid(screen, camera_y)

        for p in platforms:
            p.draw(screen, camera_y)

        draw_player(screen, px, py - camera_y, facing)

        screen.blit(font_med.render(f"{score}", True, SCORE_COLOR), (12, 10))
        screen.blit(font_tiny.render(f"Best: {best_score}", True, (120, 120, 140)), (12, 44))
        screen.blit(
            font_tiny.render(f"Same platform streak: {repeated_platform_jumps}", True, (120, 120, 140)),
            (12, 60),
        )

        hint = font_tiny.render("R = restart   ESC = quit", True, (160, 160, 180))
        screen.blit(hint, (WIDTH - hint.get_width() - 10, 10))

        for color, label, ly in [
            (PLATFORM_COLOR, "Normal", 35),
            (MOVING_COLOR, "Moving", 51),
            (BREAKING_COLOR, "Fragile", 67),
            (SPRING_COLOR, "Spring", 83),
        ]:
            pygame.draw.rect(screen, color, (WIDTH - 70, ly, 12, 8), border_radius=2)
            screen.blit(font_tiny.render(label, True, (120, 120, 140)), (WIDTH - 54, ly - 2))

        if alive and aim_target is not None:
            player_screen_y = py - camera_y
            platform_screen_y = aim_target.y - camera_y

            pygame.draw.line(
                screen,
                (200, 120, 120),
                (int(px), int(player_screen_y)),
                (int(aim_target.centre_x), int(platform_screen_y)),
                1,
            )

        if not alive:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))

            for surf, yoff in [
                (font_med.render("Agent fell!", True, DEAD_COLOR), -50),
                (font_med.render(f"Score: {score}", True, (255, 255, 255)), 0),
                (font_small.render("Press R to watch again", True, (200, 200, 200)), 40),
            ]:
                screen.blit(
                    surf,
                    (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 2 + yoff),
                )

        pygame.display.flip()


# entry

if __name__ == "__main__":
    best = train()

    if SHOW_BEST_AGENT:
        watch_agent(best)

    pygame.quit()