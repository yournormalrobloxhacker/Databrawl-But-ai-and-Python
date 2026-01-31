# Databrawl-style AI arena: Friendlies vs Corruptions — full feature set
# Python 3.11
# Power-ups, capture points, boss phases, minimap, hazards, leaderboard, etc.

import pygame
import random
import math
import numpy as np
from collections import deque
import os
import json

# ------- CONFIG -------
WIDTH, HEIGHT = 1280, 800
FPS = 60
ARENA_PADDING = 48
MINIMAP_W, MINIMAP_H = 120, 75
MINIMAP_X, MINIMAP_Y = WIDTH - MINIMAP_W - 12, 12

PLAYER_RADIUS = 12
HACKER_BODY_RADIUS = 22
ATTACK_RANGE = 52
BASE_DAMAGE = 18
COOLDOWN_TICKS = 26

NUM_AGENTS = 24
BEST_OF_ROUNDS = 3   # win X rounds to win match
CURRICULUM_ROUNDS = 5   # every N rounds, difficulty increases
MODEL_SAVE_PATH = "databrawl_models.npz"
LEADERBOARD_PATH = "databrawl_leaderboard.json"

# Power-ups
POWERUP_TYPES = ["speed", "damage", "shield", "heal"]
POWERUP_DURATION = 180   # frames
POWERUP_SPAWN_INTERVAL = 300
POWERUP_PICKUP_RADIUS = 20

# Capture points (data nodes)
CAPTURE_RADIUS = 50
CAPTURE_SPEED = 0.015
CAPTURE_BUFF_DMG = 1.15
CAPTURE_BUFF_HEAL = 0.05

# Boss phase (enrage below this HP%)
BOSS_ENRAGE_HP_PCT = 0.35
BOSS_ENRAGE_DMG = 1.3
BOSS_ENRAGE_HEAL = 1.4

# Commander ability cooldowns (frames)
ABILITY_COOLDOWN = 420
MOTHERBOARD_ABILITY_HEAL = 40
HACKER_ABILITY_DMG_AURA = 0.5

# Team & role definitions
TEAMS = {
    "Motherboard": {"team": "Friendly", "color": (140, 255, 140)},
    "Program": {"team": "Friendly", "color": (60, 200, 60)},
    "Firewall Security": {"team": "Friendly", "color": (200, 60, 60)},
    "Antivirus": {"team": "Friendly", "color": (220, 40, 40)},

    "Bloatware": {"team": "Unknown", "color": (140, 200, 120)},

    "Hacker": {"team": "Corruption", "color": (160, 80, 255)},
    "Malware": {"team": "Corruption", "color": (80, 130, 255)},
    "Virus": {"team": "Corruption", "color": (255, 130, 200)},
    "Spyware": {"team": "Corruption", "color": (240, 240, 90)},
    "Ransomware": {"team": "Corruption", "color": (255, 80, 80), "unlock_at": 5},
}

# Desired counts (balanced)
ROLE_COUNTS = {
    "Motherboard": 1,
    "Hacker": 1,
    "Firewall Security": 3,
    "Antivirus": 3,
    "Program": 6,
    "Malware": 4,
    "Virus": 3,
    "Spyware": 1,
    "Bloatware": 2,
}

# RL hyperparams (expanded state for power-ups, capture points)
STATE_SIZE = 18
ACTION_SIZE = 6
GAMMA = 0.99
LR = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9996
MEMORY_SIZE = 40000
BATCH_SIZE = 128
TRAIN_EVERY = 4

# Motherboard/Hacker abilities
MOTHERBOARD_RADIUS = 320
MOTHERBOARD_HEAL = 0.28
MOTHERBOARD_DMG_BUFF = 1.25
HACKER_BUFF_RADIUS = 380   # Hacker's aura range for buffing corruptions
HACKER_BUFF = 1.18

# Firewall shield
FIREWALL_SHIELD_RADIUS = 110
FIREWALL_REGEN = 0.6
FIREWALL_CORRUPTION_SLOW = 0.65
FIREWALL_CORRUPTION_DMG = 0.22

# Map: walls, destructible cover, hazards (damage, slow, safe zones), teleporter
MAP_WALLS = [
    pygame.Rect(320, 180, 640, 22),
    pygame.Rect(320, 608, 640, 22),
    pygame.Rect(240, 260, 22, 280),
    pygame.Rect(1018, 260, 22, 280),
    pygame.Rect(560, 360, 160, 22),
]
# Destructible cover: rect, hp (0 = destroyed)
DESTRUCTIBLE = [
    {"rect": pygame.Rect(440, 400, 60, 80), "hp": 80},
    {"rect": pygame.Rect(780, 400, 60, 80), "hp": 80},
]
# Hazards: damage, slow (mult), safe (heal), teleporter (target x,y)
HAZARD_DAMAGE = [pygame.Rect(520, 200, 240, 160)]
HAZARD_SLOW = [pygame.Rect(340, 220, 100, 100)]   # 0.5x speed
HAZARD_SAFE = [pygame.Rect(840, 500, 100, 120)]   # slight heal
TELEPORTER_FROM = pygame.Rect(380, 500, 40, 40)
TELEPORTER_TO = (900, 300)
HAZARD_DMG_PER_TICK = 0.35
HAZARD_SLOW_MULT = 0.5
HAZARD_SAFE_HEAL = 0.08

# ------- UTIL -------

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# ------- DQN (NumPy) -------
class DQN:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden=96):
        self.s = state_size
        self.a = action_size
        self.w1 = np.random.randn(self.s, hidden) * 0.08
        self.b1 = np.zeros(hidden)
        self.w2 = np.random.randn(hidden, hidden) * 0.08
        self.b2 = np.zeros(hidden)
        self.w3 = np.random.randn(hidden, self.a) * 0.08
        self.b3 = np.zeros(self.a)

    def forward(self, x):
        z1 = np.tanh(x @ self.w1 + self.b1)
        z2 = np.tanh(z1 @ self.w2 + self.b2)
        out = z2 @ self.w3 + self.b3
        return out, (x, z1, z2)

    def predict(self, x):
        return self.forward(x)[0]

    def train(self, x, y):
        preds, cache = self.forward(x)
        x_in, z1, z2 = cache
        m = max(1, x.shape[0])
        grad_out = (preds - y) / m
        dw3 = z2.T @ grad_out
        db3 = grad_out.sum(axis=0)
        dz2 = grad_out @ self.w3.T * (1 - z2 ** 2)
        dw2 = z1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = dz2 @ self.w2.T * (1 - z1 ** 2)
        dw1 = x_in.T @ dz1
        db1 = dz1.sum(axis=0)
        self.w3 -= LR * dw3
        self.b3 -= LR * db3
        self.w2 -= LR * dw2
        self.b2 -= LR * db2
        self.w1 -= LR * dw1
        self.b1 -= LR * db1

    def save_weights(self, path, prefix=""):
        data = {f"{prefix}w1": self.w1, f"{prefix}b1": self.b1, f"{prefix}w2": self.w2,
                f"{prefix}b2": self.b2, f"{prefix}w3": self.w3, f"{prefix}b3": self.b3}
        np.savez(path, **data)

    def load_weights(self, path, prefix=""):
        if os.path.exists(path):
            try:
                data = np.load(path)
                if f"{prefix}w1" in data.files:
                    self.w1, self.b1 = data[f"{prefix}w1"], data[f"{prefix}b1"]
                    self.w2, self.b2 = data[f"{prefix}w2"], data[f"{prefix}b2"]
                    self.w3, self.b3 = data[f"{prefix}w3"], data[f"{prefix}b3"]
            except Exception:
                pass

# ------- Replay Buffer -------
class ReplayBuffer:
    def __init__(self, maxlen=MEMORY_SIZE):
        self.buffer = deque(maxlen=maxlen)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, n=BATCH_SIZE):
        if len(self.buffer) < n:
            return None
        batch = random.sample(self.buffer, n)
        s, a, r, ns, d = zip(*batch)
        return np.array(s, dtype=np.float32), np.array(a, dtype=np.int32), np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32), np.array(d, dtype=np.bool_)

    def __len__(self):
        return len(self.buffer)

# ------- Agent -------
class Agent:
    def __init__(self, x, y, idx, role):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.idx = idx
        self.role = role
        self.alignment = TEAMS[role]["team"]
        self.color = TEAMS[role]["color"]
        self.radius = HACKER_BODY_RADIUS if role == "Hacker" else PLAYER_RADIUS
        self.max_hp = 220 if role in ("Motherboard", "Hacker") else 100
        self.hp = self.max_hp
        self.cooldown = 0
        self.command_target = None
        self.kills = 0
        self.last_state = None
        self.last_action = None
        self.attack_flash_until = 0
        self.buffs = {}   # "speed"|"damage"|"shield" -> until (step)
        self.trail = deque(maxlen=8)   # for trail effect
        self.death_effect_until = 0

    def alive(self):
        return self.hp > 0

    def has_buff(self, name):
        return name in self.buffs and self.buffs[name] > 0

    def apply_buff(self, name, until):
        self.buffs[name] = until

# ------- Helpers -------

def build_roles_list():
    roles = []
    for r, cnt in ROLE_COUNTS.items():
        roles += [r] * cnt
    # adjust if too many/too few
    if len(roles) < NUM_AGENTS:
        remaining = NUM_AGENTS - len(roles)
        choices = ["Program", "Malware", "Virus"]
        roles += random.choices(choices, k=remaining)
    elif len(roles) > NUM_AGENTS:
        roles = roles[:NUM_AGENTS]
    random.shuffle(roles)
    return roles


def nearest_enemy(agent, agents, prefer=None):
    if prefer and prefer.alive() and prefer.alignment != agent.alignment:
        return prefer, math.hypot(prefer.x - agent.x, prefer.y - agent.y)
    best = None
    best_d = 1e9
    for o in agents:
        if o is agent or not o.alive():
            continue
        if o.alignment == agent.alignment:
            continue
        d = math.hypot(o.x - agent.x, o.y - agent.y)
        if d < best_d:
            best, best_d = o, d
    return best, best_d


def get_active_walls():
    walls = list(MAP_WALLS)
    for d in DESTRUCTIBLE:
        if d["hp"] > 0:
            walls.append(d["rect"])
    return walls

def inside_walls(rect):
    for w in get_active_walls():
        if rect.colliderect(w):
            return True
    return False

def in_hazard_damage(agent):
    r = pygame.Rect(agent.x - agent.radius, agent.y - agent.radius, agent.radius * 2, agent.radius * 2)
    for hz in HAZARD_DAMAGE:
        if r.colliderect(hz):
            return True
    return False

def in_hazard_slow(agent):
    r = pygame.Rect(agent.x - agent.radius, agent.y - agent.radius, agent.radius * 2, agent.radius * 2)
    for hz in HAZARD_SLOW:
        if r.colliderect(hz):
            return True
    return False

def in_hazard_safe(agent):
    r = pygame.Rect(agent.x - agent.radius, agent.y - agent.radius, agent.radius * 2, agent.radius * 2)
    for hz in HAZARD_SAFE:
        if r.colliderect(hz):
            return True
    return False

def in_teleporter(agent):
    r = pygame.Rect(agent.x - agent.radius, agent.y - agent.radius, agent.radius * 2, agent.radius * 2)
    return r.colliderect(TELEPORTER_FROM)


# Spawn side by base role (Friendly roles vs Corruption roles); Bloatware = random side
FRIENDLY_ROLES = {"Motherboard", "Program", "Firewall Security", "Antivirus"}
CORRUPTION_ROLES = {"Hacker", "Malware", "Virus", "Spyware", "Ransomware"}


def spawn_side_x(role):
    if role in FRIENDLY_ROLES:
        return WIDTH * 0.25
    if role in CORRUPTION_ROLES:
        return WIDTH * 0.75
    return random.choice([WIDTH * 0.25, WIDTH * 0.75])


def build_state(agent, target, motherboard, hacker, powerups, capture_points):
    tx = (target.x - agent.x) / WIDTH if target else 0.0
    ty = (target.y - agent.y) / HEIGHT if target else 0.0
    # nearest power-up
    px, py = 0.0, 0.0
    if powerups:
        best_p = min(powerups, key=lambda p: math.hypot(p["x"] - agent.x, p["y"] - agent.y))
        px = (best_p["x"] - agent.x) / WIDTH
        py = (best_p["y"] - agent.y) / HEIGHT
    # capture point control (average)
    cap_f = sum(cp["friendly"] for cp in capture_points) / max(1, len(capture_points))
    cap_c = sum(cp["corruption"] for cp in capture_points) / max(1, len(capture_points))
    return np.array([
        agent.x / WIDTH,
        agent.y / HEIGHT,
        agent.vx / 3.5,
        agent.vy / 3.5,
        agent.hp / agent.max_hp,
        agent.cooldown / COOLDOWN_TICKS,
        tx, ty,
        1.0 if agent.alignment == "Corruption" else 0.0,
        1.0 if agent.role == "Motherboard" else 0.0,
        (motherboard.x - agent.x) / WIDTH if motherboard and motherboard.alive() else 0.0,
        (motherboard.y - agent.y) / HEIGHT if motherboard and motherboard.alive() else 0.0,
        (hacker.x - agent.x) / WIDTH if hacker and hacker.alive() else 0.0,
        (hacker.y - agent.y) / HEIGHT if hacker and hacker.alive() else 0.0,
        px, py, cap_f, cap_c,
    ], dtype=np.float32)

# ------- Setup -------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)

model_friendly = DQN()
model_corruption = DQN()
memory_friendly = ReplayBuffer()
memory_corruption = ReplayBuffer()
# load saved models if exist
if os.path.exists(MODEL_SAVE_PATH):
    model_friendly.load_weights(MODEL_SAVE_PATH, "friendly_")
    model_corruption.load_weights(MODEL_SAVE_PATH, "corruption_")

# Leaderboard
def load_leaderboard():
    if os.path.exists(LEADERBOARD_PATH):
        try:
            with open(LEADERBOARD_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"Friendly": 0, "Corruption": 0, "rounds_played": 0}

def save_leaderboard(lb):
    try:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(lb, f, indent=2)
    except Exception:
        pass

leaderboard = load_leaderboard()
match_wins = {"Friendly": 0, "Corruption": 0}
# Unlockable: Ransomware after 5 total match wins
total_wins = leaderboard.get("Friendly", 0) + leaderboard.get("Corruption", 0)
if total_wins >= 5:
    ROLE_COUNTS["Ransomware"] = 1

roles = build_roles_list()
agents = []
for i, role in enumerate(roles):
    x = random.randint(ARENA_PADDING, WIDTH - ARENA_PADDING)
    y = random.randint(ARENA_PADDING, HEIGHT - ARENA_PADDING)
    agents.append(Agent(x, y, i, role))

motherboard = next((a for a in agents if a.role == "Motherboard"), None)
hacker = next((a for a in agents if a.role == "Hacker"), None)

# place Motherboard and Hacker purposely to opposite sides
if motherboard:
    motherboard.x, motherboard.y = WIDTH * 0.25, HEIGHT * 0.5
if hacker:
    hacker.x, hacker.y = WIDTH * 0.75, HEIGHT * 0.5

# Capture points (data nodes)
capture_points = [
    {"x": WIDTH * 0.4, "y": HEIGHT * 0.5, "friendly": 0.5, "corruption": 0.5},
    {"x": WIDTH * 0.6, "y": HEIGHT * 0.5, "friendly": 0.5, "corruption": 0.5},
]
powerups = []
powerup_spawn_timer = 0
death_effects = []   # {x, y, until, color}

# Commander ability cooldowns
motherboard_ability_cd = 0
hacker_ability_cd = 0

# Neutral Bloatware spawn points
BLOATWARE_SPAWN_POINTS = [(WIDTH * 0.5, HEIGHT * 0.3), (WIDTH * 0.5, HEIGHT * 0.7)]
bloatware_spawn_timer = 0

# ------- Main Loop -------
epsilon = EPSILON
step = 0
running = True
kills = {a.idx: 0 for a in agents}
paused = False
team_kills = {"Friendly": 0, "Corruption": 0}   # kills scored by each team
game_over = False
winner = None
round_wins = {"Friendly": 0, "Corruption": 0}
match_over = False   # true when best-of reached

while running:
    clock.tick(FPS)
    step += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_p and not game_over:
            paused = not paused
        if event.type == pygame.KEYDOWN and game_over and event.key in (pygame.K_r, pygame.K_SPACE):
            # reset round or match
            game_over = False
            winner = None
            if match_over:
                round_wins["Friendly"] = 0
                round_wins["Corruption"] = 0
                match_over = False
            powerups.clear()
            powerup_spawn_timer = 0
            death_effects.clear()
            for cp in capture_points:
                cp["friendly"], cp["corruption"] = 0.5, 0.5
            for d in DESTRUCTIBLE:
                d["hp"] = 80
            motherboard_ability_cd = 0
            hacker_ability_cd = 0
            for a in agents:
                a.hp = a.max_hp
                a.cooldown = 0
                a.command_target = None
                a.last_state = None
                a.last_action = None
                a.buffs.clear()
                a.trail.clear()
                a.death_effect_until = 0
                if a.role == "Motherboard":
                    a.x, a.y = WIDTH * 0.25, HEIGHT * 0.5
                elif a.role == "Hacker":
                    a.x, a.y = WIDTH * 0.75, HEIGHT * 0.5
                else:
                    side_x = spawn_side_x(a.role)
                    a.x = random.uniform(side_x - 60, side_x + 60)
                    a.y = random.uniform(ARENA_PADDING, HEIGHT - ARENA_PADDING)

    if paused:
        # render only
        screen.fill((14, 14, 20))
        for hz in HAZARD_DAMAGE:
            pygame.draw.rect(screen, (80, 40, 40), hz)
        for w in MAP_WALLS:
            pygame.draw.rect(screen, (60, 60, 80), w)
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for fw in agents:
            if fw.role == "Firewall Security" and fw.alive():
                pygame.draw.circle(surf, (200, 60, 60, 48), (int(fw.x), int(fw.y)), FIREWALL_SHIELD_RADIUS)
        screen.blit(surf, (0, 0))
        if motherboard and motherboard.alive():
            for a in agents:
                if a.alignment == "Friendly" and a.command_target is not None and a.alive():
                    pygame.draw.line(screen, (200, 220, 120), (motherboard.x, motherboard.y), (a.x, a.y), 1)
        if hacker and hacker.alive():
            for a in agents:
                if a.alignment == "Corruption" and a.command_target is not None and a.alive():
                    pygame.draw.line(screen, (180, 100, 255), (hacker.x, hacker.y), (a.x, a.y), 1)
        for a in agents:
            color = a.color if a.alive() else (80, 80, 80)
            pygame.draw.circle(screen, color, (int(a.x), int(a.y)), a.radius)
            lab = font.render(a.role if len(a.role) < 12 else a.role[:11], True, (230, 230, 230))
            screen.blit(lab, (a.x - a.radius - 2, a.y - a.radius - 16))
            hp_w = int(((a.hp / a.max_hp) * (a.radius * 2)))
            hp_rect = pygame.Rect(int(a.x - a.radius), int(a.y - a.radius - 8), max(0, hp_w), 4)
            pygame.draw.rect(screen, (50, 200, 50), hp_rect)
        if motherboard and motherboard.alive():
            pygame.draw.circle(screen, (100, 255, 180), (int(motherboard.x), int(motherboard.y)), motherboard.radius, 2)
        if hacker and hacker.alive():
            pygame.draw.circle(screen, (180, 120, 255), (int(hacker.x), int(hacker.y)), hacker.radius, 2)
        y = 10
        for role, data in TEAMS.items():
            pygame.draw.rect(screen, data["color"], (10, y, 10, 10))
            txt = font.render(f"{role} ({data['team']})", True, (220, 220, 220))
            screen.blit(txt, (26, y - 2))
            y += 14
        pause_txt = font.render("PAUSED (P to resume)", True, (255, 200, 80))
        screen.blit(pause_txt, (WIDTH // 2 - 70, 12))
        friendly_alive = sum(1 for a in agents if a.alignment == "Friendly" and a.alive())
        corrupt_alive = sum(1 for a in agents if a.alignment == "Corruption" and a.alive())
        score_txt = font.render(f"Friendly: {team_kills['Friendly']} kills | {friendly_alive} alive   Corruption: {team_kills['Corruption']} kills | {corrupt_alive} alive", True, (200, 200, 200))
        screen.blit(score_txt, (8, HEIGHT - 40))
        info = font.render(f"Agents: {len(agents)}  Memory: {len(memory_friendly)+len(memory_corruption)}  Eps: {epsilon:.3f}", True, (200, 200, 200))
        screen.blit(info, (8, HEIGHT - 22))
        pygame.display.flip()
        continue

    if game_over:
        # no game logic; render only, show winner
        pass
    else:
        # Power-up spawn
        powerup_spawn_timer += 1
        if powerup_spawn_timer >= POWERUP_SPAWN_INTERVAL:
            powerup_spawn_timer = 0
            powerups.append({
                "x": random.uniform(ARENA_PADDING + 80, WIDTH - ARENA_PADDING - 80),
                "y": random.uniform(ARENA_PADDING + 80, HEIGHT - ARENA_PADDING - 80),
                "type": random.choice(POWERUP_TYPES),
                "until": step + 120,
            })
        powerups[:] = [p for p in powerups if p["until"] > step]

        # Capture point updates
        for cp in capture_points:
            friendly_count = sum(1 for a in agents if a.alignment == "Friendly" and a.alive()
                    and math.hypot(a.x - cp["x"], a.y - cp["y"]) <= CAPTURE_RADIUS)
            corrupt_count = sum(1 for a in agents if a.alignment == "Corruption" and a.alive()
                    and math.hypot(a.x - cp["x"], a.y - cp["y"]) <= CAPTURE_RADIUS)
            diff = (friendly_count - corrupt_count) * CAPTURE_SPEED
            cp["friendly"] = clamp(cp["friendly"] + diff, 0, 1)
            cp["corruption"] = 1.0 - cp["friendly"]

        # Boss enrage
        mb_enrage = motherboard and motherboard.alive() and (motherboard.hp / motherboard.max_hp) <= BOSS_ENRAGE_HP_PCT
        hk_enrage = hacker and hacker.alive() and (hacker.hp / hacker.max_hp) <= BOSS_ENRAGE_HP_PCT

        # Commander ability (use when ready)
        if motherboard and motherboard.alive() and motherboard_ability_cd <= 0 and mb_enrage:
            for a in agents:
                if a.alignment == "Friendly" and a.alive():
                    d = math.hypot(a.x - motherboard.x, a.y - motherboard.y)
                    if d <= MOTHERBOARD_RADIUS:
                        a.hp = min(a.max_hp, a.hp + MOTHERBOARD_ABILITY_HEAL * BOSS_ENRAGE_HEAL)
            motherboard_ability_cd = ABILITY_COOLDOWN
        if motherboard_ability_cd > 0:
            motherboard_ability_cd -= 1
        if hacker and hacker.alive() and hacker_ability_cd <= 0 and hk_enrage:
            for a in agents:
                if a.alignment == "Corruption" and a.alive():
                    d = math.hypot(a.x - hacker.x, a.y - hacker.y)
                    if d <= HACKER_BUFF_RADIUS:
                        a.apply_buff("ability_dmg", step + 90)
            hacker_ability_cd = ABILITY_COOLDOWN
        if hacker_ability_cd > 0:
            hacker_ability_cd -= 1

        # commanders pick targets and broadcast
        if motherboard and motherboard.alive():
            # friendlies defend motherboard: pick nearby corruption as prioritized target
            best, bd = None, 1e9
            for a in agents:
                if a.alignment == "Corruption" and a.alive():
                    d = math.hypot(a.x - motherboard.x, a.y - motherboard.y)
                    if d < bd:
                        best, bd = a, d
            # broadcast to friendlies within radius
            for a in agents:
                if a.alignment == "Friendly" and a.alive():
                    d = math.hypot(a.x - motherboard.x, a.y - motherboard.y)
                    if d <= MOTHERBOARD_RADIUS:
                        a.command_target = best
                    else:
                        # keep previous command if exists
                        pass
        if hacker and hacker.alive():
            best, bd = None, 1e9
            for a in agents:
                if a.alignment == "Friendly" and a.alive():
                    d = math.hypot(a.x - hacker.x, a.y - hacker.y)
                    if d < bd:
                        best, bd = a, d
            for a in agents:
                if a.alignment == "Corruption" and a.alive():
                    a.command_target = best

        # step each agent
        for agent in agents:
            if not agent.alive():
                continue
            # compute firewall effects
            firewall_slow = 1.0
            in_firewall = False
            for fw in agents:
                if fw.role == "Firewall Security" and fw.alive():
                    d_fw = math.hypot(fw.x - agent.x, fw.y - agent.y)
                    if d_fw <= FIREWALL_SHIELD_RADIUS:
                        in_firewall = True
                        if agent.alignment == "Friendly":
                            agent.hp = min(agent.max_hp, agent.hp + FIREWALL_REGEN * 0.2)
                        else:
                            firewall_slow = FIREWALL_CORRUPTION_SLOW
                            agent.hp -= FIREWALL_CORRUPTION_DMG
            # motherboard passive (enrage = stronger heal)
            if motherboard and motherboard.alive() and agent.alignment == "Friendly":
                d_mb = math.hypot(agent.x - motherboard.x, agent.y - motherboard.y)
                if d_mb <= MOTHERBOARD_RADIUS:
                    heal_mult = BOSS_ENRAGE_HEAL if mb_enrage else 1.0
                    agent.hp = min(agent.max_hp, agent.hp + MOTHERBOARD_HEAL * 0.2 * heal_mult)
            # hacker passive buff (enrage = stronger)
            if hacker and hacker.alive() and agent.alignment == "Corruption":
                d_h = math.hypot(agent.x - hacker.x, agent.y - hacker.y)
                if d_h <= HACKER_BUFF_RADIUS:
                    pass  # dmg buff applied in combat
            # hazard damage
            if in_hazard_damage(agent):
                agent.hp -= HAZARD_DMG_PER_TICK
            # hazard slow
            if in_hazard_slow(agent):
                firewall_slow *= HAZARD_SLOW_MULT
            # hazard safe (heal)
            if in_hazard_safe(agent):
                agent.hp = min(agent.max_hp, agent.hp + HAZARD_SAFE_HEAL)
            # capture point heal for friendlies
            for cp in capture_points:
                if cp["friendly"] > 0.6 and agent.alignment == "Friendly" and math.hypot(agent.x - cp["x"], agent.y - cp["y"]) <= CAPTURE_RADIUS:
                    agent.hp = min(agent.max_hp, agent.hp + CAPTURE_BUFF_HEAL)
                    break
            # teleporter
            if in_teleporter(agent):
                agent.x, agent.y = TELEPORTER_TO[0], TELEPORTER_TO[1]
            # power-up pickup
            for p in powerups[:]:
                if math.hypot(agent.x - p["x"], agent.y - p["y"]) <= POWERUP_PICKUP_RADIUS:
                    if p["type"] == "heal":
                        agent.hp = min(agent.max_hp, agent.hp + 50)
                    else:
                        agent.apply_buff(p["type"], step + POWERUP_DURATION)
                    powerups.remove(p)
                    break
            # buff expiry
            for k in list(agent.buffs.keys()):
                if agent.buffs[k] <= step:
                    del agent.buffs[k]
            # select target (prefer command)
            pref = agent.command_target if agent.command_target and agent.command_target.alive() else None
            target, dist = nearest_enemy(agent, agents, prefer=pref)
            state = build_state(agent, target, motherboard, hacker, powerups, capture_points)
            model = model_friendly if agent.alignment == "Friendly" else model_corruption
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                action = int(np.argmax(model.predict(state.reshape(1, -1))))
            # movement
            base_speed = 2.6 if agent.role not in ("Motherboard", "Hacker") else 1.8
            if agent.has_buff("speed"):
                base_speed *= 1.4
            speed = base_speed * firewall_slow
            agent.vx = agent.vy = 0.0
            if action == 0: agent.vy = -speed
            elif action == 1: agent.vy = speed
            elif action == 2: agent.vx = -speed
            elif action == 3: agent.vx = speed
            # attack
            if action == 4 and agent.cooldown == 0 and target and dist <= ATTACK_RANGE:
                curriculum = 1.0 + 0.04 * min((leaderboard.get("Friendly",0) + leaderboard.get("Corruption",0)) // (BEST_OF_ROUNDS * 2), 5)
                dmg = BASE_DAMAGE * curriculum
                if agent.role == "Antivirus" and target.alignment == "Corruption":
                    dmg *= 1.5
                if agent.has_buff("damage"):
                    dmg *= 1.35
                if agent.has_buff("ability_dmg"):
                    dmg *= 1.25
                # motherboard buff to friendlies near it
                if motherboard and motherboard.alive() and agent.alignment == "Friendly":
                    if math.hypot(agent.x - motherboard.x, agent.y - motherboard.y) <= MOTHERBOARD_RADIUS and agent.command_target is target:
                        dmg *= MOTHERBOARD_DMG_BUFF * (BOSS_ENRAGE_DMG if mb_enrage else 1.0)
                # hacker buff to corruptions
                if hacker and hacker.alive() and agent.alignment == "Corruption":
                    if math.hypot(agent.x - hacker.x, agent.y - hacker.y) <= HACKER_BUFF_RADIUS:
                        dmg *= HACKER_BUFF * (BOSS_ENRAGE_DMG if hk_enrage else 1.0)
                # capture point buff
                for cp in capture_points:
                    if math.hypot(agent.x - cp["x"], agent.y - cp["y"]) <= CAPTURE_RADIUS:
                        if agent.alignment == "Friendly" and cp["friendly"] > 0.6:
                            dmg *= CAPTURE_BUFF_DMG
                        elif agent.alignment == "Corruption" and cp["corruption"] > 0.6:
                            dmg *= CAPTURE_BUFF_DMG
                        break
                # apply damage
                # if target is inside firewall and is friendly, reduce incoming (friendly protected)
                protected = False
                for fw in agents:
                    if fw.role == "Firewall Security" and fw.alive():
                        if target.alignment == "Friendly" and math.hypot(fw.x - target.x, fw.y - target.y) <= FIREWALL_SHIELD_RADIUS:
                            protected = True
                            break
                if protected:
                    dmg *= 0.6
                if target.has_buff("shield"):
                    dmg *= 0.6
                target.hp -= dmg
                # destructible cover near impact
                for d in DESTRUCTIBLE:
                    if d["hp"] > 0 and math.hypot(target.x - d["rect"].centerx, target.y - d["rect"].centery) < 90:
                        d["hp"] = max(0, d["hp"] - 8)
                if target.hp <= 0:
                    death_effects.append({"x": target.x, "y": target.y, "until": step + 15, "color": target.color})
                    agent.kills += 1
                    kills[agent.idx] = kills.get(agent.idx, 0) + 1
                    if agent.alignment == "Friendly":
                        team_kills["Friendly"] = team_kills.get("Friendly", 0) + 1
                    else:
                        team_kills["Corruption"] = team_kills.get("Corruption", 0) + 1
                    # win condition: Hacker or Motherboard killed → killer's team wins
                    if target.role in ("Motherboard", "Hacker"):
                        game_over = True
                        winner = agent.alignment
                        round_wins[winner] = round_wins.get(winner, 0) + 1
                        leaderboard[winner] = leaderboard.get(winner, 0) + 1
                        leaderboard["rounds_played"] = leaderboard.get("rounds_played", 0) + 1
                        save_leaderboard(leaderboard)
                        if round_wins[winner] >= BEST_OF_ROUNDS:
                            match_over = True
                        # save both models
                        fw = model_friendly
                        cw = model_corruption
                        np.savez(MODEL_SAVE_PATH, friendly_w1=fw.w1, friendly_b1=fw.b1, friendly_w2=fw.w2, friendly_b2=fw.b2, friendly_w3=fw.w3, friendly_b3=fw.b3,
                                corruption_w1=cw.w1, corruption_b1=cw.b1, corruption_w2=cw.w2, corruption_b2=cw.b2, corruption_w3=cw.w3, corruption_b3=cw.b3)
                    # corruption conversion
                    if agent.alignment == "Corruption" and target.alignment == "Friendly":
                        # corpse corruption
                        new_role = random.choice(["Malware", "Virus"])
                        target.role = new_role
                        target.alignment = "Corruption"
                        target.color = TEAMS[new_role]["color"]
                        target.radius = PLAYER_RADIUS
                        target.max_hp = 100
                        target.hp = target.max_hp
                        target.last_state = None
                        target.last_action = None
                    else:
                        # terminal transition for killed agent (not converted); only store if we have valid (s,a)
                        if (hasattr(target, "last_state") and target.last_state is not None
                                and target.last_action is not None):
                            term_state = build_state(target, None, motherboard, hacker, powerups, capture_points)
                            mem = memory_friendly if target.alignment == "Friendly" else memory_corruption
                            mem.add(target.last_state, target.last_action, -1.0, term_state, True)
                        # respawn later
                        pass
                agent.cooldown = COOLDOWN_TICKS
                agent.attack_flash_until = step + 4
            # movement & collision with walls
            next_x = clamp(agent.x + agent.vx, ARENA_PADDING, WIDTH - ARENA_PADDING)
            next_y = clamp(agent.y + agent.vy, ARENA_PADDING, HEIGHT - ARENA_PADDING)
            r = pygame.Rect(next_x - agent.radius, next_y - agent.radius, agent.radius * 2, agent.radius * 2)
            blocked = inside_walls(r)
            if not blocked:
                agent.x, agent.y = next_x, next_y
            if agent.cooldown > 0:
                agent.cooldown -= 1
            reward = -0.001 * (dist if dist < 1e9 else math.hypot(WIDTH, HEIGHT))
            # reward shaping: protect motherboard, target Hacker
            if motherboard and motherboard.alive() and agent.alignment == "Friendly":
                if math.hypot(agent.x - motherboard.x, agent.y - motherboard.y) < 140:
                    reward += 0.002
                if target and target.role == "Hacker":
                    reward += 0.001
            if hacker and hacker.alive() and agent.alignment == "Corruption":
                if target and target.role == "Motherboard":
                    reward += 0.001
            # capture point reward
            for cp in capture_points:
                if math.hypot(agent.x - cp["x"], agent.y - cp["y"]) <= CAPTURE_RADIUS:
                    if agent.alignment == "Friendly" and cp["friendly"] > cp["corruption"]:
                        reward += 0.0005
                    elif agent.alignment == "Corruption" and cp["corruption"] > cp["friendly"]:
                        reward += 0.0005
                    break
            agent.trail.append((agent.x, agent.y))
            agent.last_state = state.copy()
            agent.last_action = action
            mem = memory_friendly if agent.alignment == "Friendly" else memory_corruption
            mem.add(state, action, reward, build_state(agent, target, motherboard, hacker, powerups, capture_points), not agent.alive())

        # Training (separate models for each team)
        if step % TRAIN_EVERY == 0:
            for model, memory in [(model_friendly, memory_friendly), (model_corruption, memory_corruption)]:
                batch = memory.sample()
                if batch:
                    s, a, r, ns, d = batch
                    q = model.predict(s)
                    q_next = model.predict(ns).max(axis=1)
                    for i in range(len(s)):
                        q[i, a[i]] = r[i] + GAMMA * q_next[i] * (0.0 if d[i] else 1.0)
                    model.train(s, q)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # respawn logic (spawn side by base role, not current alignment)
        for agent in agents:
            if not agent.alive() and random.random() < 0.01:
                if agent.role == "Motherboard" and motherboard:
                    agent.x, agent.y = motherboard.x, motherboard.y
                elif agent.role == "Hacker" and hacker:
                    agent.x, agent.y = hacker.x, hacker.y
                else:
                    side_x = spawn_side_x(agent.role)
                    agent.x = random.uniform(side_x - 60, side_x + 60)
                    agent.y = random.uniform(ARENA_PADDING, HEIGHT - ARENA_PADDING)
                agent.hp = agent.max_hp
                agent.cooldown = 0
                agent.command_target = None
                agent.last_state = None
                agent.last_action = None

    # Render
    screen.fill((14, 14, 20))
    # hazard zones
    for hz in HAZARD_DAMAGE:
        pygame.draw.rect(screen, (80, 40, 40), hz)
    for hz in HAZARD_SLOW:
        pygame.draw.rect(screen, (60, 60, 120), hz)
    for hz in HAZARD_SAFE:
        pygame.draw.rect(screen, (40, 80, 60), hz)
    pygame.draw.rect(screen, (120, 80, 180), TELEPORTER_FROM)
    pygame.draw.circle(screen, (100, 60, 140), (int(TELEPORTER_TO[0]), int(TELEPORTER_TO[1])), 25)
    # walls & destructible
    for w in get_active_walls():
        if w in MAP_WALLS:
            pygame.draw.rect(screen, (60, 60, 80), w)
    for d in DESTRUCTIBLE:
        if d["hp"] > 0:
            pygame.draw.rect(screen, (80, 70, 90), d["rect"])
    # capture points
    for cp in capture_points:
        col = (100, 255, 100) if cp["friendly"] > cp["corruption"] else (200, 100, 255)
        srf = pygame.Surface((CAPTURE_RADIUS * 2, CAPTURE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(srf, (*col, 60), (CAPTURE_RADIUS, CAPTURE_RADIUS), CAPTURE_RADIUS)
        screen.blit(srf, (int(cp["x"]) - CAPTURE_RADIUS, int(cp["y"]) - CAPTURE_RADIUS))
        pygame.draw.circle(screen, col, (int(cp["x"]), int(cp["y"])), CAPTURE_RADIUS, 2)
    # power-ups
    for p in powerups:
        col = {"speed": (100, 200, 255), "damage": (255, 100, 100), "shield": (100, 100, 255), "heal": (100, 255, 100)}.get(p["type"], (200, 200, 200))
        pygame.draw.circle(screen, col, (int(p["x"]), int(p["y"])), 10)
    # death effects (fading circles)
    death_effects[:] = [e for e in death_effects if e["until"] > step]
    for e in death_effects:
        alpha = int(180 * (e["until"] - step) / 15)
        r = 12 + (15 - (e["until"] - step))
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*e["color"], alpha), (r + 2, r + 2), r)
        screen.blit(s, (int(e["x"]) - r - 2, int(e["y"]) - r - 2))
    # firewall shields (semi-transparent)
    surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for fw in agents:
        if fw.role == "Firewall Security" and fw.alive():
            pygame.draw.circle(surf, (200, 60, 60, 48), (int(fw.x), int(fw.y)), FIREWALL_SHIELD_RADIUS)
    screen.blit(surf, (0, 0))
    # draw links from commanders
    if motherboard and motherboard.alive():
        for a in agents:
            if a.alignment == "Friendly" and a.command_target is not None and a.alive():
                pygame.draw.line(screen, (200, 220, 120), (motherboard.x, motherboard.y), (a.x, a.y), 1)
    if hacker and hacker.alive():
        for a in agents:
            if a.alignment == "Corruption" and a.command_target is not None and a.alive():
                pygame.draw.line(screen, (180, 100, 255), (hacker.x, hacker.y), (a.x, a.y), 1)
    # commander trails
    for a in agents:
        if a.role in ("Motherboard", "Hacker") and a.alive() and len(a.trail) > 1:
            pts = [(int(x), int(y)) for x, y in list(a.trail)]
            for i in range(len(pts) - 1):
                alpha = 80 + 100 * i // max(1, len(pts))
                col = (100, 255, 180) if a.role == "Motherboard" else (180, 120, 255)
                pygame.draw.line(screen, col, pts[i], pts[i + 1], 2)
    # agents
    for a in agents:
        color = a.color if a.alive() else (80, 80, 80)
        if a.attack_flash_until > step and a.alive():
            color = (min(255, color[0] + 100), min(255, color[1] + 80), min(255, color[2] + 80))
        pygame.draw.circle(screen, color, (int(a.x), int(a.y)), a.radius)
        if a.alive() and a.buffs:
            pygame.draw.circle(screen, (255, 220, 100), (int(a.x), int(a.y)), a.radius + 3, 2)
        lab = font.render(a.role if len(a.role) < 12 else a.role[:11], True, (230, 230, 230))
        screen.blit(lab, (a.x - a.radius - 2, a.y - a.radius - 16))
        hp_w = int(((a.hp / a.max_hp) * (a.radius * 2)))
        hp_rect = pygame.Rect(int(a.x - a.radius), int(a.y - a.radius - 8), max(0, hp_w), 4)
        pygame.draw.rect(screen, (50, 200, 50), hp_rect)
    # commander glyphs
    if motherboard and motherboard.alive():
        pygame.draw.circle(screen, (100, 255, 180), (int(motherboard.x), int(motherboard.y)), motherboard.radius, 2)
    if hacker and hacker.alive():
        pygame.draw.circle(screen, (180, 120, 255), (int(hacker.x), int(hacker.y)), hacker.radius, 2)
    # legend
    y = 10
    for role, data in TEAMS.items():
        pygame.draw.rect(screen, data["color"], (10, y, 10, 10))
        txt = font.render(f"{role} ({data['team']})", True, (220, 220, 220))
        screen.blit(txt, (26, y - 2))
        y += 14
    # score, round wins, leaderboard
    friendly_alive = sum(1 for a in agents if a.alignment == "Friendly" and a.alive())
    corrupt_alive = sum(1 for a in agents if a.alignment == "Corruption" and a.alive())
    score_txt = font.render(f"Friendly: {team_kills['Friendly']} | {friendly_alive} alive   Corruption: {team_kills['Corruption']} | {corrupt_alive} alive", True, (200, 200, 200))
    screen.blit(score_txt, (8, HEIGHT - 52))
    round_txt = font.render(f"Rounds: Friendly {round_wins['Friendly']}-{round_wins['Corruption']} Corruption  (best of {BEST_OF_ROUNDS})", True, (180, 200, 220))
    screen.blit(round_txt, (8, HEIGHT - 38))
    lb_txt = font.render(f"Leaderboard: F {leaderboard.get('Friendly', 0)} | C {leaderboard.get('Corruption', 0)}  [P] Pause", True, (200, 200, 200))
    screen.blit(lb_txt, (8, HEIGHT - 24))
    info = font.render(f"Memory: {len(memory_friendly)+len(memory_corruption)}  Eps: {epsilon:.3f}", True, (180, 180, 180))
    screen.blit(info, (8, HEIGHT - 10))
    # minimap
    mm = pygame.Surface((MINIMAP_W, MINIMAP_H))
    mm.fill((20, 20, 28))
    scale_x, scale_y = MINIMAP_W / WIDTH, MINIMAP_H / HEIGHT
    for a in agents:
        if a.alive():
            mx, my = int(a.x * scale_x), int(a.y * scale_y)
            col = (100, 255, 100) if a.alignment == "Friendly" else (200, 100, 255)
            sz = 2 if a.role in ("Motherboard", "Hacker") else 1
            pygame.draw.rect(mm, col, (mx - sz//2, my - sz//2, sz, sz))
    screen.blit(mm, (MINIMAP_X, MINIMAP_Y))
    pygame.draw.rect(screen, (80, 80, 100), (MINIMAP_X - 2, MINIMAP_Y - 2, MINIMAP_W + 4, MINIMAP_H + 4), 2)
    # win screen when Hacker or Motherboard was killed
    if game_over and winner:
        win_font = pygame.font.SysFont(None, 72)
        win_text = f"{winner.upper()} WINS!" if not match_over else f"{winner.upper()} WINS THE MATCH!"
        win_color = (100, 255, 140) if winner == "Friendly" else (180, 120, 255)
        win_surf = win_font.render(win_text, True, win_color)
        win_rect = win_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
        screen.blit(win_surf, win_rect)
        sub = font.render("(Hacker or Motherboard eliminated)" if not match_over else f"Score: {round_wins['Friendly']}-{round_wins['Corruption']}", True, (180, 180, 180))
        screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 24))
        reset_hint = font.render("Press R or SPACE to " + ("start new match" if match_over else "next round"), True, (200, 200, 200))
        screen.blit(reset_hint, (WIDTH // 2 - reset_hint.get_width() // 2, HEIGHT // 2 + 44))
    pygame.display.flip()

pygame.quit()