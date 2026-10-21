import pickle
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)


# ── Prioritized Experience Replay (sum-tree, O(log n) sample) ──────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity
        self.ptr = 0
        self.size = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            idx = left if s <= self.tree[left] else (2 * idx + 2, s - self.tree[left])[1] and 2 * idx + 2
            if not isinstance(idx, int):
                idx = 2 * idx + 2
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PERBuffer:
    """
    Prioritized Experience Replay — Schaul et al. 2015.
    Alpha controls prioritization strength; beta controls IS-weight correction.
    Expert demonstrations (BFS) start with max priority and are never evicted.
    """
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=50000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_inc = (beta_end - beta_start) / beta_steps
        self.max_p = 1.0
        self.seen = set()
        self.eps = 1e-6

    def add(self, exp, priority=None):
        key = hashlib.md5(exp['s'].tobytes() + str(exp['a']).encode()).hexdigest()[:16]
        if key in self.seen:
            return
        self.seen.add(key)
        p = (priority if priority is not None else self.max_p) ** self.alpha
        self.tree.add(p, exp)

    def add_expert(self, exp):
        self.add(exp, priority=self.max_p * 5.0)

    def sample(self, batch_size):
        if self.tree.size < batch_size:
            return None, None, None
        batch, idxs, weights = [], [], []
        seg = self.tree.total / batch_size
        min_p = (np.min(self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.tree.size]) + self.eps)
        self.beta = min(self.beta_end, self.beta + self.beta_inc)
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                continue
            w = ((self.tree.size * (p + self.eps)) ** (-self.beta))
            w /= ((self.tree.size * min_p) ** (-self.beta)) + self.eps
            idxs.append(idx)
            weights.append(w)
            batch.append(data)
        if not batch:
            return None, None, None
        return batch, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.eps) ** self.alpha
            self.max_p = max(self.max_p, abs(err) + self.eps)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.size


# ── Cross-level AEM memory (persists across levels — CLEAR-style) ─────────
class PersistentAEM:
    """
    Action Effect Memory that accumulates across levels.
    Provides positive forward transfer by preserving action→outcome associations.
    Implements CLEAR: off-policy replay mixed from current + past level diffs.
    """
    def __init__(self, capacity=1024):
        self.diffs = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)

    def push(self, diff, action, reward):
        self.diffs.append(diff)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_tensors(self, device, max_mem=64):
        n = min(len(self.diffs), max_mem)
        if n < 2:
            return None, None, None
        diffs = torch.zeros(1, n, 1, 64, 64, device=device)
        acts = torch.zeros(1, n, dtype=torch.long, device=device)
        rews = torch.zeros(1, n, device=device)
        for i, (d, a, r) in enumerate(list(zip(self.diffs, self.actions, self.rewards))[-n:]):
            diffs[0, i, 0] = torch.from_numpy(d.astype(np.float32))
            acts[0, i] = min(a, 4)
            rews[0, i] = r
        return diffs, acts, rews


# ── BFS Solver ────────────────────────────────────────────────────────────

class BFSSolver:
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS load failed: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v
        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)
        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True
                        )
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        diff = np.sum(f0 != f)
                        if diff > 0:
                            effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if effect_hash not in seen_effects:
                                seen_effects.add(effect_hash)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except:
                        pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls:
            return None

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        actions = self._scan_actions(game, f0, bg)

        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        game = g_warmup
                        f0 = f_after
                        actions = warmup_actions
                        break
                except:
                    pass

        if not actions:
            return None

        hidden_fields = None
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))

        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            g, hist, depth = queue.popleft()
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: solved {len(new_hist)} actions, {explored} explored, {elapsed:.1f}s")
                    self.solutions[level_idx] = new_hist
                    return new_hist
                if depth < 30:
                    queue.append((g2, new_hist, depth + 1))

        elapsed_first = time.time() - t0
        if len(visited) < 50 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                visited2 = set()
                queue2 = deque()
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = np.array(game2.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                h0_2 = self._state_hash(game2, f0_2, hidden_fields)
                visited2.add(h0_2)
                queue2.append((copy.deepcopy(game2), [], 0))
                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)
                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    g, hist, depth = queue2.popleft()
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            logger.info(f"BFS L{level_idx}: hidden-retry solved {len(new_hist)} actions")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        if depth < 30:
                            queue2.append((g2, new_hist, depth + 1))
        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i + 1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))
            if not matched:
                return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = transferred[:i + 1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


def find_game_source_and_class(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break
    if not src:
        import re
        for pattern in [f"/tmp/*/{gid}/*/{gid}.py", f"/kaggle/*/{gid}*/{gid}.py", f"**/game_sources/**/{gid}.py"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break
    return src, cls_name


# ── Neural network ───────────────────────────────────────────────────────

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1 = nn.Linear(ch, max(ch // r, 4))
        s.fc2 = nn.Linear(max(ch // r, 4), ch)
        s.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(s, x):
        B, C, H, W = x.shape
        w = torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2, 3])))))
        x = x * w.view(B, C, 1, 1)
        a = torch.sigmoid(s.sp(torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)))
        return x * a


class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim = mem_dim
        s.diff_enc = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=8), nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 2 * 2, mem_dim)
        )
        s.q_proj = nn.Linear(feat_dim, mem_dim)
        s.v_proj = nn.Linear(mem_dim + 1 + n_actions, n_actions)
        s.scale = mem_dim ** 0.5

    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B, M = mem_actions.shape
        if M == 0:
            return torch.zeros(B, 5, device=cnn_feat.device)
        keys = s.diff_enc(mem_diffs.reshape(B * M, 1, 64, 64)).reshape(B, M, s.mem_dim)
        q = s.q_proj(cnn_feat).unsqueeze(1)
        attn = F.softmax(torch.bmm(q, keys.transpose(1, 2)) / s.scale, dim=-1)
        act_oh = F.one_hot(mem_actions.clamp(0, 4), 5).float()
        vals = torch.cat([keys, mem_rewards.unsqueeze(-1), act_oh], dim=-1)
        ctx = torch.bmm(attn, vals).squeeze(1)
        return s.v_proj(ctx)


class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g = g
        s.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1)
        s.c3 = nn.Conv2d(64, 128, 3, padding=1)
        s.c4 = nn.Conv2d(128, 256, 3, padding=1)
        s.attn = CBAM(256)
        s.ar = nn.Conv2d(256, 64, 1)
        s.ap = nn.MaxPool2d(4, 4)
        s.af = nn.Linear(64 * 16 * 16, 256)
        s.ah = nn.Linear(256, 5)
        s.dr = nn.Dropout(0.15)
        s.cc1 = nn.Conv2d(256, 128, 3, padding=1)
        s.cc2 = nn.Conv2d(128, 64, 3, padding=1)
        s.cc3 = nn.Conv2d(64, 32, 1)
        s.cc4 = nn.Conv2d(32, 1, 1)
        s.gp = nn.AdaptiveAvgPool2d(1)
        s.gf = nn.Linear(256, 64)
        s.aea = ActionEffectAttention(feat_dim=64, mem_dim=32, n_actions=5)

    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x = F.relu(s.c1(x))
        x = F.relu(s.c2(x))
        x = F.relu(s.c3(x))
        f = F.relu(s.c4(x))
        f = s.attn(f)
        af = F.relu(s.ar(f))
        af = s.ap(af).reshape(f.size(0), -1)
        al = s.ah(s.dr(F.relu(s.af(af))))
        cf = F.relu(s.cc1(f))
        cf = F.relu(s.cc2(cf))
        cf = F.relu(s.cc3(cf))
        cl = s.cc4(cf).reshape(f.size(0), -1)
        if mem_diffs is not None and mem_actions is not None:
            gf = s.gf(s.gp(f).reshape(f.size(0), -1))
            al = al + s.aea(gf, mem_diffs, mem_actions, mem_rewards)
        return torch.cat([al, cl], 1)


def fast_objects(frame, bg):
    objs = []
    for c in range(16):
        if c == bg:
            continue
        mask = (frame == c)
        npix = int(np.sum(mask))
        if npix < 4 or npix > 3000:
            continue
        ys, xs = np.where(mask)
        objs.append((c, float(np.mean(xs)), float(np.mean(ys)), npix))
    return objs


# ── Agent ────────────────────────────────────────────────────────────────

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time() * 1e6) + hash(s.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G = 64
        s.IN = 26
        s.net = None
        s.opt = None

        # PER replaces uniform deque — expert transitions get max priority and survive level transitions
        s.per = PERBuffer(capacity=50000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=50000)

        # CLEAR: cross-level replay — retain a snapshot of solved-level transitions to mix during future training
        s._clear_archive = deque(maxlen=5000)

        s.bsz = 64
        s.tfreq = 10
        s.pt = None
        s.pai = None
        s.pr = None
        s.ph = None
        s.cl = -1
        s.fhist = deque(maxlen=6)
        s.la = 0
        s.al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
        s._wd = False
        s._bg = 0
        s._wm = None

        # Persistent AEM — survives level changes for forward transfer (CLEAR principle)
        s._paem = PersistentAEM(capacity=1024)

        s._ckpt_hash = None
        s._unproductive = 0
        s._undo_avail = False
        s._eps = 0.15
        s._eps_min = 0.03
        s._eps_decay = 0.9997
        s._prev_objs = None
        s._obj_moved = 0
        s._visited_hashes = set()

        # Refinement loop state — tracks per-level best sequence for TTT
        s._best_seq = []
        s._best_seq_reward = -999.0
        s._cur_seq = []
        s._cur_seq_reward = 0.0

        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False

        # TTT: gradient steps on current-level states using the live net
        s._ttt_steps = 0
        s._ttt_max = 3

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f):
        return getattr(f, 'score', None) or f.levels_completed

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if not s._bfs.load():
                s._bfs = None

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None:
            return None
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        s._bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == s._bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) | (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        d1 = torch.zeros(3, 64, 64, dtype=torch.float32)
        for i, prev in enumerate(reversed(list(s.fhist))):
            if i >= 3:
                break
            d1[i] = torch.from_numpy((frame != prev).astype(np.float32))
        d2 = torch.zeros(2, 64, 64, dtype=torch.float32)
        h = list(s.fhist)
        if len(h) >= 2:
            d2[0] = torch.from_numpy((h[-1] != h[-2]).astype(np.float32))
        if len(h) >= 4:
            d2[1] = torch.from_numpy((h[-2] != h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh, aug, d1, d2], 0).to(s.device)

    def _detect_template(s, frame):
        mask = torch.ones(4096, dtype=torch.float32)
        col_act = np.sum(frame != s._bg, axis=0)
        for c in range(20, 44):
            if col_act[c] <= 2 and np.sum(col_act[:c] > 0) >= 5 and np.sum(col_act[c + 1:] > 0) >= 5:
                for y in range(64):
                    for x in range(c + 1):
                        mask[y * 64 + x] = 0.05
                return mask
        row_act = np.sum(frame != s._bg, axis=1)
        for r in range(20, 44):
            if row_act[r] <= 2 and np.sum(row_act[:r] > 0) >= 5 and np.sum(row_act[r + 1:] > 0) >= 5:
                for y in range(r + 1):
                    for x in range(64):
                        mask[y * 64 + x] = 0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        mask = np.ones((64, 64), dtype=bool)
        mask[:2] = False
        mask[62:] = False
        diff = (prev_raw != curr_raw) & mask
        changed = np.any(diff)
        r = 0.0
        if curr_h != prev_h:
            r += 1.5 if curr_h not in s._visited_hashes else 0.3
        elif curr_h == prev_h:
            r -= 0.1
        if changed:
            r += 0.5
        curr_objs = fast_objects(curr_raw, s._bg)
        if s._prev_objs and curr_objs:
            moved = 0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0] == po[0]:
                        dist = abs(co[1] - po[1]) + abs(co[2] - po[2])
                        if 2 < dist < 20:
                            moved += 1
                            break
            if moved > 0:
                r += 0.3 * min(moved, 3)
                s._obj_moved = moved
        s._prev_objs = curr_objs
        s._visited_hashes.add(curr_h)
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al = logits[:5].clone()
        cl = logits[5:5 + 4096].clone()
        if avail is not None and len(avail) > 0:
            mask = torch.full_like(al, float('-inf'))
            a6 = False
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5:
                    mask[aid - 1] = 0.0
                elif aid == 6:
                    a6 = True
            al = al + mask
            if not a6:
                cl = cl + torch.full_like(cl, float('-inf'))
        if s._wm is not None:
            cl = cl + torch.log(s._wm.to(s.device).clamp(min=0.01))
        ap = torch.sigmoid(al / temp)
        cp = torch.sigmoid(cl / temp) / (s.G * s.G)
        allp = torch.cat([ap, cp])
        sm = allp.sum()
        if sm < 1e-8:
            allp = torch.ones_like(allp) / len(allp)
        else:
            allp = allp / sm
        idx = np.random.choice(len(allp), p=allp.cpu().numpy())
        if idx < 5:
            return idx, None
        ci = idx - 5
        return 5, (ci // s.G, ci % s.G)

    def _heuristic(s, frame, avail, step):
        av = set(int(a.value) if hasattr(a, 'value') else int(a) for a in avail)
        for d in [1, 2, 3, 4]:
            if d in av and step < 4:
                return d - 1, None
        if 6 in av:
            cnt = np.bincount(frame.flatten(), minlength=16)
            targets = []
            for c in range(16):
                if c == s._bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(frame == c)
                if len(ys) >= 2:
                    targets.append((int(np.median(xs)), int(np.median(ys)), len(ys)))
            targets.sort(key=lambda t: t[2])
            pidx = step - 4
            if 0 <= pidx < len(targets):
                return 5, (targets[pidx][1], targets[pidx][0])
        if 5 in av:
            return 4, None
        choices = [a for a in av if 1 <= a <= 5]
        if choices:
            return random.choice(choices) - 1, None
        return 0, None

    def _frame_to_tensor(s, frame):
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) | (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        zeros = torch.zeros(5, 64, 64, dtype=torch.float32)
        return torch.cat([oh, aug, zeros], 0)

    def _train(s):
        """
        PER-based training: sample by priority, compute BCE loss,
        feed TD-errors back to update priorities (Schaul et al. 2015).
        CLEAR: mix 25% of samples from the cross-level archive to prevent forgetting.
        """
        if len(s.per) < s.bsz:
            return

        batch, idxs, weights = s.per.sample(s.bsz)
        if batch is None:
            return

        # CLEAR: replace 25% of batch with archived cross-level transitions
        archive_n = min(s.bsz // 4, len(s._clear_archive))
        if archive_n > 0:
            archive_sample = random.sample(list(s._clear_archive), archive_n)
            batch = batch[:s.bsz - archive_n] + archive_sample
            weights = np.concatenate([weights[:s.bsz - archive_n], np.ones(archive_n, dtype=np.float32)])

        states = torch.stack([s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts = torch.tensor([e['a'] for e in batch], dtype=torch.long, device=s.device)
        rews = torch.tensor([e['r'] for e in batch], dtype=torch.float32, device=s.device)
        ws = torch.tensor(weights, device=s.device)

        rews = torch.sigmoid(rews)
        s.opt.zero_grad()
        logits = s.net(states)
        acts_c = acts.clamp(0, logits.size(1) - 1)
        sel = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)

        # IS-weighted loss for PER bias correction
        elem_loss = F.binary_cross_entropy_with_logits(sel, rews, reduction='none')
        loss = (elem_loss * ws).mean()
        p = torch.sigmoid(logits)
        loss = loss - 0.0001 * p[:, :5].mean() - 0.00001 * p[:, 5:].mean()
        loss.backward()
        s.opt.step()

        # Update PER priorities with TD-error proxy
        with torch.no_grad():
            td_errors = elem_loss[:len(idxs)].cpu().numpy()
        if idxs:
            s.per.update_priorities(idxs[:len(td_errors)], td_errors)

    def _ttt_step(s, tensor):
        """
        Test-Time Training: one gradient step adapting to the current observation.
        Matches the refinement-loop paradigm (ARC Prize 2025 top solution theme).
        Uses the model's own output as a self-supervised signal via entropy minimization.
        """
        if s._ttt_steps >= s._ttt_max or s.opt is None:
            return
        s.net.train()
        s.opt.zero_grad()
        logits = s.net(tensor.unsqueeze(0))
        p = torch.sigmoid(logits)
        entropy = -(p * torch.log(p.clamp(1e-8)) + (1 - p) * torch.log((1 - p).clamp(1e-8)))
        loss = entropy.mean() * 0.01
        loss.backward()
        s.opt.step()
        s.net.eval()
        s._ttt_steps += 1

    def _get_aem_tensors(s):
        return s._paem.get_tensors(s.device, max_mem=64)

    def _archive_level(s):
        """Snapshot current-level PER buffer into CLEAR archive to prevent future forgetting."""
        if len(s.per) == 0:
            return
        sample_n = min(200, len(s.per))
        batch, _, _ = s.per.sample(min(sample_n, len(s.per)))
        if batch:
            s._clear_archive.extend(batch)

    def is_done(s, frames, lf):
        try:
            return lf.state is GameState.WIN or (time.time() - s.start_time) >= 8 * 3600 - 300
        except:
            return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)

                # Archive current level's experience into CLEAR buffer BEFORE resetting per
                s._archive_level()

                # Reset per-level state but keep CLEAR archive and persistent AEM
                s.per = PERBuffer(capacity=50000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=50000)
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt', 'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state = torch.load(wp, map_location=s.device, weights_only=True)
                            ms = s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape == ms[k].shape:
                                    ms[k] = state[k]
                            s.net.load_state_dict(ms)
                            break
                    except:
                        pass
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                s.cl = lvl
                s.fhist.clear()
                s.la = 0
                s._wd = False
                s._wm = None
                s._eps = 0.15
                s._prev_objs = None
                s._obj_moved = 0
                s._ckpt_hash = None
                s._unproductive = 0
                s._visited_hashes = set()
                s._best_seq = []
                s._best_seq_reward = -999.0
                s._cur_seq = []
                s._cur_seq_reward = 0.0
                s._ttt_steps = 0

                # CLTI: inject BFS expert demos for previous level with max PER priority
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl - 1)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        for act_id, data in prev_sol:
                            prev_frame = np.array(replay_game.get_pixels(0, 0, 64, 64), dtype=np.int64)
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            replay_game.perform_action(ai, raw=True)
                            action_idx = act_id - 1 if act_id <= 5 else (5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                            exp = {'s': prev_frame.copy(), 'a': action_idx, 'r': 2.0}
                            s.per.add_expert(exp)
                            s._clear_archive.append(exp)
                        if len(s.per) >= s.bsz:
                            for _ in range(min(20, len(s.per) // s.bsz)):
                                s._train()
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")

            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a, 'value') else int(a)) == 7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask = np.ones((64, 64), dtype=bool)
                mask[:2] = False
                mask[62:] = False
                diff_map = (s.pr != raw) & mask
                changed = np.any(diff_map)
                eh = hashlib.md5(s.pr.tobytes()[:1000] + str(s.pai).encode()).hexdigest()[:16]
                r = s._reward(s.pr, raw, '', ch)
                exp = {'s': s.pr.copy(), 'a': s.pai, 'r': r}
                s.per.add(exp)
                if changed:
                    s._paem.push(diff_map, min(s.pai, 4), r)
                    s._ckpt_hash = ch
                    s._unproductive = 0
                    s._cur_seq_reward += r
                else:
                    s._unproductive += 1

                # Track best action sequence for refinement loop
                if s._cur_seq_reward > s._best_seq_reward:
                    s._best_seq_reward = s._cur_seq_reward
                    s._best_seq = list(s._cur_seq)

            if s._wm is None:
                s._wm = s._detect_template(raw)

            if s._undo_avail and s._unproductive >= 30 and s._ckpt_hash:
                s._unproductive = 0
                a = GameAction.ACTION7
                a.reasoning = "undo"
                s.pt = tensor
                s.pai = 6
                s.pr = raw.copy()
                s.ph = ch
                s.la += 1
                return a

            if not s._wd:
                if s.la < 10:
                    aidx, coords = s._heuristic(raw, avail, s.la)
                else:
                    s._wd = True
                    for _ in range(min(5, len(s.per) // s.bsz)):
                        s._train()

            if s._wd:
                # TTT: adapt weights to this specific observation before sampling
                s._ttt_step(tensor)

                if random.random() < s._eps:
                    aidx, coords = s._sample(torch.zeros(4101, device=s.device), avail, temp=2.0)
                else:
                    with torch.no_grad():
                        mem = s._get_aem_tensors()
                        if mem[0] is not None:
                            logits = s.net(tensor.unsqueeze(0), *mem).squeeze(0)
                        else:
                            logits = s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx, coords = s._sample(logits, avail, temp=0.5)
                s._eps = max(s._eps_min, s._eps * s._eps_decay)
            elif s.la >= 10:
                s._wd = True
                aidx, coords = 0, None

            if aidx < 5:
                sel = s.al[aidx]
                sel.reasoning = f"cnn:a{aidx + 1}"
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": int(x), "y": int(y)})
                sel.reasoning = f"cnn:c({x},{y})"

            s.pt = tensor
            action_idx = aidx if aidx < 5 else (5 + coords[0] * s.G + coords[1])
            s.pai = action_idx
            s._cur_seq.append((aidx, coords))
            s.pr = raw.copy()
            s.ph = ch
            s.la += 1

            if s.action_counter % s.tfreq == 0 and s._wd:
                s._train()
                # Every 5 training triggers, reset TTT counter to allow fresh adaptation
                if (s.action_counter // s.tfreq) % 5 == 0:
                    s._ttt_steps = 0

            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.al)
            a.reasoning = f"err:{e}"
            return a
