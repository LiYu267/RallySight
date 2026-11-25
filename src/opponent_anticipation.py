"""Predict the next opponent contact and outgoing puck state from past XY."""
from collections import deque
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HISTORY = 8
MAX_WAIT_S = 1.6
CONFIDENCE = .60
READY_TOLERANCE_M = .05


def observation(frame, side):
    other = 'right' if side == 'left' else 'left'
    sign = 1 if side == 'left' else -1
    return np.asarray(frame['puck_xy'] + frame[f'{other}_xy'] + frame[f'{side}_xy'], dtype=np.float32)*sign


def outgoing_contact(event, side):
    return event is not None and event['side'] == side and event['forward'] and event['phase'] == 'strike'


def inbound_contact(event, side):
    return event is not None and event['side'] != side and event['forward']


def history_tensor(history):
    points = np.asarray(history, dtype=np.float32)
    delta = np.diff(points, axis=0, prepend=points[:1])*20.
    return np.concatenate((points, delta), axis=1)


class OpponentGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(12, 48, batch_first=True)
        self.output = nn.Sequential(nn.Linear(49, 48), nn.SiLU(), nn.Linear(48, 6))

    def forward(self, history, age):
        _, hidden = self.gru(history)
        return self.output(torch.cat((hidden[-1], age), dim=-1))


def hindsight_examples(episode):
    """Future contact labels come exclusively from the recorded episode."""
    events = episode['contacts']
    examples = []
    for side in ('left', 'right'):
        sign = 1 if side == 'left' else -1
        history = deque(maxlen=HISTORY)
        current = 0
        for frame in episode['trajectory']:
            t = frame['t_s']
            history.append(observation(frame, side))
            while current < len(events) and events[current]['t_s'] <= t:
                current += 1
            last = events[current-1] if current else None
            if len(history) < HISTORY or not frame['idle'][side] or not outgoing_contact(last, side):
                continue
            age = t-last['t_s']
            if age > MAX_WAIT_S:
                continue
            upcoming = events[current] if current < len(events) else None
            if upcoming is None and episode['reason'] == 'TIME_LIMIT' and episode['elapsed_s']-t < MAX_WAIT_S:
                continue  # Unknown outcome across collection cutoff.
            positive = inbound_contact(upcoming, side) and upcoming['t_s']-t <= MAX_WAIT_S
            target = np.zeros(6, dtype=np.float32)
            if positive:
                target[:] = [1., upcoming['t_s']-t,
                             *(np.asarray(upcoming['puck_xy'])*sign),
                             *(np.asarray(upcoming['velocity'])*sign*.5)]
            examples.append((history_tensor(history), [age], target))
    return examples


def batch(examples):
    if not examples:
        raise ValueError('No eligible pre-contact observations')
    return tuple(torch.tensor(np.asarray(x), dtype=torch.float32) for x in zip(*examples))


def expert_examples(path):
    """Read measured expert replay, using the recorded EE XY as a paddle proxy."""
    import h5py
    with h5py.File(path, 'r') as f:
        t = f['task_state/timestamps'][:]
        if len(t) < HISTORY or not np.allclose(np.diff(t), .05, atol=.001):
            return []
        xy = f['task_state/puck_pose'][:, :2]
        velocity = f['task_state/puck_linear_velocity'][:, :2]
        left = f['observations/left/eef_pose'][:len(t), :2]
        right = f['observations/right/eef_pose'][:len(t), :2]
        phases = f['task_state/phase'][:]
        metadata = json.loads(f['episode/metadata_json'][()])
    events = []
    for entry in metadata['contacts']:
        side, time_s = entry['arm'], entry['episode_time']
        index = np.searchsorted(t, time_s+1e-6)
        if index >= len(t) or t[index]-time_s > .051:
            continue
        vxy = velocity[index]
        events.append(dict(side=side, t_s=time_s, phase='strike',
                           forward=bool((1 if side=='left' else -1)*vxy[0] > .05),
                           puck_xy=(xy[index]-(t[index]-time_s)*vxy).tolist(),
                           velocity=vxy.tolist()))
    if len(events) != 2 or not all(event['forward'] for event in events):
        return []
    trajectory = [dict(t_s=float(s), puck_xy=p.tolist(), left_xy=l.tolist(), right_xy=r.tolist(),
                       idle=dict(left=not phase.startswith(b'DIRECT'), right=phase != b'RIGHT_HIT'))
                  for s,p,l,r,phase in zip(t, xy, left, right, phases) if s < events[-1]['t_s']]
    return hindsight_examples(dict(trajectory=trajectory, contacts=events, reason='TIME_LIMIT',
                                    elapsed_s=float(t[-1])))


def fit(out, expert_files):
    files = sorted(out.glob('train_*.json'))
    if len(files) < 10:
        raise ValueError('Expected at least ten separate simulation episodes')
    torch.manual_seed(42)
    groups = [hindsight_examples(json.loads(f.read_text())) for f in files]
    demos = [expert_examples(path) for path in expert_files]
    expert_rows = [row for ep in demos for row in ep]
    if not expert_rows:
        raise ValueError('No verified expert examples')
    tr = batch([row for ep in groups[:-5] for row in ep]+expert_rows)
    val = batch([row for ep in groups[-5:] for row in ep])
    net = OpponentGRU()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002)

    def loss(prediction, target):
        positive = target[:, 0] > .5
        classification = F.binary_cross_entropy_with_logits(prediction[:, 0], target[:, 0])
        regression = F.smooth_l1_loss(prediction[positive, 1:], target[positive, 1:]) if positive.any() else classification*0
        return classification+4*regression

    best = float('inf')
    for epoch in range(55):
        for idx in torch.randperm(len(tr[0])).split(64):
            h, a, target = (x[idx] for x in tr)
            optimizer.zero_grad()
            loss(net(h,a), target).backward()
            optimizer.step()
        with torch.no_grad():
            pred = net(*val[:2])
            objective = float(loss(pred, val[2]))
            if objective < best:
                best = objective
                torch.save(net.state_dict(), out/'opponent.pt')
                positive = val[2][:, 0] > .5
                errors = (pred[positive, 1:]-val[2][positive, 1:]).abs().mean(dim=0)
                summary = dict(episodes=len(files), expert_episodes=sum(bool(ep) for ep in demos),
                               expert_rows=len(expert_rows), train_rows=len(tr[0]), val_rows=len(val[0]),
                               train_positive=int(tr[2][:,0].sum()), val_positive=int(positive.sum()),
                               epoch=epoch, validation_loss=best,
                               validation_accuracy=float(((pred[:,0]>=0)==positive).float().mean()),
                               positive_mae_contact_s_xy_m_vxy_m_s=(errors*torch.tensor([1,1,1,2,2])).tolist())
    (out/'fit.json').write_text(json.dumps(summary, indent=2))
    print('FIT', summary, flush=True)


def return_line(xy, velocity, line_x, rail_y):
    """Project an outgoing shot to a line, reflecting the side rails."""
    x, y = xy
    vx, vy = velocity
    if vx > -.2 or x <= line_x or rail_y <= 0:
        return None
    dt = (line_x-x)/vx
    crossing_y = rail_y-abs((y+vy*dt+rail_y) % (4*rail_y)-2*rail_y)
    return float(dt), float(crossing_y)


class Anticipate:
    def __init__(self, net, settings, rail_y, adapter=None):
        self.net, self.settings, self.rail_y = net, settings, rail_y
        self.history = deque(maxlen=HISTORY)
        self.adapter = adapter

    def eligible(self, frame, side, last, idle):
        if not idle or len(self.history) < HISTORY or not outgoing_contact(last, side):
            return False
        age = frame['t_s']-last['t_s']
        if not 0 <= age <= MAX_WAIT_S:
            return False
        # Returning puck or a rebound already headed into our half: use V4.
        if frame['puck_xy'][0] < 0 and self.history[-1][0] < self.history[-2][0]:
            return False
        return True

    @torch.inference_mode()
    def __call__(self, frame, side, last, idle):
        if side != 'left':
            return None
        self.history.append(observation(frame, side))
        if not self.eligible(frame, side, last, idle):
            return None
        age = frame['t_s']-last['t_s']
        prediction = self.net(torch.from_numpy(history_tensor(self.history))[None],
                              torch.tensor([[age]], dtype=torch.float32))[0]
        confidence = float(prediction[0].sigmoid())
        delay, x, y, vx, vy = map(float, prediction[1:])
        if confidence < CONFIDENCE or not 0 <= delay <= MAX_WAIT_S or x < 0:
            return None
        crossing = return_line((x,y), (2*vx,2*vy), self.settings['intercept_x_m'], self.rail_y)
        if crossing is None:
            return None
        flight, intercept_y = crossing
        limit = self.settings['ready_y_limit_m']
        ready_y = float(np.clip(intercept_y, -limit, limit))
        correction = 0.
        if self.adapter is not None:
            features = [*self.history[-1], confidence, delay, x, y, 2*vx, 2*vy,
                        ready_y, max(0.,1-frame['t_s']/self.settings['episode_seconds'])]
            correction = self.adapter(features, frame['t_s'])
        return dict(confidence=confidence, opponent_delay_s=delay, opponent_xy_m=[x,y],
                    outgoing_vxy_m_s=[2*vx,2*vy], intercept_delay_s=delay+flight,
                    nominal_ready_y_m=ready_y, ready_y_correction_m=correction,
                    ready_y_m=float(np.clip(ready_y+correction, -limit, limit)))


class StableAnticipate(Anticipate):
    """Commit only to a consistent pre-contact lane and slew its command."""

    def __init__(self, net, settings, rail_y, *, agreement_frames=2, ready_speed_m_s=.95):
        super().__init__(net, settings, rail_y)
        if agreement_frames < 2 or not math.isfinite(ready_speed_m_s) or ready_speed_m_s <= 0:
            raise ValueError('Stable anticipation needs at least two frames and a positive ready speed')
        self.votes = deque(maxlen=agreement_frames)
        self.ready_speed_m_s = ready_speed_m_s
        self.hit_time = None
        self.goal_y = self.command_y = self.last_time = None

    def __call__(self, frame, side, last, idle):
        forecast = super().__call__(frame, side, last, idle)
        if side != 'left':
            return None
        if not self.eligible(frame, side, last, idle):
            self.hit_time = None
            self.votes.clear()
            return None
        now = frame['t_s']
        if self.hit_time != last['t_s']:
            self.hit_time = last['t_s']
            self.command_y = float(np.clip(frame['left_xy'][1], -self.settings['ready_y_limit_m'],
                                           self.settings['ready_y_limit_m']))
            self.goal_y = self.command_y
            self.last_time = now
            self.votes.clear()
        if forecast is None or not math.isfinite(forecast['ready_y_m']):
            self.votes.clear()
        else:
            self.votes.append(forecast['ready_y_m'])
            if len(self.votes) == self.votes.maxlen and max(self.votes)-min(self.votes) <= .12:
                proposal = float(np.median(self.votes))
                if abs(proposal-self.goal_y) > .025:
                    self.goal_y = proposal
        # A missing or contradictory frame holds the last agreed lane.
        step = self.ready_speed_m_s*(now-self.last_time)
        self.command_y += float(np.clip(self.goal_y-self.command_y, -step, step))
        self.last_time = now
        return {**(forecast or {}), 'raw_ready_y_m': forecast['ready_y_m'] if forecast else None,
                'committed_ready_y_m': self.goal_y, 'ready_y_m': self.command_y}


def metrics(episode, cfg):
    contacts = episode['contacts']
    frames = episode['trajectory']
    ready_leads, ready_errors = [], []
    received = opportunities = 0
    for i, event in enumerate(contacts):
        if not inbound_contact(event, 'left'):
            continue
        subsequent = contacts[i+1] if i+1 < len(contacts) else None
        if subsequent is None and episode['reason'] == 'TIME_LIMIT' and episode['elapsed_s']-event['t_s'] < 1.5:
            continue
        opportunities += 1
        received += int(outgoing_contact(subsequent, 'left'))
        deadline = subsequent['t_s'] if subsequent else episode['elapsed_s']
        start = contacts[i-1]['t_s'] if i and outgoing_contact(contacts[i-1], 'left') else None
        if start is None:
            continue
        # Measured incoming lane before the next physical contact. A successful
        # interception before the reference line uses the actual contact lane.
        incoming = [f for f in frames if event['t_s'] <= f['t_s'] < deadline]
        cross = next((f for f in incoming if f['puck_xy'][0] <= cfg['intercept_x_m']), None)
        if cross:
            y = cross['puck_xy'][1]
        elif subsequent and subsequent['side'] == 'left':
            y = subsequent['puck_xy'][1]
        else:
            continue
        target = np.clip(y, -cfg['ready_y_limit_m'], cfg['ready_y_limit_m'])
        before = [f for f in frames if start <= f['t_s'] <= event['t_s']]
        if not before:
            continue
        errors = [abs(f['left_xy'][1]-target) for f in before]
        ready_errors.append(float(errors[-1]))
        # Sustained lateral readiness until opponent contact, at 20 Hz.
        lead = 0.
        for frame, error in reversed(list(zip(before, errors))):
            if error > READY_TOLERANCE_M:
                break
            lead = event['t_s']-frame['t_s']
        ready_leads.append(float(lead))
    return dict(returns=episode['alternating_returns'], reason=episode['reason'],
                opportunities=opportunities, received=received, lead_s=ready_leads,
                lateral_error_m=ready_errors, forecast_updates=len(episode['anticipation']))


def aggregate(pairs):
    report = {}
    for mode in ('baseline', 'supervised', 'candidate'):
        if mode not in pairs[0]:
            continue
        rows = [p[mode] for p in pairs]
        opportunities = sum(r['opportunities'] for r in rows)
        received = sum(r['received'] for r in rows)
        leads = [v for r in rows for v in r['lead_s']]
        errors = [v for r in rows for v in r['lateral_error_m']]
        report[mode] = dict(mean_returns=float(np.mean([r['returns'] for r in rows])),
                            received=received, opportunities=opportunities,
                            catch_rate=received/opportunities if opportunities else None,
                            ready_samples=len(leads), mean_lead_ms=float(np.mean(leads)*1000) if leads else None,
                            mean_error_mm=float(np.mean(errors)*1000) if errors else None,
                            forecast_updates=sum(r['forecast_updates'] for r in rows))
    return report
