# Method and evidence

## Design

After the left paddle executes a forward strike, the treatment changes its next ready position. Both arms retain J1. The right arm uses the same fixed style variation across all methods within a seed. Physical time continues between hits.

The 48-hidden-unit GRU consumes eight 20 Hz samples of the puck and both paddles, finite-difference velocities, and time since our last hit. Training targets are forward-return probability within 1.6 s, contact delay and location, and outgoing planar velocity. Training used 70 expert simulation episodes and 19 autonomous simulation episodes. Five additional autonomous episodes were used for model validation. The expert archive uses end-effector XY as a paddle-coordinate proxy. Online inputs use past and current simulator measurements and observed contacts.

The gated variant accepts a lane after two consecutive predictions differ by at most 12 cm. It uses a 2.5 cm deadband, holds the confirmed goal through conflicting or missing predictions, and moves the pre-positioning target at up to 0.95 m/s. V4 takes over after opponent contact.

## Diagnostic comparisons

Each run uses 12 seconds and a ±0.48 m lateral ready-position range. The prepare-speed gate and Cartesian/joint speed caps are disabled for this simulation ablation. Workspace and joint-position checks still apply. There is no early swing cancellation. Gating bounds only the anticipation target.

| Seed | J1 effective / longest active | Framewise effective / longest active | Gated effective / longest active |
|---|---:|---:|---:|
| 4100 | 12 / 12 | 16 / 16 | 15 / 9 |
| 4110 | 12 / 12 | 9 / 9 | 10 / 8 |
| 4111 | 6 / 4 | 3 / 3 | 12 / 9 |

Effective returns count new alternating forward physical contacts. An active streak requires uninterrupted, alternating forward strikes. A non-strike or backward contact breaks the streak, and a repeated same-side strike begins a new one. Pre-contact lateral speed averages the absolute finite difference of measured left-paddle Y while idle after a left strike and before the next contact, up to 1.6 seconds. Predicted target jump is the maximum lateral change between consecutive anticipation commands at 20 Hz. The reference emits no such commands.

These scenarios were used for development. The gated variant had more total returns in this comparison and shorter uninterrupted streaks on average than J1. Independent held-out opponents, camera tracking, and hardware-feasible actuation remain to be evaluated.

## Provenance and reproduction

The [core module](../src/opponent_anticipation.py) contains the supervised model, label generation and positioning rule. The [evidence directory](../evidence/) contains nine exported traces with contacts, sampled poses and anticipation commands. Each trace includes the original trace's SHA-256 and is an exported subset. Run `python scripts/summarize.py` from the repo root to reconstruct the showcase metrics.

Full rollout requires the Buaa-WAM SAPIEN environment, simulator assets, and trained V4/J1 checkpoints from the broader project. The standalone showcase does not bundle those dependencies. All videos are simulated seed-4100 recordings. The poster is a generated concept illustration.
