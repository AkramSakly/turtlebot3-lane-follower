"""Pair two marked boundaries in the same rows of the camera image."""
import numpy as np

SIZE = (256, 192)


def stripe_centers(row):
    starts, ends = np.flatnonzero(np.diff(np.r_[False, row, False])).reshape(-1, 2).T
    return [(a+b-1)/2 for a,b in zip(starts, ends) if 2 <= b-a <= 24]


def lane_band(yellow, white, top, bottom, side=1):
    """side=+1: yellow left of white. side=-1: yellow right of white (inverse direction)."""
    h, w = yellow.shape
    samples = []
    for y in range(int(top*h), int(bottom*h)):
        left = stripe_centers(yellow[y])
        right = stripe_centers(white[y])
        pairs = [(a,b) for a in left for b in right if .17*w <= side*(b-a) <= .91*w]
        if len(pairs) == 1:
            samples.append((y,*pairs[0]))
    if len(samples) < max(4, .35*(bottom-top)*h):
        return None
    points = np.array(samples)
    med = np.median(points, axis=0)
    # Reject bands where support jumps between different line markings.
    if np.max(np.median(np.abs(points[:,1:]-med[1:]), axis=0)) > .055*w:
        return None
    return tuple(float(k) for k in med) + (len(samples),)


def _geometry_for_side(yellow, white, side):
    # Find adjacent supported bands when a curve leaves the fixed lookahead area.
    bands = [lane_band(yellow, white, t, t+.05, side) for t in (.90,.85,.80,.75,.70,.65)]
    for near, ahead in zip(bands, bands[1:]):
        if near is None or ahead is None:
            continue
        _, a, b, n1 = near
        _, c, d, n2 = ahead
        if abs((a+b-c-d)/2) <= yellow.shape[1]*.18 and .45 <= (d-c)/(b-a) <= 1.55:
            return near[:3], ahead[:3], n1+n2
    return None


def lane_geometry(yellow, white, mode='auto'):
    """mode: 'yellow_left', 'yellow_right' or 'auto' (pick the better-supported ordering)."""
    sides = {'yellow_left': (1,), 'yellow_right': (-1,), 'auto': (1, -1)}[mode]
    found = [g for g in (_geometry_for_side(yellow, white, s) for s in sides) if g is not None]
    if not found:
        return None
    best = max(found, key=lambda g: g[2])
    return best[0], best[1]


def lane_side_of(geometry):
    """+1 if yellow is left of white in this geometry, -1 otherwise."""
    near = geometry[0]
    return 1 if near[2] > near[1] else -1


def recovery(yellow, white, side, min_pixels=40):
    """Steering target when the two lines cannot be paired (e.g. inside a tight curve).

    Uses the lower 45% of the image. side=+1 means yellow is on the left.
    Returns the target x in pixels, or None when no line is usable."""
    h, w = yellow.shape
    top = int(.55*h)
    ys, xs = np.nonzero(yellow[top:])
    ny, yx = len(xs), (xs.mean() if len(xs) else 0.)
    ys, xs = np.nonzero(white[top:])
    nw, wx = len(xs), (xs.mean() if len(xs) else 0.)
    if ny >= min_pixels and nw >= min_pixels:
        return (yx + wx) / 2
    if ny >= min_pixels:
        return yx + side*.35*w        # only yellow: lane lies on the white side
    if nw >= min_pixels:
        return wx - side*.35*w        # only white: lane lies on the yellow side
    return None


def _scale(speed):
    """Steering gains were tuned at <=0.05 m/s. Faster -> proportionally stronger turning
    (same path curvature), capped at 1.5 rad/s."""
    return max(1., speed/.05)


def recovery_command(target_x, width, speed=.035):
    """Smooth turn toward the target: no pivot, slower when turning hard."""
    k = _scale(speed)
    lim = min(1.5, .8*k)
    yaw = float(np.clip(k*2.0*(.5 - target_x/(width-1)), -lim, lim))
    return .5*speed*(1-.6*abs(yaw)/lim), yaw


def command(geometry, width, speed=.035):
    if geometry is None:
        return 0., 0.
    near, ahead = geometry
    center = (near[1]+near[2])/(2*(width-1))
    lookahead = (ahead[1]+ahead[2])/(2*(width-1))
    k = _scale(speed)
    lim = min(1.5, .55*k)
    yaw = float(np.clip(k*(1.7*(.5-center) + 1.3*(center-lookahead)), -lim, lim))
    return speed*(1-.65*abs(yaw)/lim), yaw
