# models/pfms.py
import tensorflow as tf
import math

# ---------- sRGB -> XYZ (D65) constants ----------
# This matrix is the standard linear transformation from sRGB to CIE XYZ color space,
# defined by the IEC 61966-2-1:1999 specification (the official sRGB standard).
# - Input must be linear RGB (after gamma correction).
# - Output is CIE XYZ under D65 illuminant (standard daylight white).
# Why needed?
#   Because CIE Lab (L*, a*, b*) is defined in terms of XYZ, not directly from RGB.
#   So the conversion pipeline is:
#       sRGB → (undo gamma) → linear RGB → (this matrix) → XYZ → Lab
# In our case, we mainly want the a* and b* channels for perceptual features,
# but this step is essential for correct Lab computation.
_SRGB_TO_XYZ = tf.constant([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=tf.float32)

# D65 reference white point (used for XYZ → Lab normalization)
_Xn, _Yn, _Zn = 0.95047, 1.0, 1.08883

# -------- sRGB -> linear RGB -> XYZ -> Lab(a,b) --------
def _srgb_to_linear(s):
    """Inverse companding: sRGB [0,1] -> linear RGB [0,1]."""
    a = 0.055
    return tf.where(s <= 0.04045, s / 12.92, tf.pow((s + a) / (1.0 + a), 2.4))

def _rgb_to_xyz(rgb_lin):
    """Per-pixel (H,W,3)*(3,3) -> (H,W,3) XYZ under D65."""
    return tf.tensordot(rgb_lin, _SRGB_TO_XYZ, axes=[[-1], [1]])

def _f_lab(t):
    """CIE Lab helper nonlinearity."""
    delta = 6.0 / 29.0
    t0 = delta ** 3
    t1 = (1.0/3.0) * (29.0/6.0) ** 2
    t2 = 4.0 / 29.0
    return tf.where(t > t0, tf.pow(t, 1.0/3.0), t1 * t + t2)

def _rgb_to_lab_ab(rgb01):
    """
    RGB [0,1] -> Lab(a,b), return a_norm, b_norm, mag01:
      a_norm = a/128, b_norm = b/128   (≈ [-1,1])
      mag01  = clip( sqrt(a_norm^2 + b_norm^2) / sqrt(2), 0, 1 )
    """
    rgb01 = tf.clip_by_value(rgb01, 0.0, 1.0)
    rgb_lin = _srgb_to_linear(rgb01)
    X, Y, Z = tf.unstack(_rgb_to_xyz(rgb_lin), axis=-1)

    xr, yr, zr = X/_Xn, Y/_Yn, Z/_Zn
    fx, fy, fz = _f_lab(xr), _f_lab(yr), _f_lab(zr)

    # Lab (L is not used by design)
    a = 500.0 * (fx - fy)   # ~[-128,127] (not exact bounds, but typical)
    b = 200.0 * (fy - fz)

    a_n = a / 128.0
    b_n = b / 128.0

    # chroma magnitude in a/b plane; normalized to [0,1] using sqrt(2) bound
    mag = tf.sqrt(tf.maximum(a_n*a_n + b_n*b_n, 1e-8))
    mag01 = tf.clip_by_value(mag / math.sqrt(2.0), 0.0, 1.0)
    return a_n, b_n, mag01


# -------- utilities --------
def _stack3(x):
    """(H,W,1) -> (H,W,3) by channel repeat; pass-through if already 3."""
    if x.shape.rank == 3 and x.shape[-1] == 3:
        return x
    return tf.repeat(x, repeats=3, axis=-1)

def _minmax01_per_image(x, eps=1e-8):
    """Per-image min-max normalize a single map x: (H,W,1)->[0,1]."""
    x = tf.convert_to_tensor(x)
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    return (x - x_min) / tf.maximum(x_max - x_min, eps)


# -------- PFMs (a/b only) --------
def build_pfms(rgb_float01: tf.Tensor):
    """
    Input:
      rgb_float01: (H, W, 3) float32 in [0,1]

    Output:
      List of 5 tensors, each (H, W, 3) float32:
        [ LD, CF, B1, B2, B3 ]
      - LD:  Sobel edge magnitude of mag01 (replicated to 3 channels)
      - CF:  stack of [a_norm, b_norm, mag01]
      - B1:  (mag01 < 1/3) mask, replicated to 3 channels
      - B2:  (1/3 <= mag01 < 2/3) mask, replicated to 3 channels
      - B3:  (mag01 >= 2/3) mask, replicated to 3 channels
    """
    a_n, b_n, mag01 = _rgb_to_lab_ab(rgb_float01)        # each (H,W)

    # ---- LD: edge magnitude on mag01 ----
    mag1 = mag01[..., None]                              # (H,W,1)
    sob = tf.image.sobel_edges(mag1[None, ...])          # (1,H,W,1,2)
    sob = sob[0]                                         # (H,W,1,2)
    gx, gy = sob[..., 0], sob[..., 1]                    # (H,W,1)
    grad = tf.sqrt(tf.maximum(gx*gx + gy*gy, 1e-12))     # (H,W,1)
    ld01 = _minmax01_per_image(grad)                     # [0,1]
    LD = _stack3(ld01)                                   # (H,W,3)

    # ---- CF: a/b/mag stack ----
    CF = tf.stack([a_n, b_n, mag01], axis=-1)            # (H,W,3)

    # ---- Bands on mag01 ----
    m = mag01[..., None]                                 # (H,W,1)
    B1 = tf.cast(m < 1.0/3.0, tf.float32)
    B2 = tf.cast((m >= 1.0/3.0) & (m < 2.0/3.0), tf.float32)
    B3 = tf.cast(m >= 2.0/3.0, tf.float32)
    B1 = _stack3(B1); B2 = _stack3(B2); B3 = _stack3(B3)

    return [LD, CF, B1, B2, B3]