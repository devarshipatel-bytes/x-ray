"""Material-proxy channels derived from pseudo-color X-ray LUT.

IMPORTANT (honest scope): public X-ray datasets ship *pseudo-color RGB*, not the raw
high/low-energy sinograms, so we CANNOT compute a true effective atomic number (Z_eff).
What we CAN do is invert the (roughly universal) security-scanner color convention:

    orange  -> organic / low-Z   (explosives, drugs, food, plastics)
    blue    -> metal / high-Z    (knives, guns, wires)
    green   -> mixed / medium-Z
    dark    -> thick / dense (high attenuation)

We turn the RGB pseudo-color into two physically-motivated *proxy* channels
(organic_low_z, metal_high_z) in [0, 1]. These are appended to RGB so the detector
gets an explicit material cue instead of having to re-learn the color convention.

This is a proxy, not physics. When real dual-energy raw data is available, replace
`rgb_to_material` with an actual basis-material / Z_eff decomposition; the rest of the
pipeline (5-channel input) is unchanged.
"""
from __future__ import annotations

import numpy as np


def rgb_to_material(rgb: np.ndarray) -> np.ndarray:
    """Map an HxWx3 uint8 (or float[0,1]) RGB pseudo-color image to HxWx2 float32 proxies.

    Returns channels stacked as [organic_low_z, metal_high_z], each in [0, 1].
    """
    x = rgb.astype(np.float32)
    if x.max() > 1.5:  # uint8 -> [0,1]
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]

    # Attenuation: darker pixel => more/denser material in the path (transmissive X-ray).
    intensity = (r + g + b) / 3.0
    attenuation = 1.0 - intensity                      # 0 = air, 1 = fully blocked

    # Color opponency: orange(=organic) has R>>B; blue(=metal) has B>>R.
    organic_hue = np.clip(r - b, 0.0, 1.0)             # orange-ness
    metal_hue = np.clip(b - r, 0.0, 1.0)               # blue-ness

    # Weight the material hue by how much material is actually there (attenuation),
    # so empty/bright background stays ~0 in both proxy channels.
    organic = organic_hue * attenuation
    metal = metal_hue * attenuation

    # Normalize to a comfortable range (hue*attenuation maxes well below 1).
    organic = np.clip(organic * 2.0, 0.0, 1.0)
    metal = np.clip(metal * 2.0, 0.0, 1.0)

    return np.stack([organic, metal], axis=-1).astype(np.float32)


def material_colormap(material: np.ndarray) -> np.ndarray:
    """Visualization helper: HxWx2 proxies -> HxWx3 uint8 false-color for figures.

    organic -> warm (orange), metal -> cool (blue). Used by the viz suite only.
    """
    organic = material[..., 0]
    metal = material[..., 1]
    r = organic
    g = 0.3 * organic
    b = metal
    img = np.stack([r, g, b], axis=-1)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
