# ComfyUI Looper Workflow JSON - Best Known Methods (BKM)

## Rules

### Rule 1: Minimum Denoise Threshold
`denoise_amt` below 0.4 does basically nothing. Don't waste iterations on values that low unless you have a specific reason (e.g. a ramp-up expression that passes through low values briefly).

### Rule 2: Scene Transition Technique
When moving from one scene to another, temporarily bump `denoise_amt` on the first frame of the new section. This helps the model "break away" from the previous scene. Common pattern:
```
"0.8 if (n - offset) == 0 else <other expression>"
```
This makes the first frame use 0.8 (strong generation) while subsequent frames use a lower/dynamic value.

### Rule 3: Aggressive Zoom Degrades Edges
Zooming in too aggressively causes details to be lost at the edges, sometimes permanently. The lost edge content gets filled with artifacts that compound over iterations. Keep `zoom_amt` conservative (0.01-0.03 for subtle, max ~0.09 for dramatic effect).

### Rule 4: Emphasis Frame Pattern
Use single-iteration sections (loop_iterations: 1) with high denoise (0.7-0.8) at major narrative beats. Follow with a multi-iteration section (30-80 frames) at lower denoise (0.55-0.65) to let the scene "breathe" and evolve.

### Rule 5: Sine Oscillation for Organic Feel
Using `sin()` in denoise_amt expressions creates a breathing/pulsing effect that prevents the animation from feeling static. Common patterns:
- `"0.6 + 0.15*sin(0.4*n)"` - subtle pulse
- `"0.55 + 0.3*sin(0.3*n)"` - more dramatic oscillation
The frequency (0.2-0.4) controls how fast the pulse cycles. Keep amplitude reasonable so denoise doesn't drop below 0.4 (Rule 1).

### Rule 6: CFG Oscillation with relu()
Use `relu()` to keep CFG from dropping below a baseline:
```
"8 + relu(3*sin(n))"
```
This oscillates between 8 and 11, never going below 8. Adds variety without destabilizing generation.

---

## JSON Structure

### Top Level
```json
{
    "all_settings": [ ...array of LoopSettings... ],
    "version": 1
}
```

### LoopSettings Fields
| Field | Type | Notes |
|-------|------|-------|
| `loop_iterations` | int | Frames this section generates. Required per section. |
| `seed` | int | Random seed. Omit to inherit or let randomize. |
| `checkpoint` | string | Model checkpoint filename. Usually only set in first section. |
| `prompt` | string | Positive prompt. |
| `neg_prompt` | string | Negative prompt (optional). |
| `denoise_steps` | int/expr | Number of denoising steps. SDXL: 30-45, SD3.5: 8, Flux: 5-10. |
| `denoise_amt` | float/expr | Denoising strength 0.0-1.0. See rules above. |
| `cfg` | float/expr | Classifier-free guidance. Typical range: 4.0-8.0. |
| `canny` | object/null | Edge detection: `{low_thresh, high_thresh, strength}`. Set to `null` or `[]` to disable. |
| `con_deltas` | list | Conditional deltas: `{pos, neg, strength}`. Strength can be an expression. |
| `loras` | list | LoRA adapters: `{name, strength}`. Strength typically 0.7-1.1. |
| `transforms` | list | Image transforms applied per iteration. See Transforms section. |

### Inheritance
Fields omitted from a section inherit from the previous section. This is the primary mechanism for reducing redundancy. Only specify what changes between sections.

### Comments
JSON files support `//` line comments, which are stripped during parsing.

---

## Expressions

### Variables
- `n` - Current iteration index (0-based, global across all sections)
- `offset` - The iteration number where the current section starts
- `total_n` - Total iterations across all sections
- `(n - offset)` - Local iteration index within current section (0-based)

### Available Functions
`sin`, `cos`, `sqrt`, `floor`, `ceil`, `abs`, `relu` (max(0, x)), `get_power_at_freq_range(low_f, high_f)` (FFT audio)

### Common Expression Patterns

**Linear ramp:**
```
"0.2 + (0.5/35)*n"       // 0.2 -> 0.7 over 35 total iterations
```

**Sine oscillation:**
```
"0.6 + 0.15*sin(0.4*n)"  // Pulse around 0.6
```

**Conditional first-frame bump (Rule 2):**
```
"0.8 if (n - offset) == 0 else 0.6"
```

**Phase-based conditional:**
```
"0.005 if (n - offset) < 35 else 0"   // Apply for first 35 frames only
"1.0 if (n - offset) <= 50 else -1.0" // Switch direction mid-section
```

**CFG with floor via relu (Rule 6):**
```
"8 + relu(3*sin(n))"
```

---

## Transforms

All transform parameters marked with (expr) support expression strings. Transforms are applied in sequence — order matters.

### Zoom Transforms
Crop a portion of the image and resize back to original dimensions. Creates a "camera moving" effect over iterations.

**`zoom_in`** — Crop from center equally on all sides.
- `zoom_amt` (expr): Fraction to remove. 0.01-0.03 subtle, 0.05-0.09 dramatic. See Rule 3.
```json
{"name": "zoom_in", "zoom_amt": 0.03}
```

**`zoom_in_left`** — Crop from the right edge, keeping left content. Camera pans left.
- `zoom_amt` (expr)

**`zoom_in_right`** — Crop from the left edge, keeping right content. Camera pans right.
- `zoom_amt` (expr)

**`zoom_in_up`** — Crop from the bottom edge, keeping top content. Camera pans up.
- `zoom_amt` (expr)

**`zoom_in_down`** — Crop from the top edge, keeping bottom content. Camera pans down.
- `zoom_amt` (expr)

### Fold Transforms
Remove a strip from the center and push the two halves together, then resize. Creates a narrowing/compressing mirror effect.

**`fold_vertical`** — Remove a vertical strip from the center, push left and right halves together.
- `fold_amt` (expr): Width in pixels to remove (split evenly from center).
```json
{"name": "fold_vertical", "fold_amt": 20}
```

**`fold_horizontal`** — Remove a horizontal strip from the center, push top and bottom together.
- `fold_amt` (expr): Height in pixels to remove.

### Squeeze Transforms
Crop from the center and stretch back to original size. Similar to zoom but only along one axis, creating a stretching/squashing effect.

**`squeeze_wide`** — Crop horizontally from center, stretch width back. Makes image appear wider/stretched.
- `squeeze_amt` (expr): Fraction to remove. Use very small values: 0.005-0.025.
```json
{"name": "squeeze_wide", "squeeze_amt": 0.01}
```

**`squeeze_tall`** — Crop vertically from center, stretch height back. Makes image appear taller/stretched.
- `squeeze_amt` (expr): Same range as squeeze_wide.

### Distortion Transforms

**`fisheye`** — Barrel distortion radiating from center. Positive values push outward (convex), negative values pull inward (concave).
- `strength` (expr): Distortion intensity. 0.1 mild, 0.21 strong. Can be negative for inverse.
```json
{"name": "fisheye", "strength": 0.15}
```

**`spiral`** — Twists the image around its center. Rotation amount increases with distance from center.
- `strength` (expr): Twist intensity in radians. Positive = counterclockwise. Try 0.5-2.0.
```json
{"name": "spiral", "strength": 1.0}
```

**`wave`** — Sine wave distortion applied as a mesh deformation. Direction alternates automatically based on iteration count (`n`) and `rate`.
- `strength` (expr): Wave amplitude in pixels. 8-12 typical.
- `period` (expr): Wavelength in pixels. 30 is common.
- `rate` (expr): Controls direction alternation speed. 4-8 typical.
```json
{"name": "wave", "strength": 10, "period": 30, "rate": 6}
```

**`ripple`** — Concentric circular wave distortion radiating from center. Like dropping a stone in water.
- `amplitude` (expr): Wave height in pixels.
- `wavelength` (expr): Distance between ripple rings in pixels.
- `phase` (expr, optional): Phase offset. Use expressions like `0.5*n` for animated ripples.
```json
{"name": "ripple", "amplitude": 5, "wavelength": 40, "phase": "0.5*n"}
```

**`elastic`** — Random smooth deformation (like viewing through textured glass). Uses Gaussian-filtered random displacement.
- `strength` (expr): Displacement magnitude.
- `smoothness` (expr): Gaussian sigma — higher = smoother/larger-scale warps.
- `seed` (expr, optional): Random seed for reproducibility.
```json
{"name": "elastic", "strength": 30, "smoothness": 5}
```

### Geometric Transforms

**`rotate`** — Rotate around center by a fixed angle per frame. Automatically crops to the largest inscribed rectangle (no black borders). Over many frames creates continuous rotation.
- `angle` (expr): Degrees per frame. 1.5-5.0 typical. Negative for clockwise.
```json
{"name": "rotate", "angle": 2.0}
```

**`perspective`** — 3D perspective warp that shrinks one edge, creating a receding-into-distance effect.
- `strength` (expr): How many pixels to indent the shrinking edge. 10-35 typical.
- `shrink_edge`: Which edge to shrink: `"top"`, `"bottom"`, `"left"`, or `"right"`. NOT an expression.
```json
{"name": "perspective", "strength": 20, "shrink_edge": "bottom"}
```

**`pan`** — Shift the entire image by pixel offsets. Wraps around (pixels that go off one edge appear on the opposite side).
- `dx` (expr): Horizontal shift in pixels. Positive = shift right.
- `dy` (expr): Vertical shift in pixels. Positive = shift down.
```json
{"name": "pan", "dx": 5, "dy": 0}
```

**`mirror`** — Copy one half of the image onto the other half (flipped). Creates symmetry.
- `mode`: One of `"left_to_right"`, `"right_to_left"`, `"top_to_bottom"`, `"bottom_to_top"`. NOT an expression.
```json
{"name": "mirror", "mode": "left_to_right"}
```

**`kaleidoscope`** — Radial symmetry effect. Maps all angles into repeating wedges with alternating mirroring.
- `segments` (expr): Number of symmetry segments. Must be >= 2. 4-8 typical.
```json
{"name": "kaleidoscope", "segments": 6}
```

### Color Transforms

**`hue_shift`** — Rotate all colors around the color wheel by a fixed amount.
- `shift_deg` (expr): Degrees to shift (0-360). Use expressions like `5*n` for progressive rainbow shift.
```json
{"name": "hue_shift", "shift_deg": "5*n"}
```

**`color_channel_offset`** — Shift R, G, B channels independently along the horizontal axis. Creates chromatic aberration / glitch effects.
- `r_offset` (expr): Red channel horizontal shift in pixels.
- `g_offset` (expr): Green channel horizontal shift in pixels.
- `b_offset` (expr): Blue channel horizontal shift in pixels.
```json
{"name": "color_channel_offset", "r_offset": 5, "g_offset": 0, "b_offset": -5}
```

**`contrast_brightness`** — Adjust contrast and brightness.
- `contrast` (expr): Multiplier. 1.0 = no change, >1.0 = more contrast, <1.0 = less.
- `brightness` (expr): Additive offset. 0.0 = no change, positive = brighter, negative = darker.
```json
{"name": "contrast_brightness", "contrast": 1.2, "brightness": 0.05}
```

### Composite Transforms

**`paste_img`** — Blend another image on top of the current frame. The paste image is resized to match frame dimensions, then alpha-blended.
- `img_path`: Absolute path to the image file. NOT an expression.
- `opacity` (expr): Blend factor 0.0-1.0. 0.0 = all original, 1.0 = all paste image.
```json
{"name": "paste_img", "img_path": "/path/to/overlay.png", "opacity": 0.3}
```

**`blend_ref`** — Advanced reference image blending with blend modes, gradient masks, and easing. Superior to `paste_img` for scene transitions where you want AI to create novel content guided by a reference image's layout.
- `img_path`: Absolute path to the reference image. NOT an expression.
- `opacity` (expr): Blend strength 0.0-1.0.
- `blend_mode` (optional): `"normal"` (default), `"overlay"`, `"soft_light"`, `"screen"`, `"multiply"`. Overlay and soft_light preserve base structure while incorporating reference layout — best for guiding diffusion.
- `mask` (optional): Spatial gradient mask. `"uniform"` (default), `"radial"` (center-out), `"horizontal"` (left-to-right), `"vertical"` (top-to-bottom).
- `mask_invert` (optional): `true` to reverse the mask gradient direction. Default `false`.
- `easing` (optional): Easing curve applied to opacity. `"linear"` (default), `"ease_in"`, `"ease_out"`, `"ease_in_out"`.
```json
{"name": "blend_ref", "img_path": "/path/to/scene.png", "opacity": "(n-offset)/30", "blend_mode": "overlay", "easing": "ease_in_out"}
```

### Retro/Stylization

**`pixelate`** — Downscale then upscale with nearest-neighbor interpolation for a blocky pixel-art look.
- `block_size` (expr): Size of each pixel block. 2-4 subtle, 8-16 dramatic.
```json
{"name": "pixelate", "block_size": 8}
```

### Transform Stacking
Transforms apply in the order listed. Common combos:
- `fisheye` + `zoom_in` — distort then zoom (prevents fisheye edge artifacts from accumulating)
- `wave` + `zoom_in` — psychedelic ocean effect
- `squeeze_*` + `zoom_in` — compression with forward motion
- `perspective` + `zoom_in` — 3D depth tunnel
- `hue_shift` + `spiral` — psychedelic color vortex
- `mirror` + `zoom_in` — symmetric tunnel effect

Transform parameters can be expressions, enabling animated transforms (e.g. `"angle": "2 if (n - offset) < 30 else -2"`).

---

## LoRA Usage
- **Style LoRAs** (e.g. PsyAI): strength 1.0-1.1, used across most scenes
- **Detail LoRAs** (e.g. detailer): strength 1.0
- **Thematic LoRAs**: strength 0.4-1.0, changed per scene
- Carry 2-4 LoRAs through multiple scenes for style coherence

---

## Prompt Tips
- Lead with mood/style: "lsd", "dmt", "psychedelic", "trip"
- Core subject next: character, creature, scene
- End with quality modifiers: "detailed", "4k", "photorealistic"
- Keep prompts concise. Rarely exceed 150 tokens.
- Negative prompts are optional but useful for correcting recurring issues (e.g. "blurry, overexposed")

---

## Workflow Design Patterns

### Narrative Arc
Structure workflows as a story with 8-16 scenes. Each scene: 1-100 iterations (typically 30-80). Pacing comes from varying iteration counts and denoise intensity.

### Scene Transition Checklist
1. Change the prompt
2. Bump denoise on first frame (Rule 2)
3. Optionally change transforms/LoRAs
4. Consider a 1-frame emphasis section before the main section (Rule 4)

### Model-Specific Defaults
| Model | denoise_steps | Notes |
|-------|---------------|-------|
| SDXL | 30-45 | Most flexible |
| SD3.5 | 8 | Turbo mode |
| Flux.1D | 5-10 | Fast, beta scheduler |
