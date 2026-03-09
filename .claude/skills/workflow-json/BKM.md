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

### Available Transforms
| Transform | Key Param | Typical Range | Notes |
|-----------|-----------|---------------|-------|
| `zoom_in` | `zoom_amt` | 0.01-0.09 | Center crop. See Rule 3. |
| `zoom_in_left/right/up/down` | `zoom_amt` | 0.01-0.09 | Directional zoom. |
| `fold_vertical` | `fold_amt` | — | Mirror effect. |
| `fold_horizontal` | `fold_amt` | — | Mirror effect. |
| `squeeze_wide` | `squeeze_amt` | 0.005-0.025 | Very small values! |
| `squeeze_tall` | `squeeze_amt` | 0.005-0.025 | Very small values! |
| `fisheye` | `strength` | 0.1-0.21 | Barrel distortion. Can be negative. |
| `rotate` | `angle` | 1.5-5.0 (degrees) | Per-frame rotation. |
| `wave` | `strength`, `period`, `rate` | 8-12, 30, 4-8 | Sine wave distortion. |
| `perspective` | `strength`, `shrink_edge` | 10-35, bottom/left/top/right | 3D warp. |
| `paste_img` | path, position, scale | — | Composite another image. |

### Transform Stacking
Transforms apply in sequence. Common combos:
- `fisheye` + `zoom_in` - distort then zoom
- `wave` + `zoom_in` - psychedelic ocean
- `squeeze_*` + `zoom_in` - compression effect
- `perspective` + `zoom_in` - 3D depth

Transform parameters can also be expressions.

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
