"""Generate the GLiNER2 economics card as a 1600x900 PNG."""

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BG = (13, 17, 23)
TEAL = (63, 185, 160)
WHITE = (230, 237, 243)
GRAY = (139, 148, 158)
DIM = (72, 79, 88)
DARK_BORDER = (33, 38, 45)
CARD_BG = (22, 27, 34)
BLUE = (88, 166, 255)

FONTS = "/tmp/fonts"
inter_bold = ImageFont.truetype(f"{FONTS}/inter_bold.ttf", 30)
inter_semi = ImageFont.truetype(f"{FONTS}/inter_semi.ttf", 18)
inter_med = ImageFont.truetype(f"{FONTS}/inter_med.ttf", 15)
inter_reg = ImageFont.truetype(f"{FONTS}/inter_reg.ttf", 15)
inter_reg_sm = ImageFont.truetype(f"{FONTS}/inter_reg.ttf", 13)
inter_reg_xs = ImageFont.truetype(f"{FONTS}/inter_reg.ttf", 11)
jbm_med = ImageFont.truetype(f"{FONTS}/jbm_med.ttf", 13)
jbm_med_sm = ImageFont.truetype(f"{FONTS}/jbm_med.ttf", 12)
jbm_med_xs = ImageFont.truetype(f"{FONTS}/jbm_med.ttf", 11)
jbm_reg = ImageFont.truetype(f"{FONTS}/jbm_reg.ttf", 12)
jbm_reg_xs = ImageFont.truetype(f"{FONTS}/jbm_reg.ttf", 11)

# Big number font — use Inter bold at maximum size
inter_giant = ImageFont.truetype(f"{FONTS}/inter_bold.ttf", 138)
inter_headline = ImageFont.truetype(f"{FONTS}/inter_bold.ttf", 28)
inter_sub = ImageFont.truetype(f"{FONTS}/inter_semi.ttf", 17)
jbm_metric = ImageFont.truetype(f"{FONTS}/jbm_med.ttf", 30)
jbm_label = ImageFont.truetype(f"{FONTS}/jbm_reg.ttf", 12)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# ── Left side ──
LX = 80
y = 68

# Eyebrow
d.text((LX, y), "STRUCTURED EXTRACTION ECONOMICS", font=jbm_med, fill=TEAL)
y += 36

# Main number
d.text((LX - 4, y), "$0.004", font=inter_giant, fill=TEAL)
y += 148

# Headline
d.text((LX, y), "GLiNER2 at ~$0.004 per 1M input tokens", font=inter_headline, fill=WHITE)
y += 38

# Subheadline
d.text((LX, y), "DeBERTa GLiNER2", font=inter_semi, fill=(201, 209, 217))
bx = d.textlength("DeBERTa GLiNER2", font=inter_semi)
d.text((LX + bx, y), "  served with vLLM Factory", font=inter_sub, fill=GRAY)
y += 38

# Support line — border left
d.line([(LX, y), (LX, y + 40)], fill=DARK_BORDER, width=2)
d.text((LX + 16, y), "Zero-shot NER, classification, and schema-driven extraction", font=inter_reg, fill=GRAY)
d.text((LX + 16, y + 22), "at 40 req/s on a single RTX A5000", font=inter_reg, fill=GRAY)
y += 56

# Micro-labels — draw teal dots instead of checkmarks
labels = ["scored outputs", "schema-constrained", "dataset-scale friendly"]
mx = LX + 18
for label in labels:
    # Teal dot
    dot_y = y + 7
    d.ellipse([(mx, dot_y), (mx + 6, dot_y + 6)], fill=TEAL)
    d.text((mx + 12, y), label, font=inter_med, fill=BLUE)
    mx += d.textlength(label, font=inter_med) + 32
y += 30

# Muted note
d.text((LX + 18, y), "No token-out cost. No free-form generation loop.", font=inter_reg_sm, fill=DIM)

# ── Divider ──
div_x = 1040
d.line([(div_x, 68), (div_x, H - 50)], fill=DARK_BORDER, width=1)

# ── Right side ──
RX = 1080
ry = 68

# Proof title
d.text((RX, ry), "PRODUCTION SNAPSHOT", font=jbm_med_sm, fill=GRAY)
ry += 36

# Metric stack — 4 rows
metrics = [
    ("40 req/s", "RTX A5000 · 512 tokens/req"),
    ("144k pages/hr", "1 page ≈ 512 tokens"),
    ("~73M tok/hr", "~20k tokens/s sustained"),
    ("$0.28/hr", "Runpod → ~$2 / 1M pages"),
]

card_w = 440
row_h = 72
for i, (val, label) in enumerate(metrics):
    rx1, ry1 = RX, ry
    rx2, ry2 = RX + card_w, ry + row_h

    # Background
    if i == 0:
        d.rounded_rectangle([(rx1, ry1), (rx2, ry1 + 8)], radius=8, fill=CARD_BG)
        d.rectangle([(rx1, ry1 + 4), (rx2, ry2)], fill=CARD_BG)
    elif i == len(metrics) - 1:
        d.rectangle([(rx1, ry1), (rx2, ry2 - 4)], fill=CARD_BG)
        d.rounded_rectangle([(rx1, ry2 - 8), (rx2, ry2)], radius=8, fill=CARD_BG)
    else:
        d.rectangle([(rx1, ry1), (rx2, ry2)], fill=CARD_BG)

    # Border lines
    d.line([(rx1, ry1), (rx2, ry1)], fill=DARK_BORDER, width=1)
    d.line([(rx1, ry1), (rx1, ry2)], fill=DARK_BORDER, width=1)
    d.line([(rx2, ry1), (rx2, ry2)], fill=DARK_BORDER, width=1)
    if i == len(metrics) - 1:
        d.line([(rx1, ry2), (rx2, ry2)], fill=DARK_BORDER, width=1)

    # Value
    d.text((rx1 + 22, ry1 + 14), val, font=jbm_metric, fill=WHITE)
    # Label
    d.text((rx1 + 22, ry1 + 48), label, font=jbm_label, fill=GRAY)
    ry += row_h

ry += 16

# Disclaimer box
disc_h = 50
d.rounded_rectangle([(RX, ry), (RX + card_w, ry + disc_h)], radius=6, fill=BG, outline=DARK_BORDER)
d.text((RX + 16, ry + 10), "Not directly comparable to token-in / token-out", font=inter_reg_xs, fill=DIM)
d.text((RX + 16, ry + 28), "generation pricing. This is extraction, not completion.", font=inter_reg_xs, fill=DIM)

# ── Bottom pills ──
pill_y = H - 76
pill_x = LX
pill_labels = ["NER", "Classification", "Structured extraction", "Served via vllm serve"]
for plabel in pill_labels:
    tw = d.textlength(plabel, font=jbm_med_sm)
    pw = int(tw) + 30
    ph = 28
    d.rounded_rectangle(
        [(pill_x, pill_y), (pill_x + pw, pill_y + ph)],
        radius=14, fill=CARD_BG, outline=(48, 54, 61)
    )
    d.text((pill_x + 15, pill_y + 6), plabel, font=jbm_med_sm, fill=(201, 209, 217))
    pill_x += pw + 10

# ── Footer ──
foot_y = H - 30
d.text((LX, foot_y), "RTX A5000 · bf16 · page ≈ 512 tokens · Runpod $0.28/hr · vllm-factory benchmark", font=jbm_reg_xs, fill=(48, 54, 61))
d.text((W - 70 - d.textlength("same schema, no probabilistic generation", font=jbm_reg_xs), foot_y),
       "same schema, no probabilistic generation", font=jbm_reg_xs, fill=DIM)

out = "/workspace/vllm-factory/assets/social/gliner2-economics-card.png"
img.save(out, "PNG")
print(f"Saved: {out}  ({W}x{H})")
