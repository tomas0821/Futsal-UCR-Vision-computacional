"""Annotated example stills from a tracked half: (1) frames where people tracking is complete, (2) frames where the ball
is missed (interpolated) or barely detected, with a 4x zoom inset at the ball position to show how small it is.
Usage: python example_frames.py video_1080p.mp4 track_details.txt stitched.csv ball_track.csv out_prefix [--fps 50]
(stitched ids are joined back to the original track ids through track_pitch_coords_teams.csv next to the stitched file)
Outputs out_prefix_people_good.jpg and out_prefix_ball_struggles.jpg (plus the individual frames).
"""
import sys, csv, subprocess, collections, io
from PIL import Image, ImageDraw, ImageFont
vid, trk, sti, bal, out = sys.argv[1:6]; fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18); FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
import os
key2tid = {(r["frame_number"], r["confidence"], r["pitch_x_m"], r["pitch_y_m"]): r["track_id"] for r in csv.DictReader(open(os.path.join(os.path.dirname(sti), "track_pitch_coords.csv")))}
lab = {}
for r in csv.DictReader(open(sti)):
    t = key2tid.get((r["frame_number"], r["confidence"], r["pitch_x_m"], r["pitch_y_m"]))
    if t: lab[(int(r["frame_number"]), t)] = r["player_id"]
boxes = collections.defaultdict(list)
for r in csv.DictReader(open(trk)):
    f = int(r["frame_number"]); boxes[f].append((lab.get((f, r["track_id"]), "GK" if r["class_name"]=="port" else r["class_name"]), float(r["x_center"]), float(r["y_center"]), float(r["width"]), float(r["height"]), float(r["confidence"]), r["class_name"]))
ball = {int(r["frame_number"]): (float(r["x_px"])/2, float(r["y_px"])/2, r["source"], float(r["confidence"] or 0), float(r["pitch_x_m"] or 0), float(r["pitch_y_m"] or 0)) for r in csv.DictReader(open(bal))}
# --- people: frames with 4 A + 4 B labelled players, 2 GK, >=1 ref, highest mean confidence; 2 frames >= 8 min apart
cand = []
for f, bs in boxes.items():
    ids = [b[0] for b in bs]; a = sum(i.startswith("A") for i in ids); b = sum(i.startswith("B") for i in ids); g = sum(i.startswith("GK") for i in ids); rf = sum(x[6]=="ref" for x in bs)
    if a == 4 and b == 4 and g == 2 and rf >= 1 and len(bs) <= 12: cand.append((sum(x[5] for x in bs)/len(bs), f))
cand.sort(reverse=True); people = []
for c, f in cand:
    if all(abs(f-p) > 8*60*fps for p in people): people.append(f)
    if len(people) == 2: break
# --- ball: longest interpolated gaps while the ball moves (>3 m over the gap), and lowest-confidence detections
frames = sorted(ball); gaps = []   # (gap frames, mid frame, last-seen frame, next-seen frame, distance m): no ball row at all
for a, b in zip(frames, frames[1:]):
    if b - a > 1:
        d = ((ball[b][4]-ball[a][4])**2 + (ball[b][5]-ball[a][5])**2) ** .5
        if d > 5 and ball[a][2] == "det" and ball[b][2] == "det": gaps.append((b-a-1, (a+b)//2, a, b, d))
gaps.sort(reverse=True)
igaps = []; i = 0   # short interpolated gaps while moving (occlusion at the feet)
while i < len(frames):
    if ball[frames[i]][2] != "det":
        j = i
        while j+1 < len(frames) and ball[frames[j+1]][2] != "det" and frames[j+1] == frames[j]+1: j += 1
        d = ((ball[frames[j]][4]-ball[frames[i]][4])**2 + (ball[frames[j]][5]-ball[frames[i]][5])**2) ** .5
        if d > 3: igaps.append((j-i+1, frames[(i+j)//2], d))
        i = j+1
    else: i += 1
igaps.sort(reverse=True)
lowc = sorted((v[3], f) for f, v in ball.items() if v[2] == "det")
G = [g for g in gaps if 2*fps <= g[0] <= 15*fps][:2]   # lost between 2 and 15 s
picks = [("lost", g[1], f"ball lost for {g[0]/fps:.1f} s: no detection between the orange X (last seen) and the green X (next seen), {g[4]:.0f} m apart", g) for g in G]
picks += [("occluded", igaps[0][1], f"ball not detected for {igaps[0][0]/fps:.1f} s while moving {igaps[0][2]:.0f} m (occluded at the feet; position interpolated)", None),
          ("lowconf", lowc[len(lowc)//50][1], f"ball detected with confidence {lowc[len(lowc)//50][0]:.2f} (2nd percentile): ~7 px in the 1280 input", None)]
def grab(f):
    png = subprocess.check_output(["ffmpeg","-v","error","-ss",f"{(f-1)/fps:.3f}","-i",vid,"-frames:v","1","-f","image2pipe","-vcodec","png","-"]); return Image.open(io.BytesIO(png)).convert("RGB")
def ts(f): t = (f-1)/fps; return f"{int(t//60):02d}:{int(t%60):02d}"
def draw_people(im, f):
    d = ImageDraw.Draw(im)
    for l, x, y, w, h, c, cls in boxes[f]:
        col = (31,58,147) if l.startswith("A") else (235,235,225) if l.startswith("B") else (235,104,52) if l.startswith("GK") else (255,215,0)
        d.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline=col, width=3); tw = d.textlength(f"{l} {c:.2f}", font=F)
        d.rectangle([x-w/2, y-h/2-24, x-w/2+tw+8, y-h/2-2], fill=col); d.text((x-w/2+4, y-h/2-24), f"{l} {c:.2f}", fill=(255,255,255) if l.startswith(("A","GK")) else (0,0,0), font=F)
    return d
def banner(im, txt):
    d = ImageDraw.Draw(im); d.rectangle([0,0,im.width,44], fill=(0,0,0)); d.text((14,8), txt, fill=(255,255,255), font=FB)
sheets = []
for f in people:
    im = grab(f); draw_people(im, f); n = len(boxes[f])
    banner(im, f"{ts(f)}  people: {n} boxes = 4 San Ramon + 4 Grecia + 2 GK + ref, stitched identities (grecia-v7 @1280, conf 0.3, BotSort)"); im.save(f"{out}_people_{f}.jpg", quality=90); sheets.append(im)
W = sheets[0].width; sheet = Image.new("RGB", (W, sum(s.height for s in sheets)+10*(len(sheets)-1)), (20,20,20)); y = 0
for s in sheets: sheet.paste(s, (0,y)); y += s.height+10
sheet.save(f"{out}_people_good.jpg", quality=88); print("people frames", people, "->", f"{out}_people_good.jpg")
sheets = []
for kind, f, why, g in picks:
    im = grab(f); draw_people(im, f); d = ImageDraw.Draw(im)
    if g is not None:
        xa, ya = ball[g[2]][0], ball[g[2]][1]; xb, yb = ball[g[3]][0], ball[g[3]][1]
        for (x0, y0, col) in ((xa, ya, (255,170,0)), (xb, yb, (0,230,0))): d.line([x0-14,y0-14,x0+14,y0+14], fill=col, width=5); d.line([x0-14,y0+14,x0+14,y0-14], fill=col, width=5)
        for k in range(0, 100, 8): d.line([xa+(xb-xa)*k/100, ya+(yb-ya)*k/100, xa+(xb-xa)*(k+4)/100, ya+(yb-ya)*(k+4)/100], fill=(255,255,255), width=3)
        x, y, col = (xa+xb)/2, (ya+yb)/2, (255,255,255)
    else:
        x, y, src, c, _, _ = ball[f]; r = 18; col = (255,170,0) if src != "det" else (0,230,0); d.ellipse([x-r,y-r,x+r,y+r], outline=col, width=3)
    Z = 90; crop = im.crop((int(x-Z), int(y-Z), int(x+Z), int(y+Z))).resize((4*Z*2, 4*Z*2), Image.NEAREST)
    im.paste(crop, (im.width-crop.width-20, 60)); d.rectangle([im.width-crop.width-20, 60, im.width-20, 60+crop.height], outline=col, width=4)
    d.text((im.width-crop.width-16, 64), "4x zoom" + (" between last/next seen" if g is not None else " at the ball"), fill=col, font=F)
    banner(im, f"{ts(f)}  {why}"); im.save(f"{out}_ball_{kind}_{f}.jpg", quality=90); sheets.append(im)
sheet = Image.new("RGB", (W, sum(s.height for s in sheets)+10*(len(sheets)-1)), (20,20,20)); y = 0
for s in sheets: sheet.paste(s, (0,y)); y += s.height+10
sheet.save(f"{out}_ball_struggles.jpg", quality=88); print("ball frames", [(k,f) for k,f,_,_ in picks], "->", f"{out}_ball_struggles.jpg")
