"""xG provisional (goles esperados) para los tiros de un tiempo, a partir de los CSV/JSON del sitio.

No se ajusta a los datos propios (un tiempo tiene ~15 tiros y 2-3 goles con posición: demasiado poco).
Se aplica un modelo geométrico transparente, para recalibrar cuando haya varios partidos:

  xG = p(d) * theta / theta_c(d)

  p(d) = sigmoide(a + b*d), distancia d (m) desde el punto de golpeo al centro de la portería.
      a y b salen por análisis discriminante lineal de las medias publicadas para futsal por
      Doewes, Elumalai y Azmi (2023, Rev. Iberoam. Psicol. Ejerc. Deporte 18(2):235-237): en un
      partido de la liga indonesia, 65 tiros, 5 goles; distancia media de los goles 5.27 ± 0.69 m,
      de los fallos 11.36 ± 4.44 m. Con la desviación agrupada (4.30 m):
      b = (5.27 - 11.36) / 4.30^2 = -0.329 por metro, a = ln(5/60) - b*(5.27+11.36)/2 = 0.253.
      (La pendiente coincide con la de ajustes logísticos en fútbol, ~ -0.31 por metro.)
  theta = ángulo que abarca la boca de la portería (3 m) visto desde el punto de golpeo;
  theta_c(d) = ese ángulo desde el centro a la misma distancia = 2*atan(1.5/d).
      El cociente vale 1 de frente y baja hacia 0 en ángulos cerrados.

Uso: python xg_provisional.py docs/datos_1T [--a 0.253 --b -0.329]
Escribe la columna `xg` en events.csv, teams.csv y players.csv y los campos `xg` en sitio.json.
"""
import sys, csv, json, math, collections
d = sys.argv[1]
A = float(sys.argv[sys.argv.index("--a")+1]) if "--a" in sys.argv else 0.253
B = float(sys.argv[sys.argv.index("--b")+1]) if "--b" in sys.argv else -0.329
GW = 3.0
pre = [f for f in __import__("os").listdir(d) if f.endswith("_sitio.json")][0].replace("_sitio.json", "")
site = json.load(open(f"{d}/{pre}_sitio.json"))
L, W = site["pitch"]["L"], site["pitch"]["W"]
attack = site["attack"]  # {"A": "left"|"right", "B": ...}
name2t = {site["names"][t].replace("ó", "o"): t for t in "AB"}; name2t.update({site["names"][t]: t for t in "AB"})

def xg(team, x, y):
    gx = 0.0 if attack[team] == "left" else L
    dx, dy = abs(x - gx), y - W/2
    dist = math.hypot(dx, dy)
    th = math.atan2(GW*dx, dx*dx + dy*dy - (GW/2)**2); th = th if th > 0 else th + math.pi
    thc = 2*math.atan((GW/2)/max(dist, 1e-6))
    p = 1/(1 + math.exp(-(A + B*dist)))
    return round(p*min(th/thc, 1.0), 3), round(dist, 1), round(math.degrees(th), 1)

# events
ev = list(csv.DictReader(open(f"{d}/{pre}_events.csv")))
tot = collections.Counter(); per = collections.Counter()
print("time   team       player type                 x     y   dist  ang  xG")
for e in ev:
    e["xg"] = ""
    if e["x"] and (e["type"].startswith("shot") or e["type"].startswith("goal")):
        t = name2t[e["team"]]; v, dist, ang = xg(t, float(e["x"]), float(e["y"])); e["xg"] = f"{v:.3f}"
        tot[t] += v; per[(t, e["player"])] += v
        print(f'{e["time"]}  {e["team"]:<10} {e["player"]:<6} {e["type"][:20]:<20} {float(e["x"]):5.1f} {float(e["y"]):5.1f} {dist:5.1f} {ang:4.0f}  {v:.3f}')
cols = [c for c in ev[0].keys() if c != "xg"] + ["xg"]
with open(f"{d}/{pre}_events.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, lineterminator="\n"); w.writeheader(); w.writerows(ev)

# teams
tm = list(csv.DictReader(open(f"{d}/{pre}_teams.csv")))
for r in tm: r["xg"] = f"{tot[name2t[r['team']]]:.2f}"
with open(f"{d}/{pre}_teams.csv", "w", newline="") as f:
    cols = [c for c in tm[0].keys() if c != "xg"] + ["xg"]; w = csv.DictWriter(f, cols, lineterminator="\n"); w.writeheader(); w.writerows(tm)

# players CSV: team column is the team name or GK; player is the identity
pl = list(csv.DictReader(open(f"{d}/{pre}_players.csv")))
for r in pl: r["xg"] = f"{per[(name2t[r['team']], r['player'])]:.2f}" if r["team"] in name2t else "0.00"
with open(f"{d}/{pre}_players.csv", "w", newline="") as f:
    cols = [c for c in pl[0].keys() if c != "xg"] + ["xg"]; w = csv.DictWriter(f, cols, lineterminator="\n"); w.writeheader(); w.writerows(pl)

# site json
site["events"] = ev
for t in "AB": site["teams"][t]["xg"] = round(tot[t], 2)
for p in site["players"]: p["xg"] = round(per[(p["team"], p["id"])], 2) if p["team"] in "AB" else 0.0
site["xg_model"] = dict(form="sigmoid(a+b*d)*theta/theta_c(d)", a=A, b=B, goal_width=GW,
                        source="Doewes, Elumalai & Azmi 2023, Rev. Iberoam. Psicol. Ejerc. Deporte 18(2):235-237 (LDA on published means); provisional")
json.dump(site, open(f"{d}/{pre}_sitio.json", "w"), ensure_ascii=False, separators=(",", ":"))
print({site["names"][t]: dict(xg=round(tot[t], 2), goals=site["teams"][t]["goals"], shots=site["teams"][t]["shots"]) for t in "AB"})
