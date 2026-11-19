import os, datetime
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv("../ejemplo-1-deteccion-basica/.env")
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")); ws = rf.workspace("football-ucr")
for p in ws.project_list:
    if p.get("images", 0) == 0: continue
    print("%-16s images %5s versions %s updated %s" % (str(p.get("id")).split("/")[-1], p.get("images"), p.get("versions"), datetime.datetime.fromtimestamp(p.get("updated", 0)).strftime("%Y-%m-%d")))
p = ws.project("grecia")
for v in p.versions():
    print("  grecia v%s | images %s | created %s | %s" % (v.version.split("/")[-1], v.images, datetime.datetime.fromtimestamp(float(getattr(v, "created", 0))).strftime("%Y-%m-%d"), getattr(v, "name", "")))
