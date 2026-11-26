import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv("../ejemplo-1-deteccion-basica/.env")
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")); ws = rf.workspace("football-ucr")
for p in ws.project_list:
    print("%-28s slug %-24s type %-17s images %5s versions %s updated %s" % (p.get("name","?"), str(p.get("id","?")).split("/")[-1], p.get("type","?"), p.get("images","?"), p.get("versions","?"), p.get("updated","?")))
