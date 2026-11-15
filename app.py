import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
import cv2

st.set_page_config(page_title="ALand | Landing-Site Baselines", layout="wide")
st.title("ALand — Landing-Site Baseline Explorer")
st.caption("Research prototype · heuristic candidate regions are NOT verified safe landing locations")

def candidates(img, tile=72, min_score=0.36):
    rgb=np.asarray(img.convert("RGB"))
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    edges=cv2.Canny(gray,65,140)
    results=[]
    for y in range(0,h-tile+1,max(16,tile//2)):
        for x in range(0,w-tile+1,max(16,tile//2)):
            p=gray[y:y+tile,x:x+tile]
            e=edges[y:y+tile,x:x+tile]
            s=hsv[y:y+tile,x:x+tile,1]
            # Appearance-only suitability proxy: texture, edges, saturation.
            texture=float(np.std(p))/75
            edge=float(np.mean(e>0))*5
            saturation=float(np.mean(s))/255
            score=float(np.clip(1-0.45*texture-0.45*edge-0.1*saturation,0,1))
            if score>=min_score:
                results.append(dict(x=x,y=y,w=tile,h=tile,score=round(score,3)))
    results.sort(key=lambda q:q["score"],reverse=True)
    kept=[]
    for r in results:
        if all((r["x"]-a["x"])**2+(r["y"]-a["y"])**2>(tile*.8)**2 for a in kept):
            kept.append(r)
        if len(kept)>=24:break
    return kept

def iou(a,b):
    x1=max(a["x"],b["x"]);y1=max(a["y"],b["y"])
    x2=min(a["x"]+a["w"],b["x"]+b["w"]);y2=min(a["y"]+a["h"],b["y"]+b["h"])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/(a["w"]*a["h"]+b["w"]*b["h"]-inter+1e-9)

def associate(base, previous, threshold=.18):
    # Pixel alignment is only a placeholder: true cross-altitude matching requires registration.
    out=[]
    for r in base:
        matches=[p for p in previous if iou(r,p)>.25]
        z=r.copy()
        z["score"]=round(float(np.mean([r["score"]]+[p["score"] for p in matches])),3)
        z["observations"]=1+len(matches)
        if z["score"]>=threshold:out.append(z)
    return out

def run_baselines(img,history,tile,cutoff):
    b1=candidates(img,tile,cutoff)
    # B2 retains current candidates with prior-image evidence when images are registered.
    prior=[r for prior_img in history for r in candidates(prior_img,tile,cutoff)]
    b2=associate(b1,prior,cutoff) if prior else [dict(r,observations=1) for r in b1]
    # B3 is a deterministic rule policy on B2 evidence, not a learned adaptive controller.
    b3=[]
    for r in b2:
        if r["score"]>=max(cutoff,.57) and (r["observations"]>=2 or not history):
            b3.append(dict(r,decision="candidate"))
    return {"B1":b1,"B2":b2,"B3":b3}

def render(img,boxes,color):
    output=img.copy().convert("RGB")
    draw=ImageDraw.Draw(output)
    for n,r in enumerate(boxes,1):
        draw.rectangle((r["x"],r["y"],r["x"]+r["w"],r["y"]+r["h"]),outline=color,width=3)
        draw.text((r["x"]+3,r["y"]+3),str(n),fill=color,stroke_width=1,stroke_fill="black")
    return output

def compare(result):
    groups=[]
    for name,items in result.items():
        for item in items:
            hit=next((g for g in groups if any(iou(item,q)>.35 for q in g["items"])),None)
            if hit is None:
                groups.append({"items":[item],"methods":{name}})
            else:
                hit["items"].append(item);hit["methods"].add(name)
    return groups

uploads=st.file_uploader("Upload aerial images (order: highest altitude to lowest)",type=["png","jpg","jpeg"],accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more images to begin. Multi-altitude images must be spatially registered for meaningful B2/B3 comparisons.")
    st.stop()
frames=[]
for f in uploads:
    try:
        im=Image.open(io.BytesIO(f.getvalue())).convert("RGB")
        im.thumbnail((1100,1100))
        frames.append(im)
    except Exception as exc:
        st.warning(f"Unable to read {f.name}: {exc}")
if not frames:st.stop()
idx=st.select_slider("Observation",options=list(range(len(frames))),format_func=lambda i:f"{i+1}: {uploads[i].name}")
alt=st.number_input("Altitude of selected observation (ft AGL; metadata only)",min_value=0,max_value=10000,value=250,step=25)
tile=st.slider("Candidate window size (pixels)",32,180,72,8)
cutoff=st.slider("Appearance score threshold (not a safety probability)",0.10,0.95,0.55,0.05)
method=st.radio("Select analysis",["B1","B2","B3","Compare"],horizontal=True)
st.image(frames[idx],caption=f"Original observation {idx+1} | reported altitude {alt} ft",use_container_width=True)
if st.button("▶ Play analysis",type="primary"):
    results=run_baselines(frames[idx],frames[:idx],tile,cutoff)
    st.session_state["results"]=results
    st.session_state["signature"]=(idx,tile,cutoff,tuple(hashlib.sha256(f.getvalue()).hexdigest() for f in uploads))
sig=(idx,tile,cutoff,tuple(hashlib.sha256(f.getvalue()).hexdigest() for f in uploads))
if st.session_state.get("signature")!=sig:
    st.info("Press Play analysis to compute results for this image and these settings.")
    st.stop()
results=st.session_state["results"]
if method!="Compare":
    color={"B1":"#22c55e","B2":"#3b82f6","B3":"#f97316"}[method]
    st.image(render(frames[idx],results[method],color),caption=f"{method}: {len(results[method])} appearance-based candidates",use_container_width=True)
    st.dataframe(pd.DataFrame(results[method]),use_container_width=True)
else:
    groups=compare(results)
    palette={"B1":"#22c55e","B2":"#3b82f6","B3":"#f97316","overlap":"#a855f7"}
    overlay=frames[idx].copy()
    for g in groups:
        k="overlap" if len(g["methods"])>1 else next(iter(g["methods"]))
        overlay=render(overlay,[g["items"][0]],palette[k])
    st.image(overlay,caption="Green B1 only · Blue B2 only · Orange B3 only · Purple overlap",use_container_width=True)
    rows=[{"methods":" + ".join(sorted(g["methods"])),"overlap_count":len(g["methods"]),"x":g["items"][0]["x"],"y":g["items"][0]["y"],"score":g["items"][0]["score"]} for g in groups]
    st.dataframe(pd.DataFrame(rows),use_container_width=True)
    if rows:st.download_button("Download comparison CSV",pd.DataFrame(rows).to_csv(index=False),"aland_comparison.csv","text/csv")
st.warning("Research limitation: color/texture and edge heuristics cannot determine physical flatness, clearance, load capacity, thin-wire presence, or actual landing safety. Cross-altitude correspondence is not geometrically registered; B2/B3 outputs are illustrative until image registration and uncertainty modeling are implemented.")
