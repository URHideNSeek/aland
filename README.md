# ALand — Landing-Site Baseline Explorer

Streamlit research prototype for comparing three landing-site candidate assessment methods.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methods
- **B1**: single-image texture/edge-based candidate proposals.
- **B2**: combines B1 proposals with earlier image observations using pixel overlap. **Requires spatial registration to become scientifically meaningful across altitudes.**
- **B3**: deterministic rule-based filtering of B2 candidates. Not a flight controller.
- **Compare**: overlays proposals and matches by intersection-over-union, showing unique and overlapping candidate regions.

These are initial transparent *surrogate baselines*, not faithful implementations of peer-reviewed algorithms or certified landing-site detectors.

## Roadmap
1. Add image altitude metadata, camera intrinsics and homography/feature-based registration.
2. Introduce segmentation, geometry and ground-truth annotation tools.
3. Replace B2 with calibrated probabilistic mapping and candidate tracking.
4. Extend B3 with explicit uncertainty/hold/inspect/reject decisions.
5. Implement reproducible metrics: candidate recall, false-safe acceptance, confidence calibration, runtime.
6. Add persistent datasets and experiment exports.

**Safety:** Never use the app's candidate regions to command real aircraft or authorize landings. Images may contain invisible hazards, wires, people, or weak surfaces.
