# Accessing the Demo & Notebook (remote server)

Everything runs on the server **`poincare.mathstat.uottawa.ca`** (user `boby`).
Because the services bind to the server's `localhost`, you view them from your own
machine via an **SSH tunnel**: start the service on the server, forward the port
locally, then open it in your local browser.

> SSH target: `boby@poincare.mathstat.uottawa.ca`

---

## 1. Streamlit app (segmentation → survival → Grad-CAM)

**Start on the server** (if not already running):
```bash
cd /work/boby/projects/MedAI_Proj
.venv/bin/streamlit run app.py --server.headless true --server.port 8501
```

**Tunnel from your local machine:**
```bash
ssh -L 8501:localhost:8501 boby@poincare.mathstat.uottawa.ca
```

**Open locally:** http://localhost:8501

---

## 2. EDA notebook (`eda.ipynb`)

**Start on the server:**
```bash
cd /work/boby/projects/MedAI_Proj
.venv/bin/jupyter notebook eda.ipynb --no-browser --port 8888
```
Copy the printed `http://localhost:8888/?token=…` URL.

**Tunnel from your local machine:**
```bash
ssh -L 8888:localhost:8888 boby@poincare.mathstat.uottawa.ca
```

**Open locally:** paste the tokened URL into your browser.

### Static alternative (no server, no tunnel)
The notebook is also exported as a self-contained HTML page:
```bash
# on your LOCAL machine
scp boby@poincare.mathstat.uottawa.ca:/work/boby/projects/MedAI_Proj/eda.html .
```
Then open `eda.html` in any browser.

---

## Quick reference

| Service | Start on server | Tunnel (run locally) | Open |
|---|---|---|---|
| Streamlit app | `streamlit run app.py … --server.port 8501` | `ssh -L 8501:localhost:8501 boby@poincare.mathstat.uottawa.ca` | http://localhost:8501 |
| Jupyter notebook | `jupyter notebook eda.ipynb --no-browser --port 8888` | `ssh -L 8888:localhost:8888 boby@poincare.mathstat.uottawa.ca` | tokened URL Jupyter prints |
| EDA (static) | *(already exported)* | — | `scp` `eda.html`, open locally |

## Tips

- **Port already in use locally?** Map to a different local port, e.g.
  `ssh -L 8600:localhost:8501 boby@poincare.mathstat.uottawa.ca`, then open http://localhost:8600.
- **Background tunnel** (returns your prompt): add `-N -f`, e.g.
  `ssh -N -f -L 8501:localhost:8501 boby@poincare.mathstat.uottawa.ca`. Close later with
  `pkill -f "8501:localhost:8501"`.
- **VS Code Remote-SSH** auto-forwards ports — add `8501`/`8888` in the Ports panel
  instead of tunneling manually.
- **Check what's running on the server:**
  `pgrep -af "streamlit|jupyter"`
