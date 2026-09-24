# Deployment Guide

## 1. Push to GitHub
```bash
cd meeting-summarizer
git init
git add .
git commit -m "Initial commit: AI-Based Smart Meeting Summarizer"
git branch -M main
git remote add origin https://github.com/<your-username>/meeting-summarizer.git
git push -u origin main
```
(Create the empty repo first on github.com -> New repository. Do not tick "Add README".)

## 2. Deploy (GitHub only stores code; it cannot run the app)

### Option A - Hugging Face Spaces (recommended, free 16 GB RAM)
1. huggingface.co -> New Space -> SDK: **Docker** -> hardware: CPU basic (free).
2. Push this project to the Space repo. The Space's README.md must start with:
   ```
   ---
   title: Meeting Summarizer
   sdk: docker
   app_port: 7860
   ---
   ```
3. The Dockerfile builds automatically (first build takes ~10-15 min).

### Option B - Streamlit Community Cloud (simplest, but low RAM)
share.streamlit.io -> New app -> pick your GitHub repo -> main file `app.py`.
The free tier has very limited memory; torch + DistilBART + Whisper may crash it.
If so, use Whisper "tiny" and "extractive" summary mode in the sidebar.

### Option C - Any Docker host (Render, Railway, VPS)
```bash
docker build -t meeting-summarizer .
docker run -p 7860:7860 meeting-summarizer
```
Needs at least 4 GB RAM.
