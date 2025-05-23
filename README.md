# CareDesk AI 🩺

An AI-powered Cerner helpdesk assistant built with:
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Vector search using FAISS
- ✅ Local inference with Hugging Face's DistilGPT2
- ✅ Streamlit for UI
- ✅ Delta-style CSV memory for past tickets

### Features
- Searches for similar helpdesk issues from a knowledge base
- Uses AI to suggest helpful responses
- Logs and reuses responses for repeated tickets
- If unsure, alerts: “Connecting you to a human agent”

### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
