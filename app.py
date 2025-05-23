import streamlit as st
import os
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load knowledge base articles from folder
KB_FOLDER = "kb_articles"
kb_files = sorted([f for f in os.listdir(KB_FOLDER) if f.endswith(".txt")])
kb_texts = [open(os.path.join(KB_FOLDER, f), 'r').read() for f in kb_files]
kb_titles = [text.splitlines()[0] if text.strip() else "Untitled" for text in kb_texts]
st.sidebar.success(f"✅ Loaded {len(kb_texts)} KB articles.")

# Embed articles and build FAISS index (do this once)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # MiniLM for fast, quality embeddings:contentReference[oaicite:16]{index=16}
kb_embeddings = embedder.encode(kb_texts)           # encode all articles to vectors:contentReference[oaicite:17]{index=17}
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(kb_embeddings, dtype='float32'))  # add vectors to FAISS index:contentReference[oaicite:18]{index=18}

# Load the text generation model (DistilGPT2):contentReference[oaicite:19]{index=19}
generator = pipeline('text-generation', model='distilgpt2')

# Define a helper function to search the KB and return top article + score
def search_kb(query, top_k=1):
    """Return top_k (article_text, score) pairs for the given query."""
    q_embedding = embedder.encode([query])
    D, I = index.search(np.array(q_embedding, dtype='float32'), k=top_k)
    results = []
    for rank in range(top_k):
        article_text = kb_texts[I[0][rank]]
        # convert L2 distance to a [0,1] similarity score (approximate cosine):contentReference[oaicite:20]{index=20}
        score = 1 - float(D[0][rank])
        score = round(score, 2)
        results.append((article_text, score))
    return results

# Streamlit UI setup
st.title("🤖 Cerner Helpdesk AI Assistant")
st.write("Submit an IT helpdesk issue and get an AI-suggested solution. "
         "The assistant will search a knowledge base of common Cerner issues and provide a response. "
         "If it’s not confident, it will recommend escalating to a human agent.")

# Ticket input form
with st.form("ticket_form"):
    ticket_text = st.text_area("Describe your issue:", height=100)
    submit_btn = st.form_submit_button("Submit")

if submit_btn:
    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")
    else:
        # Normalize input for comparison
        norm_text = ticket_text.strip().lower()
        log_file = "caredesk_log.csv"
        final_response = ""  # will hold the answer or fallback
        matched_article = None
        similarity = 0.0
        status = ""  # "resolved" or "escalated"
        
        # Check for duplicate ticket in the log:contentReference[oaicite:21]{index=21}
        previous_entry = None
        if os.path.exists(log_file):
            log_df = pd.read_csv(log_file)
            if 'ticket_text' in log_df.columns:
                # Normalize the ticket_text column for comparison
                log_df['ticket_text_norm'] = log_df['ticket_text'].str.strip().str.lower()
                dup_matches = log_df[log_df['ticket_text_norm'] == norm_text]
                if not dup_matches.empty:
                    previous_entry = dup_matches.iloc[-1]  # use the latest matching entry
        
        if previous_entry is not None:
            # Duplicate found: reuse the logged response
            st.info("ℹ️ This issue was seen before. Fetching the previous solution...")
            final_response = previous_entry.get('final_response', previous_entry.get('ai_response', ""))
            similarity = float(previous_entry.get('similarity_score', 0.0))
            # Determine if a KB article was used in that previous answer
            matched_title = previous_entry.get('matched_article', "None")
            if matched_title and matched_title != "None":
                # find the article content by title
                try:
                    idx = kb_titles.index(matched_title)
                    matched_article = kb_texts[idx]
                except ValueError:
                    matched_article = None
            # Retrieve status from log or infer it
            status = previous_entry.get('status', "")
            if not status:
                status = "resolved" if matched_title != "None" or len(final_response) > 0 else "escalated"
        else:
            # New ticket: perform knowledge base search
            results = search_kb(ticket_text, top_k=1)
            top_article, top_score = results[0]
            similarity = top_score
            if similarity >= 0.75:
                matched_article = top_article
                # Create prompt with the relevant article as context:contentReference[oaicite:22]{index=22}
                prompt = (f"Helpdesk Ticket: {ticket_text}\n\n"
                          f"Relevant Knowledge Article:\n{matched_article}\n\n"
                          f"Based on this information, generate a short and helpful support response for the IT helpdesk agent to use.")
            else:
                matched_article = None
                # Prompt without context (AI-only)
                prompt = (f"Helpdesk Ticket: {ticket_text}\n\n"
                          f"No relevant knowledge base article was found for this issue.\n"
                          f"Provide a helpful response to assist the user with this problem.")
            
            # Generate AI response using DistilGPT2
            result = generator(prompt, max_new_tokens=150, do_sample=True, return_full_text=False)
            ai_response = result[0]['generated_text'].strip()
            
            # Check for AI hallucination or insufficient answer
            if matched_article is None:
                # Only apply the safeguard for AI-only responses
                if len(ai_response) < 40 or ai_response.lower().startswith("i'm an ai") or ai_response.lower().startswith("as a language model"):
                    # Treat as no confident answer
                    final_response = "⚠️ No confident answer found. Connecting you to a human agent."
                    status = "escalated"
                else:
                    final_response = ai_response
                    status = "resolved"
            else:
                # With a KB context, assume the response is grounded and use it
                final_response = ai_response
                status = "resolved"
            
            # Log the new ticket to CSV:contentReference[oaicite:23]{index=23}:contentReference[oaicite:24]{index=24}
            ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            matched_title = kb_titles[kb_texts.index(matched_article)] if matched_article else "None"
            log_entry = {
                "ticket_id": ticket_id,
                "timestamp": timestamp,
                "ticket_text": ticket_text,
                "matched_article": matched_title,
                "similarity_score": round(similarity, 2),
                "final_response": final_response,
                "status": status
            }
            # Append or create the CSV log
            try:
                if os.path.exists(log_file):
                    existing_df = pd.read_csv(log_file)
                    # Normalize existing texts for duplicate safety (should not really be needed here since we handled duplicates above)
                    if 'ticket_text' in existing_df.columns:
                        existing_df['ticket_text'] = existing_df['ticket_text'].astype(str)
                        existing_df['ticket_text'] = existing_df['ticket_text'].str.strip().str.lower()
                    # Only add if not a duplicate (again)
                    if norm_text not in list(existing_df['ticket_text']):
                        new_df = pd.DataFrame([log_entry])
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.to_csv(log_file, index=False)
                else:
                    pd.DataFrame([log_entry]).to_csv(log_file, index=False)
            except Exception as e:
                st.error(f"Logging failed: {e}")
        
        # --- Display the results in the UI ---
        if matched_article:
            # Show the matched KB article title and content
            title = matched_article.splitlines()[0] if matched_article else "Knowledge Article"
            with st.expander(f"📖 Matched KB Article: {title}"):
                st.code(matched_article, language="")
        else:
            st.write("*No relevant knowledge base article was found for this ticket.*")
        
        # Show the AI response or fallback message
        if status == "resolved":
            st.markdown(f"**AI Suggested Response:** {final_response}")
        else:  # escalated
            st.markdown(f"**AI Suggested Response:** {final_response}")
            st.warning("This issue may need human attention (AI could not find a confident answer).")
