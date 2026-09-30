import os, requests, streamlit as st
API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")
st.title("📚 DocuMind — RAG-поиск по документам")

q = st.text_input("Ваш вопрос:")
top = st.slider("Сколько результатов", 3, 20, 5)

if st.button("Искать") and q:
    with st.spinner("Ищу..."):
        r = requests.post(f"{API}/search", json={"q": q, "top_k": 20, "rerank": top})
    if r.ok:
        data = r.json()
        if data["answer"]:
            st.subheader("💡 Ответ")
            st.write(data["answer"])
        st.subheader(f"📄 Источники ({len(data['hits'])})")
        for h in data["hits"]:
            with st.expander(f"{h['file']} · chunk {h['chunk_idx']} · score {h['score']:.3f}"):
                st.write(h["text"])
    else:
        st.error(r.text)
