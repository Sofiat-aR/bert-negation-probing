#THIS SEGMENT LOADS LIBRARIES'''
import streamlit as st
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel


st.set_page_config(layout="wide")
st.title("🧠 Interpreting BERT Layers for Negation")


st.markdown("""
## 🔍 What This App Does

This tool helps you **probe how BERT represents negation** across its transformer layers using two strategies:

1. **Cosine Similarity** – How similar is the sentence with/without negation at each BERT layer?
2. **Classifier Accuracy** – How well can a classifier distinguish negated vs. non-negated sentences using layer-wise CLS tokens?

This helps understand **where** in BERT’s architecture negation is processed, and how we might use that information for **interpretability** or **control** in NLP applications.
---
""")


#THIS SEGMENT LOADS THE HIDDEN LAYERS REP.'''
cls_good = joblib.load("cls_good.pkl")
cls_bad = joblib.load("cls_bad.pkl")



#THIS SEGMENT CAL. COSINE SIMILARITY'''
similarities = [
    cosine_similarity(g, b, dim=-1).mean().item()
    for g, b in zip(cls_good, cls_bad)
]

#THIS SEGMENT FOR SIDE-BAR'''
st.sidebar.header("🔧 Options")
selected_layer = st.sidebar.slider(
    "Select a BERT Layer to Inspect",
    min_value=0,
    max_value=len(similarities)-1,
    value=6,
    help="Choose a BERT layer to analyze CLS token similarity and run custom comparisons."
)
show_table = st.sidebar.checkbox("Show Similarity Table", False)


'''THIS SEGMENT PLOTS GRAPH FOR COSINE FUNC.'''
st.subheader("📊 Cosine Similarity: Negated vs. Non-Negated Sentences (HIGH score-100-99% indicate the layer FAILS to encode Negation)")
fig1, ax1 = plt.subplots()
sns.lineplot(x=list(range(len(similarities))), y=similarities, marker="o", ax=ax1)
ax1.set_xlabel("BERT Layer")
ax1.set_ylabel("Avg Cosine Similarity (CLS Token)")
ax1.set_title("Layer-wise Cosine Similarity")
st.pyplot(fig1)

if show_table:
    df_sim = pd.DataFrame({
        'Layer': list(range(len(similarities))),
        'Similarity': similarities
    })
    st.dataframe(df_sim)


#THIS SEGMENT PLOTS GRAPH FOR CLASSIFIER PRED.'''

st.subheader("📈 Classifier Accuracy Across Layers (HIGH score indicate the layer ENCODES Negation)")
results_df = pd.read_csv("classifier_accuracy.csv")
fig2, ax2 = plt.subplots()
sns.lineplot(data=results_df, x='Layer', y='Accuracy', hue='Classifier', marker="o", ax=ax2)
ax2.set_xlabel("BERT Layer")
ax2.set_ylabel("Accuracy")
ax2.set_title("Layer-wise Negation Classification Accuracy")
st.pyplot(fig2)

#THIS SEGMENT INSPECTS CLS VECTOR.'''
st.subheader(f"🧬 CLS Token Vector Shapes at Layer {selected_layer}")
st.write(f"Non-negated shape: `{cls_good[selected_layer].shape}`")
st.write(f"Negated shape: `{cls_bad[selected_layer].shape}`")


#THIS SEGMENT ALLOW USERS COMPARE SENTENCES'''
st.markdown("---")
st.markdown("## ✍️ Try Your Own Sentences")
st.markdown("""
Enter a **non-negated** and a **negated** sentence. We will compute:
- Cosine similarity of their representations at your selected BERT layer
- Classifier prediction on whether this layer captures the negation
""")

sent1 = st.text_input("🟢 Non-negated Sentence", "The boy plays football.")
sent2 = st.text_input("🔴 Negated Sentence", "The boy does not play football.")

if st.button("Compare Representations"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    with torch.no_grad():
        toks1 = tokenizer(sent1, return_tensors="pt")
        toks2 = tokenizer(sent2, return_tensors="pt")
        h1 = model(**toks1).hidden_states[selected_layer][0, 0, :]
        h2 = model(**toks2).hidden_states[selected_layer][0, 0, :]
        user_sim = cosine_similarity(h1, h2, dim=0).item()

#THIS SEGMENT FEED USERS ON COMPARISON RESULT BASED ON COSINE FUNC.'''
    st.info(f"🧠 Cosine Similarity at Layer {selected_layer}: **{user_sim:.4f}**")
    if user_sim < 0.85:
        st.success("✅ This layer likely encodes negation (low similarity).")
    else:
        st.warning("⚠️ High similarity — this layer may not capture negation effectively.")

#THIS SEGMENT LOADS CLASSIFIER FOR SELECTED LAYER BY USER'''
    try:
         
        clf = joblib.load(f"classifiers/classifiers_layer_{selected_layer}.pkl")
        vec = (h2 - h1).numpy().reshape(1, -1)  # use difference vector for classification
        pred = clf.predict(vec)[0]
        label_map = {0: "Non-negated", 1: "Negated"}
        st.info(f"🧪 Classifier Prediction: **{label_map[pred]}**")

        if pred == 1:
            st.success("✅ Classifier confirms negation is encoded at this layer.")
        else:
            st.warning("⚠️ Classifier fails to detect negation — weak signal at this layer.")
    except FileNotFoundError:
        st.error(f"Classifier for layer {selected_layer} not found. Please train and save it to `classifiers_layer_{selected_layer}.pkl`.")

#THIS SEGMENT UPDATES COSINE GRAPH BASED ON REALTIME RESULT'''
    fig3, ax3 = plt.subplots()
    sns.lineplot(x=list(range(len(similarities))), y=similarities, marker="o", ax=ax3)
    ax3.axhline(user_sim, color="red", linestyle="--", label=f'Your Input (Layer {selected_layer})')
    ax3.set_xlabel("BERT Layer")
    ax3.set_ylabel("Avg Cosine Similarity (CLS Token)")
    ax3.set_title("Updated Cosine Similarity Plot")
    ax3.legend()
    st.pyplot(fig3)


st.markdown("""
---
🔗 [GitHub](https://github.com/Sofiat-aR/bert-negation-probing)  
Created by **Sofiat Adeola Rasheed** | Model: `bert-base-uncased` | Built with 🧡 and Streamlit
""")
