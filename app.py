"""
app.py - Interface Streamlit — Assistant Juridique Immobilier Tunisien
"""

import streamlit as st
from rag import ask
import os

st.set_page_config(
    page_title="Assistant Juridique Immobilier Tunisien",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #0f2d4a 0%, #1a5276 100%);
        padding: 1.6rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    .app-header-icon { font-size: 2.2rem; }
    .app-header-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .app-header-sub {
        color: #aed6f1;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f2d4a;
    }
    [data-testid="stSidebar"] * {
        color: #d6eaf8 !important;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-section h4 {
        color: #aed6f1 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.4rem;
    }
    .sidebar-section p, .sidebar-section li {
        font-size: 0.85rem;
        margin: 0.2rem 0;
        line-height: 1.5;
    }
    .source-tag {
        display: inline-block;
        background: rgba(41,128,185,0.3);
        border: 1px solid rgba(41,128,185,0.5);
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.78rem;
        color: #aed6f1 !important;
        margin: 0.2rem 0.1rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e6ed;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fef9f0;
        border-left: 3px solid #e67e22;
        border-radius: 0 6px 6px 0;
        padding: 0.6rem 0.9rem;
        font-size: 0.8rem;
        color: #7d6608;
        margin-top: 0.6rem;
    }

    /* ── Welcome screen ── */
    .welcome-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem 2rem 2rem;
        text-align: center;
    }
    .welcome-logo {
        font-size: 7rem;
        line-height: 1;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 4px 16px rgba(26,82,118,0.18));
    }
    .welcome-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f2d4a;
        margin: 0 0 0.7rem 0;
        letter-spacing: -0.02em;
    }
    .welcome-subtitle {
        font-size: 0.95rem;
        color: #5d7a8a;
        max-width: 520px;
        line-height: 1.7;
        margin: 0 auto 2rem auto;
    }
    .welcome-divider {
        width: 48px;
        height: 3px;
        background: linear-gradient(90deg, #1a5276, #2980b9);
        border-radius: 2px;
        margin: 0 auto 2rem auto;
    }
    .welcome-domains {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.5rem;
        max-width: 580px;
        margin: 0 auto;
    }
    .domain-chip {
        background: #ffffff;
        border: 1px solid #d0dae6;
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        font-size: 0.8rem;
        color: #1a3a52;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">⚖️</div>
    <div>
        <p class="app-header-title">Assistant Juridique Immobilier Tunisien</p>
        <p class="app-header-sub">Droit réel · Promotion immobilière · Fiscalité · Baux</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Vérification base ────────────────────────────────────────
if not os.path.exists("db"):
    st.error("**Base de données introuvable.** Exécutez d'abord : `python build_db.py`")
    st.stop()


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0; text-align:center;">
        <span style="font-size:1.8rem;">⚖️</span>
        <p style="color:#aed6f1 !important; font-weight:700; font-size:0.95rem; margin:0.3rem 0 0 0;">
            Assistant Juridique
        </p>
        <p style="color:#7fb3d3 !important; font-size:0.75rem; margin:0;">
            Tunisie · Immobilier
        </p>
    </div>
    <hr style="border-color:rgba(255,255,255,0.1); margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h4>📚 Sources juridiques</h4>
        <span class="source-tag">Code des Droits Réels 1965</span>
        <span class="source-tag">Loi Promotion Immobilière 1990</span>
        <span class="source-tag">Loi de Finances 2025</span>
    </div>

    
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <p style="font-size:0.72rem; color:#7fb3d3 !important; margin-top:1.5rem; line-height:1.5;">
        ⚠️ Les réponses sont fournies à titre informatif et ne remplacent pas le conseil d'un avocat ou d'un notaire.
    </p>
    """, unsafe_allow_html=True)


# ── Session ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Historique ───────────────────────────────────────────────
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "⚖️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ── Page d'accueil ───────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-logo">⚖️</div>
        <h2 class="welcome-title">Bienvenue sur votre assistant<br>juridique immobilier</h2>
        <div class="welcome-divider"></div>
        <p class="welcome-subtitle">
            Posez vos questions relatives au droit immobilier tunisien.
            Les réponses sont fondées sur les textes législatifs en vigueur
            et référencent les articles applicables.
        </p>
        <div class="welcome-domains">
            <span class="domain-chip">🏠 Droits réels</span>
            <span class="domain-chip">📝 Promesse de vente</span>
            <span class="domain-chip">💰 Fiscalité &amp; TVA</span>
            <span class="domain-chip">🏗️ Promotion immobilière</span>
            <span class="domain-chip">🔑 Baux &amp; expulsion</span>
            <span class="domain-chip">🔒 Hypothèque</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Saisie principale ────────────────────────────────────────
if question := st.chat_input("Posez votre question juridique…"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Recherche en cours…"):
            try:
                response = ask(question)
                st.markdown(response)
                st.markdown(
                    '<div class="disclaimer">⚠️ Information générale extraite des textes législatifs tunisiens. '
                    'Consultez un avocat ou un notaire pour votre situation personnelle.</div>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.info("Vérifiez qu'Ollama est lancé (`ollama serve`) et que les modèles sont installés.")