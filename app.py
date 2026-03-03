"""
app.py — Dashboard Immobilier Tunisien
Déploiement : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import re
import os

# ── Config page ──
st.set_page_config(
    page_title="Immobilier TN — Prédiction de Prix",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0F1117; }
.block-container { padding: 2rem 3rem; }

/* KPI cards */
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2D3142;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #F0F4FF;
    line-height: 1;
}
.kpi-label {
    font-size: 0.78rem;
    color: #6B7599;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}
.kpi-delta { font-size: 0.82rem; margin-top: 0.3rem; }

/* Prediction result */
.pred-box {
    background: linear-gradient(135deg, #1A1D27 0%, #141820 100%);
    border: 1px solid #3D4F8C;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-price {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #7B9BFF;
    letter-spacing: -0.02em;
}
.pred-pm2 {
    font-size: 1rem;
    color: #6B7599;
    margin-top: 0.3rem;
}

/* Signal badges */
.signal-green  { color: #4ADE80; font-weight: 600; }
.signal-yellow { color: #FACC15; font-weight: 600; }
.signal-red    { color: #F87171; font-weight: 600; }
.signal-gray   { color: #9CA3AF; font-weight: 600; }

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #C8D0F0;
    border-left: 3px solid #7B9BFF;
    padding-left: 0.8rem;
    margin-bottom: 1rem;
}

/* Table styling */
.dataframe { font-size: 0.82rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12141E;
    border-right: 1px solid #1E2235;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  CHARGEMENT MODÈLE & DONNÉES
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists('model_immo.pkl'):
        st.error("❌ Fichier `model_immo.pkl` introuvable. Lance d'abord le Colab pour générer le modèle.")
        st.stop()
    bundle = joblib.load('model_immo.pkl')
    return bundle

@st.cache_data
def load_predictions():
    if not os.path.exists('predictions_immobilier_tn.csv'):
        return None
    return pd.read_csv('predictions_immobilier_tn.csv')

bundle = load_model()
ensemble      = bundle['model']
FEATURES_NUM  = bundle['features_num']
FEATURES_CAT  = bundle['features_cat']
pm2_by_type   = bundle['pm2_by_type']
global_pm2    = bundle['global_pm2']
model_metrics = bundle['metrics']

df_pred = load_predictions()


# ══════════════════════════════════════════════
#  HELPER — préparation input pour prédiction
# ══════════════════════════════════════════════
CITY_TIER = {
    'Tunis': 1, 'Ariana': 1, 'La Marsa': 1,
    'Ben Arous': 2, 'Nabeul': 2, 'Hammamet': 2, 'Sousse': 2, 'Sfax': 2,
    'Monastir': 2, 'Mahdia': 3, 'Bizerte': 3, 'Kairouan': 3,
    'Djerba': 3, 'Gabès': 3,
}

def build_input(surface, property_type, rooms, city, is_new, is_furnished, has_pool):
    tier = CITY_TIER.get(city, 4)
    bath = max(0, round(rooms / 3))
    ref  = pm2_by_type.get(property_type, global_pm2)

    buckets = pd.cut([surface],
        bins=[0, 50, 80, 120, 200, 400, 20_000],
        labels=['<50m²','50-80','80-120','120-200','200-400','>400m²'])[0]

    return pd.DataFrame([{
        'surface_m2':    surface,
        'log_surface':   np.log1p(surface),
        'rooms_imp':     rooms,
        'bath_imp':      bath,
        'ref_pm2':       ref,
        'city_tier':     tier,
        'is_furnished':  is_furnished,
        'is_new':        is_new,
        'has_pool':      has_pool,
        'property_type': property_type,
        'surface_cat':   str(buckets),
    }])

def get_predictions_from_models(inp):
    """Retourne les prédictions individuelles des 3 estimators de l'ensemble."""
    preds = []
    for est_name, est in ensemble.estimators:
        p = np.clip(est.predict(inp)[0], 0, None)
        preds.append((est_name, p))
    return preds


# ══════════════════════════════════════════════
#  SIDEBAR — navigation
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏠 Immobilier TN")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🎯 Prédiction de prix", "📊 Analyse du marché", "📋 Données & Signaux"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#6B7599;'>
    <b style='color:#C8D0F0'>Modèle</b><br>
    Ensemble (top 3)<br><br>
    <b style='color:#C8D0F0'>Performance</b><br>
    R² = {model_metrics['R2']:.3f}<br>
    MAE = {model_metrics['MAE']:,.0f} TND<br>
    RMSE = {model_metrics['RMSE']:,.0f} TND
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 — PRÉDICTION DE PRIX
# ══════════════════════════════════════════════
if page == "🎯 Prédiction de prix":
    st.markdown("## 🎯 Prédiction de prix")
    st.markdown("Renseigne les caractéristiques du bien pour obtenir une estimation.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown('<div class="section-title">Caractéristiques du bien</div>', unsafe_allow_html=True)

        property_type = st.selectbox("Type de bien", [
            'Appartement', 'Villa', 'Duplex', 'Studio', 'Penthouse',
            'Maison', 'Bureau', 'Terrain', 'Local com', 'Garage',
        ])
        surface = st.slider("Surface (m²)", min_value=20, max_value=1000, value=120, step=5)
        rooms   = st.slider("Nombre de pièces", min_value=1, max_value=12, value=4)
        city    = st.selectbox("Ville", [
            'Tunis', 'Ariana', 'La Marsa', 'Ben Arous', 'Nabeul',
            'Hammamet', 'Sousse', 'Sfax', 'Monastir', 'Mahdia',
            'Bizerte', 'Kairouan', 'Djerba', 'Gabès', 'Autre',
        ])

        st.markdown("**Options**")
        c1, c2, c3 = st.columns(3)
        is_new       = int(c1.checkbox("🆕 Neuf"))
        is_furnished = int(c2.checkbox("🛋️ Meublé"))
        has_pool     = int(c3.checkbox("🏊 Piscine"))

    with col2:
        st.markdown('<div class="section-title">Estimation</div>', unsafe_allow_html=True)

        inp     = build_input(surface, property_type, rooms, city, is_new, is_furnished, has_pool)
        price_p = np.clip(ensemble.predict(inp)[0], 0, None)
        pm2_p   = price_p / surface

        # Intervalle depuis les estimators individuels
        ind_preds = get_predictions_from_models(inp)
        p_low  = min(p for _, p in ind_preds)
        p_high = max(p for _, p in ind_preds)

        st.markdown(f"""
        <div class="pred-box">
            <div style='font-size:0.8rem;color:#6B7599;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>
                Prix estimé
            </div>
            <div class="pred-price">{price_p:,.0f} TND</div>
            <div class="pred-pm2">{pm2_p:,.0f} TND / m²</div>
            <div style='margin-top:1.2rem; font-size:0.82rem; color:#4B5380;'>
                Fourchette : {p_low:,.0f} — {p_high:,.0f} TND
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparaison avec le marché
        ref_pm2 = pm2_by_type.get(property_type, global_pm2)
        diff_pct = (pm2_p - ref_pm2) / ref_pm2 * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pm2_p,
            delta={'reference': ref_pm2, 'valueformat': ',.0f',
                   'suffix': ' TND/m²', 'relative': True,
                   'increasing': {'color': '#F87171'},
                   'decreasing': {'color': '#4ADE80'}},
            number={'suffix': ' TND/m²', 'valueformat': ',.0f',
                    'font': {'size': 24, 'color': '#F0F4FF', 'family': 'DM Mono'}},
            gauge={
                'axis': {'range': [0, ref_pm2 * 2.5],
                         'tickcolor': '#3D4F8C', 'tickfont': {'color': '#6B7599', 'size': 10}},
                'bar': {'color': '#7B9BFF', 'thickness': 0.25},
                'bgcolor': '#1A1D27',
                'bordercolor': '#2D3142',
                'steps': [
                    {'range': [0, ref_pm2 * 0.8],        'color': '#1E3A2F'},
                    {'range': [ref_pm2 * 0.8, ref_pm2 * 1.2], 'color': '#1E2A3A'},
                    {'range': [ref_pm2 * 1.2, ref_pm2 * 2.5], 'color': '#3A1E1E'},
                ],
                'threshold': {
                    'line': {'color': '#FACC15', 'width': 2},
                    'thickness': 0.8,
                    'value': ref_pm2
                }
            },
            title={'text': f"vs médiane {property_type}<br><span style='font-size:0.7em;color:#6B7599'>{ref_pm2:,.0f} TND/m²</span>",
                   'font': {'color': '#C8D0F0', 'size': 13}}
        ))
        fig.update_layout(
            paper_bgcolor='#1A1D27', plot_bgcolor='#1A1D27',
            margin=dict(t=60, b=20, l=30, r=30), height=220,
            font={'family': 'DM Sans'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prédictions par modèle
        st.markdown('<div class="section-title">Détail par modèle</div>', unsafe_allow_html=True)
        for name, p in ind_preds:
            ecart = (p - price_p) / price_p * 100
            color = '#4ADE80' if ecart < 0 else '#F87171'
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.4rem 0.8rem;margin-bottom:0.3rem;
                        background:#12141E;border-radius:8px;border:1px solid #1E2235;'>
                <span style='font-size:0.82rem;color:#9CA3AF'>{name}</span>
                <span style='font-family:DM Mono;color:#F0F4FF;font-size:0.88rem'>{p:,.0f} TND</span>
                <span style='font-size:0.78rem;color:{color}'>{ecart:+.1f}%</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 2 — ANALYSE DU MARCHÉ
# ══════════════════════════════════════════════
elif page == "📊 Analyse du marché":
    st.markdown("## 📊 Analyse du marché")
    st.markdown("---")

    if df_pred is None:
        st.warning("Aucune donnée de prédictions trouvée. Lance d'abord le pipeline Colab pour générer `predictions_immobilier_tn.csv`.")
        st.stop()

    df = df_pred.copy()
    df.columns = df.columns.str.strip()

    # Renommage pour compatibilité
    rename_map = {
        'Prix annoncé TND': 'price', 'Prix prédit TND': 'prix_predit',
        'Écart %': 'ecart_pct', 'Signal': 'signal',
        'Type': 'property_type', 'Ville': 'city',
        'Surface m²': 'surface_m2', 'Pièces': 'rooms',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── KPIs ──
    n_total  = len(df)
    n_priced = df['price'].notna().sum() if 'price' in df.columns else 0
    median_p = df['price'].median() if 'price' in df.columns else 0
    n_oppo   = (df['signal'] == '🟢 Sous-évalué').sum() if 'signal' in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, delta in [
        (k1, f"{n_total:,}",           "Annonces vente",        None),
        (k2, f"{n_priced:,}",          "Avec prix annoncé",     None),
        (k3, f"{median_p/1000:.0f}k",  "Prix médian (TND)",     None),
        (k4, f"{n_oppo}",              "Opportunités détectées", "🟢 Sous-évalué"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            {f'<div class="kpi-delta signal-green">{delta}</div>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques ──
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-title">Distribution des prix annoncés</div>', unsafe_allow_html=True)
        if 'price' in df.columns:
            df_plot = df[df['price'].between(10_000, 3_000_000)]
            fig = px.histogram(df_plot, x='price', nbins=40,
                               color_discrete_sequence=['#7B9BFF'])
            fig.update_layout(
                paper_bgcolor='#1A1D27', plot_bgcolor='#1A1D27',
                font={'color': '#C8D0F0', 'family': 'DM Sans'},
                xaxis_title="Prix (TND)", yaxis_title="Nb annonces",
                margin=dict(t=10, b=40, l=40, r=10), height=280,
                showlegend=False,
                xaxis=dict(gridcolor='#1E2235', tickformat=',.0f'),
                yaxis=dict(gridcolor='#1E2235'),
            )
            fig.add_vline(x=df_plot['price'].median(), line_dash="dash",
                          line_color="#FACC15", line_width=1.5,
                          annotation_text=f"Médiane: {df_plot['price'].median():,.0f}",
                          annotation_font_color="#FACC15", annotation_font_size=10)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Signaux marché</div>', unsafe_allow_html=True)
        if 'signal' in df.columns:
            sig_counts = df['signal'].value_counts().reset_index()
            sig_counts.columns = ['signal', 'count']
            color_map = {
                '🟢 Sous-évalué': '#4ADE80', '🟡 Prix marché': '#FACC15',
                '🔴 Surévalué': '#F87171', '⚪ Pas de prix': '#6B7280',
            }
            colors = [color_map.get(s, '#9CA3AF') for s in sig_counts['signal']]
            fig2 = px.bar(sig_counts, x='signal', y='count',
                          color='signal', color_discrete_map=color_map)
            fig2.update_layout(
                paper_bgcolor='#1A1D27', plot_bgcolor='#1A1D27',
                font={'color': '#C8D0F0', 'family': 'DM Sans'},
                xaxis_title="", yaxis_title="Nb annonces",
                margin=dict(t=10, b=40, l=40, r=10), height=280,
                showlegend=False,
                xaxis=dict(gridcolor='#1E2235'),
                yaxis=dict(gridcolor='#1E2235'),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Prix médian par ville ──
    if 'city' in df.columns and 'price' in df.columns:
        st.markdown('<div class="section-title">Prix médian par ville</div>', unsafe_allow_html=True)
        city_stats = (df[df['price'].notna()]
                      .groupby('city')['price']
                      .agg(['median', 'count'])
                      .reset_index()
                      .rename(columns={'median': 'Prix médian', 'count': 'Nb'})
                      .sort_values('Prix médian', ascending=True)
                      .tail(12))
        fig3 = px.bar(city_stats, y='city', x='Prix médian',
                      orientation='h', text='Nb',
                      color='Prix médian',
                      color_continuous_scale=[[0,'#1E3A6E'],[0.5,'#3D6FB5'],[1,'#7B9BFF']])
        fig3.update_traces(texttemplate='%{text} ann.', textposition='inside',
                           textfont_size=9, textfont_color='white')
        fig3.update_layout(
            paper_bgcolor='#1A1D27', plot_bgcolor='#1A1D27',
            font={'color': '#C8D0F0', 'family': 'DM Sans'},
            xaxis_title="Prix médian (TND)", yaxis_title="",
            margin=dict(t=10, b=40, l=10, r=10), height=350,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='#1E2235', tickformat=',.0f'),
            yaxis=dict(gridcolor='#1E2235'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter prix réel vs prédit ──
    if 'price' in df.columns and 'prix_predit' in df.columns:
        df_scatter = df[df['price'].notna() & df['prix_predit'].notna()].copy()
        if len(df_scatter) > 0:
            st.markdown('<div class="section-title">Prix annoncé vs Prix prédit</div>', unsafe_allow_html=True)
            fig4 = px.scatter(df_scatter, x='prix_predit', y='price',
                              color='signal' if 'signal' in df_scatter.columns else None,
                              color_discrete_map={
                                  '🟢 Sous-évalué': '#4ADE80',
                                  '🟡 Prix marché': '#FACC15',
                                  '🔴 Surévalué': '#F87171',
                                  '⚪ Pas de prix': '#6B7280',
                              },
                              hover_data=['property_type', 'city'] if 'property_type' in df_scatter.columns else None,
                              opacity=0.6)
            max_val = max(df_scatter['price'].max(), df_scatter['prix_predit'].max())
            fig4.add_scatter(x=[0, max_val], y=[0, max_val],
                             mode='lines', line=dict(color='#FACC15', dash='dash', width=1.5),
                             name='Prix parfait', showlegend=True)
            fig4.update_layout(
                paper_bgcolor='#1A1D27', plot_bgcolor='#1A1D27',
                font={'color': '#C8D0F0', 'family': 'DM Sans'},
                xaxis_title="Prix prédit (TND)", yaxis_title="Prix annoncé (TND)",
                margin=dict(t=10, b=40, l=40, r=10), height=380,
                legend=dict(bgcolor='#12141E', bordercolor='#2D3142', borderwidth=1,
                            font=dict(size=11)),
                xaxis=dict(gridcolor='#1E2235', tickformat=',.0f'),
                yaxis=dict(gridcolor='#1E2235', tickformat=',.0f'),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 3 — DONNÉES & SIGNAUX
# ══════════════════════════════════════════════
elif page == "📋 Données & Signaux":
    st.markdown("## 📋 Données & Signaux")
    st.markdown("---")

    if df_pred is None:
        st.warning("Lance d'abord le pipeline Colab.")
        st.stop()

    df = df_pred.copy()

    # Filtres
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        col_signal = 'Signal' if 'Signal' in df.columns else 'signal'
        if col_signal in df.columns:
            signals = ['Tous'] + df[col_signal].dropna().unique().tolist()
            sig_filter = st.selectbox("Signal", signals)

    with col_f2:
        col_type = 'Type' if 'Type' in df.columns else 'property_type'
        if col_type in df.columns:
            types = ['Tous'] + df[col_type].dropna().unique().tolist()
            type_filter = st.selectbox("Type de bien", types)

    with col_f3:
        col_city = 'Ville' if 'Ville' in df.columns else 'city'
        if col_city in df.columns:
            cities = ['Toutes'] + df[col_city].dropna().unique().tolist()
            city_filter = st.selectbox("Ville", cities)

    # Application filtres
    df_filtered = df.copy()
    if col_signal in df.columns and sig_filter != 'Tous':
        df_filtered = df_filtered[df_filtered[col_signal] == sig_filter]
    if col_type in df.columns and type_filter != 'Tous':
        df_filtered = df_filtered[df_filtered[col_type] == type_filter]
    if col_city in df.columns and city_filter != 'Toutes':
        df_filtered = df_filtered[df_filtered[col_city] == city_filter]

    st.markdown(f"**{len(df_filtered):,} annonces** correspondantes")
    st.markdown("<br>", unsafe_allow_html=True)

    # Tableau
    st.dataframe(
        df_filtered.reset_index(drop=True),
        use_container_width=True,
        height=500,
        column_config={
            'Prix annoncé TND': st.column_config.NumberColumn(format="%,.0f TND"),
            'Prix prédit TND':  st.column_config.NumberColumn(format="%,.0f TND"),
            'Écart %':          st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Export
    csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "⬇️  Télécharger la sélection (CSV)",
        data=csv,
        file_name='selection_immobilier_tn.csv',
        mime='text/csv',
    )
