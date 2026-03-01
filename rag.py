"""
rag.py - Moteur RAG optimisé pour l'assistant juridique immobilier tunisien
OPTIMISATIONS v2 :
- nomic-embed-text au lieu de Mistral pour les embeddings (5-10x plus rapide)
- llama3.2:3b au lieu de Mistral 7B pour la génération (3-4x plus rapide)
- Cache FAISS : base chargée une seule fois (plus de rechargement à chaque question)
- num_predict réduit à 600 tokens (suffisant pour les réponses juridiques)
- k=3 au lieu de k=6 (contexte plus court = prompt plus court = plus rapide)
- Contexte hardcodé condensé (moins de tokens injectés)
"""

import re
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# ============================================================
# RÈGLES FISCALES HARDCODÉES — VERSION CONDENSÉE (Tunisie 2025)
# ============================================================

FISCAL_RULES = """
=== FISCALITÉ IMMOBILIÈRE TUNISIE 2025 (LOI DE FINANCES N°48-2024) ===

TVA LOGEMENT NEUF (promoteur agréé) :
- Prix ≤ 400 000 DT → TVA 7%  | ex: 300k DT → TVA=21k → TTC=321k
- Prix > 400 000 DT → TVA 13% | ex: 600k DT → TVA=78k → TTC=678k
- Taux 19% reporté au 01/01/2026 (4ème report depuis 2022)
- PAS de TVA sur reventes entre particuliers (bien ancien)

DROITS D'ENREGISTREMENT (bien ancien / entre particuliers) :
- Prix < 500 000 DT    → 6%  | ex: 400k → 24 000 DT
- Prix 500k–999 999 DT → 8%  | ex: 600k → 48 000 DT
- Prix ≥ 1 000 000 DT  → 10% | ex: 1,2M → 120 000 DT
- Taxe CPF : +1% sur tout achat
- Frais notaire : 1% à 5% du prix

TRE (paiement 100% devises) :
- CPF 1% uniquement + droit fixe 30 DT/page (pas de % sur droits d'enregistrement)
- Économie potentielle : 20 000 à 60 000 DT selon le prix

IS PROMOTEURS :
- Taux général : 20% | Logements sociaux : exo 50% | Logements prioritaires : exo 25%
- Réinvestissement : exo jusqu'à 35% des revenus réinvestis

REVENUS LOCATIFS (personnes physiques) :
- Déduction forfaitaire : 25% du loyer brut (LF2025, augmentée de 20%)
- TIB déductible | Dépenses réparation NON déductibles depuis LF2025

TAUX BANCAIRES :
- Taux directeur BCT : 8% | Crédit immobilier marché : 8%-11%
- FOPROLOS : ≤ 2% (salaire brut ≤ 3 162 DT/mois) | Durée : jusqu'à 75 ans d'âge
- Contribution employeurs FOPROLOS : 1% masse salariale brute
"""

DOCUMENTS_VENTE = """
=== DOCUMENTS VENTE IMMOBILIÈRE TUNISIE ===

VENDEUR : CIN valide | Titre foncier original (الرسم العقاري) | Cert. situation matrimoniale
(si marié → accord écrit conjoint) | Cert. non-opposition/non-hypothèque | Quittances TIB
à jour | Attestation non-dette fiscale (شهادة عدم المديونية)

BIEN : Titre foncier (الرسم العقاري) — INDISPENSABLE | Plan cadastral/topographique
| Permis de construire original | PV réception travaux | Règlement copropriété si appartement

HÉRITIERS : Acte dévolution successorale (شهادة الورثة, tribunal) | Accord TOUS héritiers
| Jugement partage si applicable

SOCIÉTÉS : Extrait registre commerce < 3 mois | PV CA autorisant cession | Statuts à jour

PROCÉDURE : (1) Rassembler docs → (2) Acte chez notaire → (3) Payer droits enregistrement
→ (4) Dépôt Bureau Propriété Foncière → (5) Nouveau titre foncier (délai 2-8 semaines)
"""

DROITS_EXPULSION = """
=== DROITS LOCATAIRE / PROCÉDURE EXPULSION TUNISIE ===

PRINCIPE : Aucune expulsion SANS décision judiciaire. Expulsion forcée sans jugement = illégal.

CAUSES LÉGALES : Non-paiement loyer (après mise en demeure 2 mois sans suite) | Violation
grave obligations | Sous-location non autorisée | Détérioration grave | Fin bail + refus quitter

PROCÉDURE LÉGALE :
1. Mise en demeure par huissier (عدل منفذ)
2. Si sans réponse 2 mois → saisine juge cantonal (محكمة الناحية)
3. Audience contradictoire (locataire peut se défendre)
4. Jugement d'expulsion
5. Signification au locataire par huissier
6. Délai d'exécution accordé (1-3 mois généralement)
7. Exécution par huissier + forces de l'ordre si nécessaire

DROITS DU LOCATAIRE : Être entendu | Régulariser pour stopper procédure | Délai pour
reloger | Faire appel | Assistance avocat

OCCUPATION SANS BAIL : Propriétaire doit saisir tribunal (ne peut pas expulser seul)
| Référé possible si occupation récente | Plainte pénale pour violation domicile
| Délai procédure : 3-12 mois

RÉVISION LOYER : Uniquement si clause prévue au contrat | Sans clause : loyer fixe
| Basée sur IPC (INS) | Max 1 révision/an | Locataire peut refuser révision non prévue
"""


# ============================================================
# CACHE FAISS — chargé une seule fois
# ============================================================

_db_cache = None

def get_db():
    """Retourne la base FAISS depuis le cache (chargée une seule fois au démarrage)."""
    global _db_cache
    if _db_cache is None:
        try:
            _db_cache = FAISS.load_local(
                "db",
                OllamaEmbeddings(model="nomic-embed-text"),  # ✅ Modèle dédié embeddings
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"⚠️ Base FAISS non disponible: {e}")
            _db_cache = None
    return _db_cache


# ============================================================
# DÉTECTION DE LANGUE
# ============================================================

def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return "ar" if arabic_chars > len(text) * 0.15 else "fr"


# ============================================================
# DÉTECTION DU TYPE DE QUESTION
# ============================================================

def detect_question_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ['tva', 'taxe', 'impôt', 'enregistrement', 'fiscal', 'tre', 'devises',
                             'ضريبة', 'أداء', 'تسجيل', 'جباية', 'اعفاء', 'امتياز']):
        return "fiscal"
    if any(k in q for k in ['document', 'dossier', 'pièce', 'vendre', 'acte', 'notaire',
                             'وثيقة', 'ملف', 'بيع', 'كاتب عدل', 'وكالة']):
        return "documents"
    if any(k in q for k in ['expulsion', 'locataire', 'expulser', 'occupation', 'bail', 'loyer',
                             'révision', 'augmentation', 'طرد', 'مكتري', 'إخلاء', 'كراء']):
        return "expulsion"
    return "general"


# ============================================================
# CALCULS FINANCIERS
# ============================================================

def is_calculation_question(question: str) -> bool:
    calc_kw = ['calculer', 'calcul', 'combien', 'mensualité', 'mensualite', 'crédit', 'prêt',
               'emprunt', 'rendement', 'rentabilité', 'cash flow', 'roi', 'endettement',
               'intérêts', 'amortissement', 'm²', 'm2', 'mètre', 'superficie',
               'تحسب', 'احسب', 'حساب', 'قسط', 'قرض', 'مردودية', 'عائد', 'فائدة', 'متر مربع']
    q = question.lower()
    has_amount = bool(re.search(r'\d{3,}', question))
    financial_kw = ['tva', 'droit', 'loyer', 'taux', 'prix', 'valeur', 'bien', 'salaire']
    return any(k in q for k in calc_kw) or (has_amount and any(k in q for k in financial_kw))


def extract_numbers(text: str) -> list:
    cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    cleaned = cleaned.replace(',', '.')
    nums = re.findall(r'\b\d+\.?\d*\b', cleaned)
    result = []
    for n in nums:
        try:
            v = float(n)
            if v > 0:
                result.append(v)
        except Exception:
            pass
    return result


def calc_mensualite(principal, annual_rate, years):
    r = annual_rate / 100 / 12
    n = years * 12
    if r == 0:
        m = principal / n
    else:
        m = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
    total = m * n
    return {"mensualite": round(m, 2), "total": round(total, 2), "interets": round(total - principal, 2)}


def handle_calculation(question: str, lang: str):
    q = question.lower()
    numbers = extract_numbers(question)
    AR = lang == "ar"

    # ── TVA ──────────────────────────────────────────────────
    if any(k in q for k in ['tva', 'ضريبة القيمة', 'أداء على القيمة']):
        montants = [n for n in numbers if n > 1000]
        if montants:
            prix = montants[0]
            taux = 7 if prix <= 400000 else 13
            tva = prix * taux / 100
            ttc = prix + tva
            if AR:
                return f"""💰 **حساب الأداء على القيمة المضافة (TVA) — 2025**

| | |
|---|---|
| 🏠 السعر الخام (HT) | **{prix:,.0f} دت** |
| 📊 نسبة TVA | **{taux}%** {"(سعر ≤ 400 000 دت)" if taux==7 else "(سعر > 400 000 دت)"} |
| 💸 مبلغ TVA | **{tva:,.0f} دت** |
| ✅ السعر الجملي (TTC) | **{ttc:,.0f} دت** |

**القاعدة القانونية (قانون المالية 2025) :**
- سعر ≤ 400 000 دت → TVA **7%**
- سعر > 400 000 دت → TVA **13%**
- نسبة 19% مُرجأة إلى 1 جانفي 2026

⚠️ تنطبق فقط على **المساكن الجديدة** من باعث عقاري مرخص. العقارات القديمة بين الخواص لا تخضع لـ TVA."""
            else:
                return f"""💰 **Calcul de la TVA immobilière — 2025**

| | |
|---|---|
| 🏠 Prix HT | **{prix:,.0f} DT** |
| 📊 Taux TVA | **{taux}%** {"(prix ≤ 400 000 DT)" if taux==7 else "(prix > 400 000 DT)"} |
| 💸 Montant TVA | **{tva:,.0f} DT** |
| ✅ Prix TTC | **{ttc:,.0f} DT** |

**Règle légale (Loi de Finances 2025) :**
- Prix ≤ 400 000 DT → TVA **7%**
- Prix > 400 000 DT → TVA **13%**
- Taux 19% reporté au 1er janvier 2026

⚠️ S'applique uniquement aux **logements neufs** vendus par un promoteur agréé. Pas de TVA pour les reventes entre particuliers."""

    # ── DROITS D'ENREGISTREMENT ──────────────────────────────
    if any(k in q for k in ['enregistrement', "taux d'enregistrement", 'حقوق تسجيل', 'droits', 'frais enregistrement']):
        montants = [n for n in numbers if n > 1000]
        if montants:
            prix = montants[0]
            if prix < 500000:
                taux, label = 6, "Prix < 500 000 DT"
            elif prix < 1000000:
                taux, label = 8, "500 000 ≤ Prix < 1 000 000 DT"
            else:
                taux, label = 10, "Prix ≥ 1 000 000 DT"
            droits = prix * taux / 100
            cpf = prix * 0.01
            total = droits + cpf
            if AR:
                return f"""📋 **حساب معاليم التسجيل — عقار قديم (بين خواص)**

| | |
|---|---|
| 💵 ثمن البيع | **{prix:,.0f} دت** |
| 📊 نسبة معاليم التسجيل | **{taux}%** |
| 💼 معاليم التسجيل | **{droits:,.0f} دت** |
| 🏦 رسم CPF (1%) | **{cpf:,.0f} دت** |
| ✅ **المجموع** | **{total:,.0f} دت** |

**الجدول التصاعدي :** أقل من 500k → **6%** | 500k-999k → **8%** | مليون فما فوق → **10%**
يُضاف رسم CPF **1%** + أتعاب كاتب العدل (1%-5%)

⚠️ هذه المعاليم خاصة **بالعقارات القديمة**. العقارات الجديدة تخضع لـ TVA."""
            else:
                return f"""📋 **Calcul des droits d'enregistrement — bien ancien (entre particuliers)**

| | |
|---|---|
| 💵 Prix de vente | **{prix:,.0f} DT** |
| 📊 Taux applicable | **{taux}%** ({label}) |
| 💼 Droits d'enregistrement | **{droits:,.0f} DT** |
| 🏦 Taxe CPF (1%) | **{cpf:,.0f} DT** |
| ✅ **Total minimum** | **{total:,.0f} DT** |

**Barème :** < 500k → **6%** | 500k-999k → **8%** | ≥ 1M → **10%**
+ Taxe CPF **1%** + honoraires notaire (1%-5%)

⚠️ Ces droits concernent les **biens anciens**. Biens neufs chez promoteur agréé → TVA."""

    # ── MENSUALITÉ CRÉDIT ─────────────────────────────────────
    if any(k in q for k in ['mensualité', 'mensualite', 'crédit', 'prêt', 'emprunt', 'credit',
                              'قسط', 'قرض', 'تمويل', 'اقتراض']):
        large = sorted([n for n in numbers if n > 1000], reverse=True)
        taux_c = [n for n in numbers if 0 < n < 25]
        duree_c = [n for n in numbers if 5 <= n <= 40]
        if large:
            montant = large[0]
            taux = taux_c[0] if taux_c else 8.0
            duree = int(duree_c[0]) if duree_c else 20
            r = calc_mensualite(montant, taux, duree)
            if AR:
                return f"""📊 **حساب القسط الشهري للقرض العقاري**

| | |
|---|---|
| 💵 مبلغ القرض | **{montant:,.0f} دت** |
| 📊 نسبة الفائدة السنوية | **{taux}%** |
| 📅 مدة القرض | **{duree} سنة** ({duree*12} قسط) |
| 💰 **القسط الشهري** | **{r['mensualite']:,.2f} دت** |
| 📈 مجموع الفوائد | **{r['interets']:,.2f} دت** |
| 💵 المبلغ الجملي المسدَّد | **{r['total']:,.2f} دت** |

💡 نسبة BCT المرجعية : **8%** — معدل السوق : **8% إلى 11%**
⚠️ تواصل مع بنكك للحصول على العرض الرسمي"""
            else:
                return f"""📊 **Calcul de la mensualité du crédit immobilier**

| | |
|---|---|
| 💵 Montant emprunté | **{montant:,.0f} DT** |
| 📊 Taux annuel | **{taux}%** |
| 📅 Durée | **{duree} ans** ({duree*12} mensualités) |
| 💰 **Mensualité** | **{r['mensualite']:,.2f} DT** |
| 📈 Total intérêts | **{r['interets']:,.2f} DT** |
| 💵 Total remboursé | **{r['total']:,.2f} DT** |

💡 Taux BCT : **8%** — Taux marché : **8% à 11%** selon banque et profil
⚠️ Consultez votre banque pour une offre officielle personnalisée"""
        else:
            ex = calc_mensualite(200000, 8, 20)
            if AR:
                return f"""📊 **حساب القسط الشهري — مثال توضيحي**

قرض **200 000 دت** بنسبة **8%** على **20 سنة** :
- القسط الشهري : **{ex['mensualite']:,.2f} دت**
- مجموع الفوائد : **{ex['interets']:,.2f} دت**
- المبلغ الجملي : **{ex['total']:,.2f} دت**

🎯 أعطني : **المبلغ + النسبة + المدة** لأحسب قسطك الشخصي!"""
            else:
                return f"""📊 **Calcul de mensualité — exemple**

Prêt de **200 000 DT** à **8%** sur **20 ans** :
- Mensualité : **{ex['mensualite']:,.2f} DT**
- Total intérêts : **{ex['interets']:,.2f} DT**
- Total remboursé : **{ex['total']:,.2f} DT**

🎯 Donnez-moi : **montant + taux + durée** pour un calcul personnalisé !"""

    # ── RENDEMENT LOCATIF ─────────────────────────────────────
    if any(k in q for k in ['rendement', 'rentabilité', 'roi', 'cash flow', 'مردودية', 'عائد']):
        big = sorted([n for n in numbers if n > 500], reverse=True)
        if len(big) >= 2:
            valeur = big[0]
            loyer_m = big[1]
            loyer_an = loyer_m * 12
            brut = round(loyer_an / valeur * 100, 2)
            net = round((loyer_an * 0.75) / valeur * 100, 2)
            if AR:
                return f"""📈 **حساب المردودية الإيجارية**

| المؤشر | القيمة |
|--------|--------|
| 🏠 قيمة العقار | {valeur:,.0f} دت |
| 🔑 الإيجار الشهري | {loyer_m:,.0f} دت |
| 📅 الإيجار السنوي | {loyer_an:,.0f} دت |
| 📊 **مردودية خامة** | **{brut}%** |
| 💰 **مردودية صافية** | **{net}%** (بعد طرح 25% جزافي) |

💡 مردودية جيدة في تونس = **5% إلى 8%** صافية
⚠️ أرقام تقريبية — تتوقف على التكاليف الحقيقية والضرائب"""
            else:
                return f"""📈 **Calcul du rendement locatif**

| Indicateur | Valeur |
|-----------|--------|
| 🏠 Valeur du bien | {valeur:,.0f} DT |
| 🔑 Loyer mensuel | {loyer_m:,.0f} DT |
| 📅 Loyer annuel | {loyer_an:,.0f} DT |
| 📊 **Rendement brut** | **{brut}%** |
| 💰 **Rendement net** | **{net}%** (après déduction 25% LF2025) |

💡 Bon rendement en Tunisie = **5% à 8%** net
⚠️ Chiffres indicatifs — dépend des charges réelles et de la fiscalité"""

    # ── ENDETTEMENT ───────────────────────────────────────────
    if any(k in q for k in ['endettement', 'capacité', 'salaire', 'revenu', 'نسبة المديونية', 'قدرة']):
        montants = [n for n in numbers if n > 100]
        if montants:
            sal = montants[0]
            cap33 = round(sal * 0.33, 2)
            cap40 = round(sal * 0.40, 2)
            if AR:
                return f"""💼 **نسبة المديونية والقدرة على التسديد**

| | |
|---|---|
| 💰 الراتب الشهري | **{sal:,.0f} دت** |
| 📊 النسبة القياسية (33%) | → قسط أقصى **{cap33:,.2f} دت** |
| 📊 النسبة القصوى (40%) | → قسط أقصى **{cap40:,.2f} دت** |

أغلب البنوك التونسية : حد **33%** من الدخل الصافي | بعضها يقبل **40%** حسب الملف
⚠️ اتصل ببنكك للحصول على تقييم دقيق"""
            else:
                return f"""💼 **Taux d'endettement et capacité d'emprunt**

| | |
|---|---|
| 💰 Salaire mensuel | **{sal:,.0f} DT** |
| 📊 Règle des 33% (standard) | → Mensualité max **{cap33:,.2f} DT** |
| 📊 Règle des 40% (certaines banques) | → Mensualité max **{cap40:,.2f} DT** |

Standard banques tunisiennes : **33% du revenu net** | Certaines acceptent **40%** selon profil
⚠️ Consultez votre banque pour une évaluation précise"""

    # ── PRIX AU M² ────────────────────────────────────────────
    if any(k in q for k in ['m²', 'mètre', 'm2', 'superficie', 'متر مربع', 'ثمن المتر', 'prix au m']):
        nums_sorted = sorted([n for n in numbers if n > 0], reverse=True)
        if len(nums_sorted) >= 2:
            valeur = nums_sorted[0]
            surface = nums_sorted[1] if nums_sorted[1] < 2000 else None
            if surface:
                pm2 = round(valeur / surface, 2)
                if AR:
                    return f"""🏠 **تقدير قيمة العقار بالمتر المربع**

| | |
|---|---|
| 💵 القيمة الإجمالية | {valeur:,.0f} دت |
| 📐 المساحة | {surface:,.0f} م² |
| 📊 **سعر المتر المربع** | **{pm2:,.2f} دت/م²** |

**أسعار مرجعية تونس 2024-2025 :**
- تونس العاصمة / وسط : 3 500 – 6 000 دت/م²
- مناطق راقية (المرسى، قرطاج) : 4 000 – 8 000 دت/م²
- ضواحي العاصمة : 2 000 – 4 000 دت/م²
- المدن الداخلية : 800 – 2 500 دت/م²"""
                else:
                    return f"""🏠 **Estimation de valeur au m²**

| | |
|---|---|
| 💵 Valeur totale | {valeur:,.0f} DT |
| 📐 Surface | {surface:,.0f} m² |
| 📊 **Prix au m²** | **{pm2:,.2f} DT/m²** |

**Références Tunisie 2024-2025 :**
- Tunis centre : 3 500 – 6 000 DT/m²
- Zones huppées (La Marsa, Carthage) : 4 000 – 8 000 DT/m²
- Banlieues périphériques : 2 000 – 4 000 DT/m²
- Grandes villes intérieur : 800 – 2 500 DT/m²"""

    return None


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def ask(question: str, k: int = 3) -> str:  # ✅ k=3 au lieu de k=6
    lang = detect_language(question)
    question_type = detect_question_type(question)

    # 1. Calculs financiers → réponse directe (pas de LLM nécessaire)
    if is_calculation_question(question):
        result = handle_calculation(question, lang)
        if result:
            return result

    # 2. Charger base FAISS depuis le cache ✅
    db = get_db()
    if db:
        try:
            docs = db.similarity_search(question, k=k)
            rag_context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        except Exception:
            rag_context = ""
    else:
        rag_context = ""

    # 3. Contexte enrichi selon le type (version condensée)
    extra = {
        "fiscal": FISCAL_RULES,
        "documents": DOCUMENTS_VENTE,
        "expulsion": DROITS_EXPULSION,
    }.get(question_type, "")

    full_context = f"{extra}\n\n{rag_context}".strip()

    if not full_context:
        if lang == "ar":
            return "لم أجد معلومات كافية في قاعدة البيانات القانونية. يُرجى استشارة محامٍ أو كاتب عدل."
        return "Je n'ai pas trouvé d'informations suffisantes dans la base juridique. Veuillez consulter un avocat ou un notaire."

    # 4. Prompt avec langue forcée
    if lang == "ar":
        lang_instruction = "أجب باللغة العربية فقط."
        label = "الجواب:"
    else:
        lang_instruction = "Réponds UNIQUEMENT en français."
        label = "Réponse :"

    prompt = f"""Expert en droit immobilier tunisien. {lang_instruction}

RÈGLES :
1. Langue {lang.upper()} uniquement
2. Cite l'article ou la loi à chaque affirmation clé
3. Base-toi UNIQUEMENT sur les textes fournis
4. Si info absente : dis "Information non disponible" — NE PAS inventer

TEXTES JURIDIQUES :
{full_context}

QUESTION : {question}

{label}"""

    # ✅ Modèle rapide + num_predict réduit
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.05, num_predict=600)
    return llm.invoke(prompt)