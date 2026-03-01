"""
build_db.py - Construction de la base vectorielle juridique tunisienne
Utilise les vraies données de:
- مجلة الحقوق العينية 1965
- قانون البعث العقاري 1990
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def build_database():
    """Construit la base de données vectorielle à partir des textes juridiques."""
    
    # Vérifier que les fichiers de données existent
    data_files = [
        "data/loi_location.txt",
        "data/droit_reel.txt",
    ]
    
    all_docs = []
    
    # Paramètres de découpage optimisés pour les textes juridiques arabes/français
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Réduit pour capturer des articles entiers
        chunk_overlap=150,    # Plus de chevauchement pour ne pas couper les articles
        separators=[
            "\n================================================================\n",
            "\n\nالفصل ",    # Séparateur arabe pour les articles
            "\n\nArticle ",   # Séparateur français pour les articles
            "\n\n",
            "\n",
            ".",
            " "
        ]
    )
    
    for filepath in data_files:
        if not os.path.exists(filepath):
            print(f"⚠️  Fichier non trouvé: {filepath}")
            continue
            
        try:
            with open(filepath, encoding="utf-8") as f:
                text = f.read()
            
            docs = splitter.create_documents(
                [text],
                metadatas=[{"source": filepath, "file": os.path.basename(filepath)}]
            )
            all_docs.extend(docs)
            print(f"✅ Chargé: {filepath} → {len(docs)} chunks")
            
        except Exception as e:
            print(f"❌ Erreur lecture {filepath}: {e}")
    
    if not all_docs:
        print("❌ Aucun document chargé. Vérifiez le dossier 'data/'")
        return False
    
    print(f"\n📚 Total: {len(all_docs)} chunks à indexer...")
    
    try:
        # Embeddings avec Ollama - modèle mistral
        print("🔄 Création des embeddings (cela peut prendre quelques minutes)...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Construction de la base FAISS
        db = FAISS.from_documents(all_docs, embeddings)
        
        # Sauvegarde
        os.makedirs("db", exist_ok=True)
        db.save_local("db")
        
        print(f"\n✅ Base juridique créée avec succès!")
        print(f"   → {len(all_docs)} articles indexés")
        print(f"   → Base sauvegardée dans: ./db/")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur création base: {e}")
        print("\n🔧 Solutions:")
        print("1. Vérifiez qu'Ollama est lancé: ollama serve")
        print("2. Vérifiez que le modèle est installé: ollama pull mistral")
        print("3. Vérifiez la connexion à Ollama (port 11434)")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🏗️  Construction de la base juridique immobilière tunisienne")
    print("=" * 60)
    print("Sources:")
    print("  • مجلة الحقوق العينية (1965)")
    print("  • قانون البعث العقاري (1990)")
    print("=" * 60)
    build_database()