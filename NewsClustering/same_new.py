import json
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dateutil import parser
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from flair.models import SequenceTagger
from flair.data import Sentence
import string
from community import community_louvain
from tqdm import tqdm
import torch

# Constantes
PUNCT = string.punctuation + '“”‘’'
news_dir = r"../data/news_cleaned.json"  # <-- EDITABLE

# Configuración
BATCH_SIZE_ENTS = 32
BATCH_SIZE_EMB = 32
EMBED_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
FLAIR_MODEL_NAME = "flair/ner-spanish-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batch_process_entities_flair(df, batch_size=8, out_path="entities.csv"):
    import gc
    tagger = SequenceTagger.load(FLAIR_MODEL_NAME)
    results = []
    texts = df["texto"].tolist()
    indices = df.index.tolist()
    for start in tqdm(range(0, len(texts), batch_size), desc="Extrayendo entidades (Flair NER)"):
        batch = texts[start:start+batch_size]
        batch_ids = indices[start:start+batch_size]

        # (Opcional) Trunca cada texto a 512 caracteres o tokens
        batch = [t[:512] for t in batch]

        flair_sents = [Sentence(t) for t in batch]
        tagger.predict(flair_sents, mini_batch_size=batch_size, verbose=False)
        for idx, sent in zip(batch_ids, flair_sents):
            ents = {e.text.strip(PUNCT).lower()
                    for e in sent.get_spans("ner")
                    if e.get_label("ner").value in {"PER", "ORG", "LOC", "MISC", "DATE"}}
            results.append({"idx": idx, "entities": list(ents)})
        # Guardar por batch
        pd.DataFrame(results).to_csv(out_path, mode='a',
                                     header=not os.path.exists(out_path), index=False)
        results = []
        torch.cuda.empty_cache()
        gc.collect()
    df_ents = pd.read_csv(out_path)
    return df_ents.set_index("idx").sort_index()


def batched_encode_to_disk(model, texts, batch_size=BATCH_SIZE_EMB, out_path="embeddings.npy"):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch encoding embeddings"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False,
                           convert_to_numpy=True, device=DEVICE)
        embeddings.append(emb)
        np.save(out_path, np.vstack(embeddings))
    return np.vstack(embeddings)


def tfidf_score(ents, df_entity_freq, total_docs):
    score = 0
    for e in ents:
        tf = ents.count(e)
        dfreq = df_entity_freq.get(e, 1)
        idf = np.log(total_docs / (1 + dfreq))
        score += tf * idf
    return score


def compute_edge(i, j, dist, df, dates, max_date_diff, df_entity_freq, total_docs, alpha, beta, gamma):
    cos_sim = 1 - dist
    ents_i, ents_j = list(df.at[i, "entities"]), list(df.at[j, "entities"])
    if not ents_i or not ents_j:
        return None
    jacc = len(set(ents_i) & set(ents_j)) / \
        (len(set(ents_i) | set(ents_j)) + 1e-6)
    dt_diff = abs(dates.iloc[i] - dates.iloc[j])
    t_norm = 1 - (dt_diff / max_date_diff)
    tfidf_weight = (tfidf_score(ents_i, df_entity_freq, total_docs) +
                    tfidf_score(ents_j, df_entity_freq, total_docs)) / 2
    w = alpha * jacc + beta * cos_sim + gamma * t_norm
    w *= (1 + np.log1p(tfidf_weight))
    if w > 0.3:
        return (i, j, w)
    return None


def main():
    # Paso 1: Carga y preproceso
    print("Paso 1: Cargando datos...")
    with open(news_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["newspaper"] != "EL MUNDO"].copy()
    df.reset_index(drop=True, inplace=True)
    df["texto"] = (df["headline"].fillna("") + " " +
                   df["body"].fillna("")).str.lower()
    df["fecha_dt"] = df["fecha"].apply(
        lambda x: parser.parse(x, dayfirst=True))
    print(f"  -> Cargadas {len(df)} noticias")

    # Paso 2: Extracción de entidades (por lotes, GPU)
    print("Paso 2: Extrayendo entidades (Flair, GPU)...")
    entities_file = "entities.csv"
    if not os.path.exists(entities_file):
        df_ents = batch_process_entities_flair(
            df, batch_size=BATCH_SIZE_ENTS, out_path=entities_file)
    else:
        df_ents = pd.read_csv(entities_file).set_index("idx").sort_index()
    df["entities"] = df_ents["entities"].apply(eval)
    print("  -> Entidades extraídas y cargadas")

    # Paso 3: Embeddings por lotes
    print("Paso 3: Calculando embeddings por lotes y guardando en disco (GPU)...")
    sbert = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    emb_file = "embeddings.npy"
    if not os.path.exists(emb_file):
        embeddings = batched_encode_to_disk(
            sbert, df["texto"].tolist(), batch_size=BATCH_SIZE_EMB, out_path=emb_file)
    else:
        embeddings = np.load(emb_file)
    print(f"  -> Embeddings shape: {embeddings.shape}")

    # PCA para reducción
    print("Paso 4: Reducción de dimensionalidad PCA...")
    pca = PCA(n_components=100)
    embeddings_reduced = pca.fit_transform(embeddings)

    # Paso 5: k-NN
    print("Paso 5: Construyendo índice k-NN...")
    k = 30
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=1)
    nn.fit(embeddings_reduced)
    distances, neighbors = nn.kneighbors(embeddings_reduced)
    neighbors, distances = neighbors[:, 1:], distances[:, 1:]

    # Paso 6: TF-IDF
    print("Paso 6: Calculando TF-IDF sobre entidades...")
    all_entities = [ent for ents in df["entities"] for ent in ents]
    df_entity_freq = pd.Series(all_entities).value_counts()
    total_docs = len(df)

    # Paso 7: Grafo
    print("Paso 7: Construyendo grafo...")
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    dates = df["fecha_dt"].astype(np.int64)
    max_date_diff = dates.max() - dates.min()
    alpha, beta, gamma = 0.5, 0.4, 0.1
    edges = []
    for i in tqdm(range(len(df)), desc="Generando aristas"):
        for idx, j in enumerate(neighbors[i]):
            res = compute_edge(
                i, j, distances[i, idx], df, dates, max_date_diff, df_entity_freq, total_docs, alpha, beta, gamma)
            if res:
                edges.append(res)
    G.add_weighted_edges_from(edges)

    # Paso 8: Louvain
    print("Paso 8: Aplicando Louvain...")
    partition = community_louvain.best_partition(G, weight="weight")
    df["cluster"] = df.index.map(partition)

    # Paso 9: Clustering jerárquico
    print("Paso 9: Clustering jerárquico...")
    new_clusters = {}
    next_cluster_id = df["cluster"].max() + 1
    for cid, group in df.groupby("cluster"):
        if len(group) > 20:
            X = embeddings_reduced[group.index]
            hclust = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.5)
            sub_labels = hclust.fit_predict(X)
            for sub_label, idx in zip(sub_labels, group.index):
                new_clusters[idx] = next_cluster_id + sub_label
            next_cluster_id += sub_labels.max() + 1
    df["cluster"] = df.index.map(
        lambda i: new_clusters.get(i, df.at[i, "cluster"]))

    # Paso 10: Visualización
    print("Paso 10: Visualizando grafo...")
    color_map = [df["cluster"].iloc[n] for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42, k=0.15)
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(G, pos, node_size=30,
                           node_color=color_map, cmap=plt.cm.get_cmap("tab20"))
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.axis("off")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/grafo_clusters.png", dpi=300)

    # Paso 11: Guardado
    print("Paso 11: Guardando resultados...")
    with open("results/event_graph_louvain.txt", "w", encoding="utf-8") as f:
        for cid, group in df.groupby("cluster"):
            f.write(f"\n🔹 Cluster {cid} ({len(group)} noticias):\n")
            ents = [ent for ents in group["entities"] for ent in ents]
            top_ents = pd.Series(ents).value_counts().head(5).index.tolist()
            f.write(f"Tema: Entidades({', '.join(top_ents)})\n\n")
            for idx, row in group.iterrows():
                f.write(f"  - [{row['id']}] {row['headline']}\n")
    print("✅ Pipeline ejecutado correctamente (Flair+SBERT en GPU, robusto y eficiente).")


if __name__ == "__main__":
    main()
