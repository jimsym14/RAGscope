import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from document_pipeline import get_chroma_collection


@st.fragment
def _render_pca_tab() -> None:
    
    with st.container(border=True):
        with st.spinner("Generating 3D PCA Projection from ChromaDB embeddings..."):
            try:
                coll = get_chroma_collection()
                data = coll.get(include=["embeddings", "metadatas", "documents"])
                raw_emb = data.get("embeddings")
                raw_meta = data.get("metadatas")
                raw_docs = data.get("documents")

                embeddings = raw_emb if raw_emb is not None else []
                metadatas = raw_meta if raw_meta is not None else []
                documents = raw_docs if raw_docs is not None else []

                if len(embeddings) == 0:
                    st.warning("No embeddings found in vector database.")
                else:
                    embeddings_np = np.asarray(embeddings, dtype=np.float32)

                    pca = PCA(n_components=3)
                    pos_3d = pca.fit_transform(embeddings_np)

                    df = pd.DataFrame({
                        "X": pos_3d[:, 0],
                        "Y": pos_3d[:, 1],
                        "Z": pos_3d[:, 2],
                        "File": [m.get("file_name") or m.get("filename", "Unknown") if m else "Unknown" for m in metadatas],
                        "Snippet": [str(doc)[:150] + "..." for doc in documents]
                    })

                    fig_scatter = px.scatter_3d(
                        df, x="X", y="Y", z="Z", color="File",
                        hover_data=["Snippet"],
                        template="plotly_dark",
                        height=620
                    )
                    fig_scatter.update_traces(marker=dict(size=4, opacity=0.85))
                    fig_scatter.update_layout(
                        margin=dict(l=0, r=0, b=0, t=10),
                        title=dict(text=""),
                        legend=dict(
                            font=dict(size=11),
                            orientation="v",
                            yanchor="top",
                            y=0.98,
                            xanchor="right",
                            x=0.99
                        ),
                        scene=dict(
                            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    if hasattr(fig_scatter, "layout") and hasattr(fig_scatter.layout, "title") and fig_scatter.layout.title is not None:
                        fig_scatter.layout.title.text = ""
                    st.plotly_chart(fig_scatter, width="stretch")
                    st.caption(f"Displaying {len(embeddings)} text chunk vectors projected into 3D using Principal Component Analysis (PCA).")
            except Exception as e:
                st.error(f"Error loading 3D vector embeddings: {e}")


@st.fragment
def _render_network_tab() -> None:
    
    with st.container(border=True):
        with st.spinner("Computing cosine similarities and building 3D semantic graph..."):
            try:
                coll = get_chroma_collection()
                data = coll.get(include=["embeddings", "metadatas", "documents"])
                raw_emb = data.get("embeddings")
                raw_meta = data.get("metadatas")
                raw_docs = data.get("documents")

                embeddings = raw_emb if raw_emb is not None else []
                metadatas = raw_meta if raw_meta is not None else []
                documents = raw_docs if raw_docs is not None else []

                if len(embeddings) == 0:
                    st.warning("No embeddings found in vector database.")
                else:
                    embeddings_np = np.asarray(embeddings, dtype=np.float32)
                    pca = PCA(n_components=3)
                    pos_3d = pca.fit_transform(embeddings_np)

                    sim_matrix = cosine_similarity(embeddings_np)

                    K = 3
                    edges = []
                    for i in range(len(embeddings_np)):
                        sim_idx = np.argsort(sim_matrix[i])[-(K+1):-1]
                        for j in sim_idx:
                            edges.append(tuple(sorted((i, j))))
                    edges = list(set(edges))

                    edge_x, edge_y, edge_z = [], [], []
                    for e in edges:
                        p0, p1 = pos_3d[e[0]], pos_3d[e[1]]
                        edge_x.extend([p0[0], p1[0], None])
                        edge_y.extend([p0[1], p1[1], None])
                        edge_z.extend([p0[2], p1[2], None])

                    edge_trace = go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        line=dict(width=1.5, color='rgba(200, 200, 200, 0.25)'),
                        hoverinfo='none',
                        mode='lines',
                        name='Semantic Similarity'
                    )

                    files = [m.get("file_name") or m.get("filename", "Unknown") if m else "Unknown" for m in metadatas]
                    doc_texts = [str(doc) for doc in documents]
                    node_trace = go.Scatter3d(
                        x=pos_3d[:, 0], y=pos_3d[:, 1], z=pos_3d[:, 2],
                        mode='markers',
                        hoverinfo='text',
                        text=[f"<b>{f}</b><br>{d[:120]}..." for f, d in zip(files, doc_texts)],
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            size=5,
                            opacity=0.85,
                            color=np.arange(len(embeddings_np))
                        ),
                        name='Chunks'
                    )

                    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                        template="plotly_dark",
                        showlegend=False,
                        hovermode='closest',
                        title=dict(text=""),
                        margin=dict(l=0, r=0, b=0, t=10),
                        height=620,
                        scene=dict(
                            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.12)"),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    ))
                    if hasattr(fig_net, "layout") and hasattr(fig_net.layout, "title") and fig_net.layout.title is not None:
                        fig_net.layout.title.text = ""
                    st.plotly_chart(fig_net, width="stretch")
                    st.caption(f"Semantic graph displaying {len(embeddings)} chunk nodes and {len(edges)} top-3 cosine similarity edges.")
            except Exception as e:
                st.error(f"Error generating 3D semantic network: {e}")


def render_visualizer_view() -> None:
    
    tab_pca, tab_network = st.tabs([
        ":material/view_in_ar: 3D Embedding Space (PCA)",
        ":material/hub: Semantic Correlation Network"
    ])

    with tab_pca:
        _render_pca_tab()

    with tab_network:
        _render_network_tab()
