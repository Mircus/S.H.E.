import streamlit as st
import xgi
import plotly.graph_objects as go

def draw_simplicial_set(H):
    edge_traces = []
    for e in H.edges:
        nodes = list(H.edges[e])
        edge_traces.append(go.Scatter(x=list(range(len(nodes))), y=[e]*len(nodes),
                                      mode='lines+markers', name=f"Edge {e}"))

    fig = go.Figure(data=edge_traces)
    st.plotly_chart(fig)

def main():
    st.title("SHE - Social Hypergraph Engine")
    st.write("Visualize simplicial sets")

    if st.button("Load Example"):
        H = xgi.Hypergraph([[0, 1], [1, 2, 3]])
        draw_simplicial_set(H)

if __name__ == "__main__":
    main()