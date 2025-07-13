from gtda.homology import VietorisRipsPersistence
import numpy as np

def compute_persistence_diagram(distance_matrix):
    vr = VietorisRipsPersistence(metric='precomputed')
    diagrams = vr.fit_transform([distance_matrix])
    return diagrams[0]