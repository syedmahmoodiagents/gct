import streamlit as st
import numpy as np
import pandas as pd
# import seaborn as snb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.header("Data Upload")
file = st.file_uploader("Choose a Excel file", type="xlsx")

if file is not None:
    df = pd.read_excel(file)
    st.dataframe(df)

X = df.values

ls_iner = []
for k in range(1,10):
    model = KMeans(n_clusters=k)
    model.fit(X)
    ls_iner.append(model.inertia_)

ls_k = [1,2,3,4,5,6,7,8,9]

st.header("Elbow Method")

fig, ax = plt.subplots()
ax.plot(ls_k, ls_iner)
st.pyplot(fig)


k = st.sidebar.slider("Number of Clusters (k)", 1, 10, 1)

model = KMeans(n_clusters=k)
model.fit(X)
df['Cluster'] = model.labels_
c = model.cluster_centers_

st.dataframe(df)

fig2, ax2 = plt.subplots()
ax2.scatter(X[:,0], X[:,1])
for k in range(k):
    ax2.scatter(c[k,0], c[k,1])
st.pyplot(fig2)