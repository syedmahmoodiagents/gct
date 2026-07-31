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
