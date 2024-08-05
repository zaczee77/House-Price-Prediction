import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Sample data for demonstration
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 20, 30, 40, 50],
    'Prediction': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

st.title('House Prediction Model Visualization')

# Display the dataframe
st.write('### Input Data')
st.write(df)

# Plotting with matplotlib
st.write('### Matplotlib Plot')
fig, ax = plt.subplots()
ax.scatter(df['Feature1'], df['Prediction'])
ax.set_xlabel('Feature1')
ax.set_ylabel('Prediction')
st.pyplot(fig)

# Plotting with Plotly
st.write('### Plotly Plot')
fig = px.scatter(df, x='Feature1', y='Prediction', title='Feature1 vs Prediction')
st.plotly_chart(fig)
import streamlit as st

st.title('Hello Streamlit!')
st.write('This is a basic Streamlit app.')

