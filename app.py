import streamlit as st
from fastai.vision.all import *



st.title('Guruhlovchi model\n Bird Insect Animal')
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('model.pkl')
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
