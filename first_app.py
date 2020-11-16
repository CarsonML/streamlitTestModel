import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import zipfile
import pickle

with zipfile.ZipFile("model.pkl.zip","r") as zip_ref:
    zip_ref.extractall("./")
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
st.title('Language Classification') 
txt = st.text_area('')
docs = [txt]
newText = tfidf[0].transform(docs).todense()
print(type(newText))
new_data = model.predict(newText)
    # c = make_pipeline(tfidf, model)
    # class_names = ['700','705','706']
    # explainer = LimeTextExplainer(class_names=class_names)
    # exp = explainer.explain_instance(txt, c.predict_proba, num_features=6, top_labels=1)
    # # Display explainer HTML object
    # components.html(exp.as_html(), height=800)

st.title(new_data)