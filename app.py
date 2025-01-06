# Complete Application

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLAMA 2 Model
def getLLamaResponse(input_text, no_words, blog_style):
    ## Call the LLAMA 2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,'temperature': 0.01})

    ## Prompt Template
    template = """
            Write a blog for {blog_style} job profile for a topic {input_text}
            within {no_words} words.
                """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

    ## Generate the ressponse from the LLama 2 model
    response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

## User Input
input_text = st.text_input("Enter the Blog Topic")

## Create 2 more column foe additional 2 fields
## Column 1 - specify number of words you want for that Blog
## Column 2 - Detail for whom we are creating a Blog

col1,col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of Words')

with col2:
    blog_style = st.selectbox('Writing the Blog for?',
                              ('Researchers','Data Scientiest','Common People'),
                              index=0) #Dropdown with options

submit = st.button("Generate")

# Final Response
if submit:
    st.write(getLLamaResponse(input_text,no_words,blog_style))


