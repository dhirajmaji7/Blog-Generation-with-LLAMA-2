import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function to get response from LLAMA 2 model
def getLLamaResponse(input_text, num_words, blog_style):
    ## LLama2 model
    llm = CTransformers(model='./llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 512,
                                'temperature': 0.01})
    
    ## Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text} 
        within {num_words} words.
            """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_words"],
                            template=template)

    ## Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_words=num_words))
    print(response)
    return response


## App Page using streamlit
st.set_page_config(page_title='Blog Generation',
                   page_icon=':writing_hand:',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(to right, #001100, #556b2f);
    }
</style>
""", unsafe_allow_html=True)

st.header("Generate Blogs with LLAMA 2 :writing_hand:")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    num_words = st.text_input("Number of Words")
with col2:
    blog_style = st.selectbox("Writing the blog for", ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final Response from LLama 2
if submit:
    st.write(getLLamaResponse(input_text, num_words, blog_style))