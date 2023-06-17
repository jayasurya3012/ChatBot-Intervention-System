import streamlit as st
from streamlit_chat import message 
import os
from gtts import gTTS
import time
from transformers import pipeline,Conversation
import numpy as np
import datetime
from PIL import Image
from pytesseract import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r"--oem 3 --psm 11 -c tessedit_char_whitelist= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '"
path = ['C:/Users/jayas/Downloads/post.png' , 'C:/Users/jayas/Downloads/post3.png' , 'C:/Users/jayas/Downloads/post6.png' , 
        'C:/Users/jayas/Downloads/post1_1.png' , 'C:/Users/jayas/Downloads/post1_2.png' ,
        'C:/Users/jayas/Downloads/post1_3.png' , 'C:/Users/jayas/Downloads/post1_4.png' ,
        'C:/Users/jayas/Downloads/post1_5.png' , 'C:/Users/jayas/Downloads/post_1_6.png' ,
        'C:/Users/jayas/Downloads/post_1_7.png']

with open('reply.txt') as f:
    contents = f.read()
contents = contents.replace("\n"," ")
context = contents.replace("A: ","")
contents = contents.split('A:')

st.set_page_config(
    page_title="Chatbot Intervention System - Demo",
    page_icon=":robot:"
)

st.balloons()

st.title ("Robert to your service...")
st.warning("Its been detected that you have offen visited/liked some negative social media posts, it's recommended to interact with our chatbot for any assistance if there is any issue with your mental health.")
message("Hello I am Robert the AI, How can I help you?")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'temp' not in st.session_state:
        st.session_state.temp = ""
        
def bot(text):
    nlp = pipeline("conversational", model="microsoft/DialoGPT-medium")
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    model = SentimentIntensityAnalyzer()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if any(i in text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        
        # if user wishes to exit the chatbot
    elif any(i in text for i in ["exit","close"]):
        res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
        ex=False
          
        ## conversation
    else:   
            # if user needs some professional help or family intervention
        if any(i in text for i in ["talk with someone","I need help","phone number","mobile number","helpline"]):
            res = np.random.choice(["Let me help you, call '9876543210' and they will try to help you as much as possible",
                                        "Have a talk with your friend - 9753086421"])
        else:
                # 'tone' is used to find the emotion of the text sent by the user 
                # if it is -ve then we go to the question answering model
                # An answer is generated from the query and context which is given below,  
                # if there is any sentence in the contents list which contains this answer, we select that sentence as the reply
            tone = model.polarity_scores(text)
            if tone['compound'] < 0:
                reply = qa_model(question = text , context = context)
                wrd = reply['answer'] 
                for sentence in contents:
                    if wrd in sentence:
                        res = sentence
                        break
                # else we go to the nlp model for general chating
            else:
                res = nlp(Conversation(text), pad_token_id=50256)
                res = str(res)
                res = res[res.find("bot >> ")+6:].strip()
                        
    return res

def clear_text():
    
    st.session_state['temp'] = st.session_state["text"]
    st.session_state["text"] = ""

user_input = st.text_input("Message", key="text",on_change=clear_text) 
st.caption("Type 'close' or 'exit to exit the chatbot")
user_input = st.session_state['temp']

if user_input:
    output = bot(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
