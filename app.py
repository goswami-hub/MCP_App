# Core package
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
#from st_aggrid import AgGrid, GridOptionsBuilder

from params import run_params
#from nn import ANN
#from optim import optimization





html_temp="""
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center;">ML MCP </h1>
            <h4 style="color:white;text-align:center">Optimization of Graph Based Multiple Comparison Procedure </h4>
            </div>
            """
def home_page():
    st.header("Welcome to Home Page")
    st.markdown("*Add Suitable Description and Training Resources*")
def input_page():
    st.header("Input data and Assumptions")
    run_params()

def ann_page():
        st.header("Function Approximation of Expected Gain Through ANN")
        #ANN()
        
        
def optim_page():
        st.header("Optimize ANN to Find Optimal Graph")
        optimization()


stc.html(html_temp)

home= st.Page(home_page, title="Home",icon=":material/home:")
input= st.Page(input_page, title="Input",icon=":material/123:")
#advanced= st.Page(control_page, title="Advanced",icon=":material/manage_accounts:")
ann=st.Page(ann_page,title="ANN", icon=":material/network_node:")
optim=st.Page(optim_page,title="Optimization",icon=":material/dvr:")

#info=st.Page(info_page, title="Information", incon=":material/help:")



pg= st.navigation({"Home":[home],"Input":[input], "Advanced":[ann,optim]})

pg.run()




