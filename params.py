import streamlit as st

import pandas as pd
import numpy as np

import functions

import time
import base64





def clicked(button):
    st.session_state.clicked[button] = True

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False,2:False, 3:False, 4:False,5:False, 6:False}

@st.fragment
def trt_arm_sim_display(arm_input,endpoint_input,armdata_input,n_sim):
    select_trt=st.selectbox("Treatment Arm", options=arm_input)

    simID=st.number_input("Select Simulation ID", min_value=0, max_value=n_sim)

    temp_df= pd.DataFrame(armdata_input[select_trt][simID,:,:],columns=endpoint_input)
    return st.dataframe(temp_df.head())

@st.fragment
def sim_pval_display(n_sim,SimPdata, hypo_names):
    pr= st.number_input("Simulated P values: Select simulation ID", min_value=1, max_value=n_sim-5, key="dev2")
    return st.dataframe(pd.DataFrame(SimPdata,columns=hypo_names).iloc[pr:pr+5])

@st.fragment
def alpha_display(n_graph,gen_alpha):
    ralpha=st.number_input("Select a random alpha allocation", min_value=0, max_value=n_graph,key="dev3")
    return st.write(gen_alpha[ralpha])

@st.fragment
def transition_display(n_graph, transition_gen):
    rg=st.number_input("Select a random graph",min_value=0, max_value=n_graph, key="dev4")
    return st.write(transition_gen[rg,:,:])
@st.cache_data
def csv_downloader(data,timestr):
    csvfile= data.to_csv()
    b64= base64.b64encode(csvfile.encode()).decode()
    new_filename="data_{}_.csv".format(timestr)
    st.markdown("### Download Simulated Data ###")
    href= f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here: Download is Ready</a>'
    st.markdown(href, unsafe_allow_html=True)





def run_params():

    dev_mode=st.checkbox("Developer Mode")
    if dev_mode:
        st.markdown('''
                    :orange[Developer mode clutters the app.]
                    ''')

    with st.expander("Treatment Arm, Randomization (RR) and Sample Size"):

        arms=st.number_input("Number of Treatment Arms", 2, 4)

        basic_input= pd.DataFrame({"Arms":["Name"]*arms,"SS":[0]*arms})
            

        st.text("Arrange the treatment arms in Descending order of doses")
        basic_input_edited= st.data_editor(basic_input,
                                        column_config={"Arms":st.column_config.TextColumn("Arms", max_chars=8, validate=r"^[A-Za-z][A-Za-z0-9]"),
                                                        "SS": st.column_config.NumberColumn("Sample Size",min_value=5, max_value=1000, step=1)},
                                                        hide_index=True)

    with st.expander("Endpoints"):
        
        endpoints=st.number_input("Number of Endpoints", 1, 20)
            
        endpoint_list=['E'+str(i) for i in range(1,endpoints+1)]

        endpoint_df = pd.DataFrame({"Endpoints":endpoint_list,"Description":[" "]*endpoints, "Type":[" "]*endpoints,"Direction":[""]*endpoints})


        endpoint_df_edited= st.data_editor(endpoint_df,disabled=["Endpoints"],
                                           column_config={"Description":st.column_config.TextColumn("Endpoint Description", max_chars=32),
                                                          "Type":st.column_config.SelectboxColumn("Endpoint Type",width="medium", options=["Binary","Continuous"]),
                                                          "Direction":st.column_config.SelectboxColumn("Direction of Improvement", width="medium",options=["Higher","Lower","Both"])},
                                                                    hide_index=True)

    st.button("Submit 1",on_click=clicked,args=[1])

    if st.session_state.clicked[1]:

        with st.expander("Correlation of Endpoints"):
        
            corr_df=st.data_editor(pd.DataFrame(np.diag(np.full(endpoints,1.00)),columns=endpoint_list),hide_index=True)

            corr_df_array= corr_df.to_numpy()

            Err_msg="No Error"

            if (abs(corr_df)>1).sum().sum()> 0:
                Err_msg="All entries must be within -1 to 1"
            
            if not np.allclose(corr_df_array, corr_df_array.T, rtol=1e-08, atol=1e-08):
                Err_msg="Correlation matrix is not symmetric"

            if Err_msg=="No Error":
                pass
            else:
                st.error(Err_msg)

            eiv,_=np.linalg.eig(corr_df_array)
            
            if np.any(eiv<=0):
                st.error("Correlation matrix is not positive definite")
            else:
                pass

        
        st.button("Submit 2",on_click=clicked,args=[2])
        
        if st.session_state.clicked[2]:

            with st.expander("Response Parameters"):

                st.write("Binary Endpoints (Event Rates)")
                df_binary= endpoint_df_edited[endpoint_df_edited["Type"]=="Binary"]
                if df_binary.shape[0] > 0:
                    binary_data= pd.DataFrame(np.zeros((df_binary.shape[0],arms)),columns=basic_input_edited["Arms"])

                    #st.dataframe(df_binary_edited)
                    df_binary_edited=st.data_editor(pd.concat([df_binary.reset_index(drop=True),binary_data.reset_index(drop=True)],axis=1), 
                                            disabled=endpoint_df_edited.columns, hide_index=True)
                    if (df_binary_edited.loc[:,binary_data.columns]>1).sum().sum()> 0:
                        st.error("All entries must between 0 to 1")
                else:
                    st.write("No binary endpoint")
                
                st.write("Continuous Endpoints: Mean , SD")
                df_cont= endpoint_df_edited[endpoint_df_edited["Type"]=="Continuous"]

                if df_cont.shape[0]>0:
                    mnsd_list=[[Arms+"_Mean", Arms + "_SD"] for Arms in basic_input_edited["Arms"]]
                    mnsd_names=[e for items in mnsd_list for e in items ]
                    cont_data=pd.DataFrame(np.zeros((df_cont.shape[0],arms*2)),columns=mnsd_names)
                    df_cont_edited=st.data_editor(pd.concat([df_cont.reset_index(drop=True),cont_data.reset_index(drop=True)],axis=1), 
                                                    disabled=endpoint_df_edited.columns, hide_index=True, use_container_width=True)
                else:
                    st.write("No continuous endpoint")
            
            st.button("Submit 3",on_click=clicked,args=[3])

            if st.session_state.clicked[3]:

                with st.expander("Hypotheses and Gains"):
                    h_id=['H'+str(dose) + str(ep) for dose in range(1,arms) for ep in range(1,endpoints+1)]
                    h_names=[str(ep)+' :'+ dose + ' vs Placebo' for dose in basic_input_edited["Arms"][:-1] for ep in endpoint_list]
                    #h_dir=[""]*len(h_id)
                    h_gains=[0]*len(h_id)

                    hypo_df= pd.DataFrame(list(zip(h_id, h_names,h_gains)),columns=["ID","Description", "Gain"])
                    # st.dataframe(hypo_df)
                    hypo_df_edited=st.data_editor(hypo_df,hide_index=True, num_rows="dynamic",
                                                column_config={"Gain":st.column_config.NumberColumn("Gain (%)",min_value=1, max_value=99, step=0.5)})
                    if sum(hypo_df_edited["Gain"]) !=100:
                        st.warning("Sum of gain values should be 100!")

                st.button("Submit 4",on_click=clicked,args=[4])

                if st.session_state.clicked[4]:

                    with st.expander("Multiplicty Graph: Design Plan"):
                        st.markdown("**Level of Significance for the Study**")
                        alpha=st.number_input("Alpha",min_value=0.0005,max_value=0.2, format="%0.4f")

                        st.markdown("**Initial alpha allocation**")
                        st.markdown('''
                                        :orange[Consult with your statsitician. No programmatical check is implemented in this section.]
                                    
                                        Follow the instructions:
                                        - Each elemnt should be in between 0 to 1
                                        - sum of the elements should 1
                                    ''')
                        alpha_init=pd.DataFrame(index=[0],columns=hypo_df_edited["ID"])
                        alpha_init_edited=st.data_editor(alpha_init, hide_index=True)

                        st.markdown("**Initial Transition Matrix**")
                        st.markdown('''
                                    Follow the instructions:     
                                    - All the diagonal entries should be 0 
                                    - Each elemet should be in between 0 to 1
                                    - Sum of the elements in each row should be 1
                                    
                                    ''')

                        
                        transition_init=pd.DataFrame(index=hypo_df_edited["ID"], columns=hypo_df_edited["ID"])
                        for ent in transition_init.columns:
                            transition_init[ent][ent]=str(0)
                        transition_init_edited=st.data_editor(transition_init,hide_index=False)

                    st.button("Submit 5",on_click=clicked,args=[5])

                    if st.session_state.clicked[5]:
                        
                        with st.expander("Simulation Parameters"):
                            #Simulation parameters
                            nsim=st.number_input("Number of Studies",min_value=10,max_value=10000,step=10)
                            ngraph= st.number_input("Number of Graphs", min_value=10, max_value=10000,step=10)

                            st.button("Submit 6",on_click=clicked,args=[6])

                            if st.session_state.clicked[6]:
                                #st.write(st.session_state)
                                with st.spinner("Simuation in progress...", show_time=True):

                                    #st.warning("Initiating study data simulation....")
                                    armdata={}

                                    #Generate Study Data

                                    # Generate Data For each arm
                                    
                                    for i in range(arms):
                                        trt=basic_input_edited["Arms"][i]
                                        
                                        #Collect Binary responses
                                        if df_binary.shape[0]>0:
                                            if trt in df_binary_edited.columns:
                                                rates=list(df_binary_edited[trt])
                                        else:
                                                rates=[]

                                        # Collect continuous responses
                                        if df_cont.shape[0]>0:
                                            if str(trt)+'_Mean' in df_cont_edited.columns:
                                                mn=list(df_cont_edited[str(trt)+'_Mean'])
                                            if str(trt)+'_SD' in df_cont_edited.columns:
                                                sd=list(df_cont_edited[str(trt)+'_SD'])

                                            mnsd=[e for e in zip(mn,sd)]
                                        else:
                                            mnsd=[]

                                        
                                        
                                        armdata[trt]=functions.TrialDataSimOneArm(n_sim=nsim,n_ss=basic_input_edited["SS"][i], n_hypo=endpoints,dist_type=endpoint_df_edited["Type"],
                                                                                normal_param=mnsd,binom_param=rates,corr_mat=corr_df)
                                    if dev_mode:
                                        trt_arm_sim_display(arm_input=basic_input_edited["Arms"],endpoint_input=endpoint_df_edited["Endpoints"],armdata_input=armdata,n_sim=nsim)

                                        
                                    
                                    #Generate p-values
                                    #Placebo data:
                                    plc_name=list(basic_input_edited["Arms"])[-1]
                                    SimP={}

                                    for i in range(arms-1):
                                        trt=basic_input_edited["Arms"][i]
                                        SimP[trt]=functions.SimPvalues(n_sim=nsim,n_hypo=endpoints, dist_type=endpoint_df_edited["Type"],
                                                                    TrialDataTrt=armdata[trt],TrialDataPlc=armdata[plc_name], direction=endpoint_df_edited["Direction"])

                                    #Collate all the p-values 
                                    SimPall=np.concatenate(([SimP[k] for k in SimP.keys()]), axis=1)

                                    if dev_mode:
                                        sim_pval_display(n_sim=nsim,SimPdata=SimPall, hypo_names=hypo_df_edited["ID"])
                                            

   
                                    

                                    # Generate random alpha allocation
                                    # Calculate the sum of fixed elements
                                    fixed_sum_alpha = sum([float(x) for x in alpha_init_edited.iloc[0] if 'w' not in str(x)])

                                    # Calculate the number of w values
                                    w_count_alpha=sum(['w' in str(x) for x in alpha_init_edited.iloc[0]])


                                    alpha_arr= np.empty((ngraph,hypo_df_edited.shape[0]))

                                    if w_count_alpha>0:
                                        for g in range(ngraph):
                                            alpha_arr[g]=functions.var_alpha_generation(alpha_init_edited, nhypo=hypo_df_edited.shape[0], 
                                                                                        fixed_sum=fixed_sum_alpha, w_count=w_count_alpha)
                                    else:
                                        alpha_arr=np.tile(alpha_init_edited.to_numpy().flatten(),(ngraph,1))
                                    
                                    if dev_mode:
                                        alpha_display(n_graph=ngraph,gen_alpha=alpha_arr)

                                    #Generate random transition matrices
                                    

                                    fixed_sum_trans=[]
                                    remaining_sum_trans=[]
                                    w_count_trans=[]
                                    for i in range(transition_init_edited.shape[0]):
                                        # Calculate the sum of fixed numeric values in the row
                                        fixed_sum_trans.append(sum([float(x) for x in transition_init_edited.iloc[i] if 'w' not in str(x)]))

                                                
                                        # Calculate the remaining sum needed to make the row sum to 1
                                        remaining_sum_trans.append(1 - fixed_sum_trans[i])
                                    
                                                
                                        # Count the number of elements containing 'w' in the row
                                        w_count_trans.append(sum(['w' in str(x) for x in transition_init_edited.iloc[i]]))
                                    
                                    transition_arr=np.empty((ngraph,hypo_df_edited.shape[0],hypo_df_edited.shape[0]))

                                    for g in range(ngraph):
                                        transition_arr[g]=functions.normalize_random_corr_mat(transition_df=transition_init_edited,remaining_sum=remaining_sum_trans,w_count=w_count_trans)
                                    
                                    if dev_mode:
                                        transition_display(n_graph=ngraph, transition_gen=transition_arr)


                                    st.success("Part 1: Simulation of study data and gMCP strategies has been successfully completed.")


                                    g_s, p_s, a_w, tr_w= functions.SimGainLoss(n_hypo=hypo_df_edited.shape[0], n_graph=ngraph, n_sim=nsim, 
                                                                                                          alpha_w=alpha_arr, transition_w=transition_arr, 
                                                                                                          sim_pvalues=SimPall, gain_values=hypo_df_edited["Gain"].to_numpy(), alpha=alpha)

                                    st.success("Part 2: Simulation of gain values has been successfully completed.")
                                    
                                    # Save data in a numpy array

                                    tr_w_2d= np.reshape(tr_w,(tr_w.shape[0], tr_w.shape[1]*tr_w.shape[2]))

                                    output= np.column_stack([np.concatenate((a_w, tr_w_2d,p_s), axis=1),g_s])

                                    

                                    #np.save('NEJSDSEx2_data.npy',output)
                                    # Names for alpha weights
                                    names_alpha= ['a'+ str(x) for x in range(hypo_df_edited.shape[0])]

                                    # Names for transition weights
                                    names_transition = ['w'+ str(x) + str(y) for x in range(hypo_df_edited.shape[0]) for y in range(hypo_df_edited.shape[0])]


                                    # names other
                                    names_other = ['H'+str(x)+'_power' for x in range(hypo_df_edited.shape[0])]+['gain']


                                    names_all = names_alpha +  names_transition + names_other 

                                    simgain_df= pd.DataFrame(output, columns= names_all)

                                    st.dataframe(simgain_df.head())

                                    timestr=time.strftime("%Y%m%d-%H%M%S")

                                    csv_downloader(simgain_df,timestr)


                                
                                    
                                
                                
