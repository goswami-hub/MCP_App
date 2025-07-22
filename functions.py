#from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula
from statsmodels.stats.proportion import proportions_ztest
from stqdm import stqdm
from scipy import stats
from scipy.stats import norm, multivariate_normal
import numpy as np
import networkx as nx
import streamlit as st
import pickle
import base64
import matplotlib.pyplot as plt


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def gaussian_copula_sampling(n_samples, correlation_matrix, marginal_dists=None):
    """
    Generates samples from a Gaussian Copula.

    Args:
        n_samples (int): Number of samples to generate.
        correlation_matrix (np.ndarray): The correlation matrix for the Gaussian copula.
        marginal_dists (list, optional): A list of scipy.stats distributions 
                                          for the desired marginals. 
                                          If None, standard normal marginals are used.

    Returns:
        np.ndarray: An array of samples from the Gaussian Copula.
    """
    n_variables = correlation_matrix.shape[0]

    # 1. Simulate from multivariate standard normal
    Z = multivariate_normal.rvs(mean=np.zeros(n_variables), cov=correlation_matrix, size=n_samples)

    # 2. Convert to uniform margins using standard normal CDF
    U = norm.cdf(Z)

    # 3. Apply inverse CDF of desired marginal distributions (if specified)
    if marginal_dists:
        if len(marginal_dists) != n_variables:
            raise ValueError("Number of marginal distributions must match number of variables.")
        X = np.zeros_like(U)
        for i in range(n_variables):
            X[:, i] = marginal_dists[i].ppf(U[:, i])
        return X

def TrialDataSimOneArm(n_sim, n_ss, n_hypo, dist_type, normal_param, binom_param, corr_mat):
  """
  Generates random samples from multivariate normal distribustion for each arm
  Inputs:
  n_sim: Number of studies to be simulated type:integer
  n_ss: Sample size under each hypotheses type: integer
  n_hypo: Number of hypotheses type:integer
  dist_type: Distribution types of each hypothesis type: List 'Continuous': normal , 'Binary':binomial
  e.g['Continuous','Continuous','Binary','Continuous','Binary',..] => 1st, 2nd , 4th endpoints are normal and 3rd and 5th are binomial
  If there is no normal distribution in your marginal input should be=[]

  normal_param: Parameters for marginal normal distributions type: list of tuples
  e.g[(mu1,sd1), (mu2,sd2),(mu3,sd3), ....] => This order is consistent with normal distribution representation in dist_type
  i.e. parameters for 1st , 2nd and 4th endpoint
  If there is no normal distribution in your marginal input should be=[]

  binom_param: Parameters for marginal binomial distributions type: list event rates
  e.g [p1,p2,..] =>This order is consistent with normal distribution representation in dist_type
  i.e. 3rd and 5th endpoint
  corr: Correlation matrix of order n_hypo*n_hypo type: 2D array
  Order should be identical to dist_type
  Outputs:
  Simulated trial data type: A 3D array of shape nsim*n_ss *n_hypo
  Order of endpoints are identical to dist_type
  """
  marginal_dist=[]*n_hypo

  for i in range(n_hypo):
    if dist_type[i]=='Continuous':
      x=stats.norm(loc=normal_param[0][0],scale=normal_param[0][1])
      marginal_dist.append(x)
      normal_param.pop(0)
    elif dist_type[i]=='Binary':
      x=stats.bernoulli(binom_param[0])
      marginal_dist.append(x)
      binom_param.pop(0)
    else:
      print("Error:Marginal types should be Continuous (Normal) or Binary (binomial)")
      break

  #copula= GaussianCopula(corr=corr_mat, k_dim=n_hypo)
  #joint_dist= CopulaDistribution(copula=copula, marginals=marginal_dist)

  TrialData=np.empty(shape=(n_sim,n_ss,n_hypo))
  for i in range(n_sim):
    #samples= joint_dist.rvs(n_ss)
    samples=  gaussian_copula_sampling(n_ss, correlation_matrix=corr_mat, marginal_distributions=marginal_dist)
    TrialData[i,:,:]=samples

  return TrialData




def SimPvalues(n_sim, n_hypo, dist_type,TrialDataTrt, TrialDataPlc, direction):
  '''
  Inputs:
  n_sim: Number of studies to be simulated type:integer
  n_hypo: Number of hypotheses type:integer
  dist_type: Distribution types of each hypothesis type: List 'n': normal , 'b':binomial
  e.g['Continuous','Continuous','Binary','Continuous','Binary',..] => 1st, 2nd , 4th endpoints are normal and 3rd and 5th are binomial
  If there is no normal (or binomial) distribution in your marginal input should be=[]
  TrialDataTrt: Simulated data for treatment arm  Type: 2D numpy array , each array representes one endpoint
  TrialDataPlc: Simulated data for tplacebo arm  Type: 2D numpy array , each array representes one endpoint
  direction: Direction of test : List or 1D numpy array , e.g['Higher','Lower','Higher','Both',..]

  Output:
  One sided p-value for superiority of treatment over placebo type: 1D array of length as number of hypothesis
  type: 1D array of legth n_hypo
  '''
  if len(dist_type)== n_hypo:

    p_values=np.empty(shape=(n_sim,n_hypo), dtype=float)

    for sim in range(n_sim):

      for i in range(len(dist_type)):

        if dist_type[i]=='Continuous':
          if direction[i]=='Higher':
            alt='greater'
          elif direction[i]=='Lower':
            alt='less'
          else:
            alt='two-sided'

          p_value= stats.ttest_ind(TrialDataTrt[sim,:,i], TrialDataPlc[sim,:,i],equal_var=True, alternative=alt)[1]

        elif dist_type[i]=='Binary':

          if direction[i]=='Higher':
            alt='larger'
          elif direction[i]=='Lower':
            alt='smaller'
          else:
            alt='two-sided'

          trt_succ= int(sum(TrialDataTrt[sim,:,i]))
          plc_succ= int(sum(TrialDataPlc[sim,:,i]))
          p_value= proportions_ztest(count=[trt_succ, plc_succ],nobs=[TrialDataTrt.shape[1],TrialDataPlc.shape[1]], prop_var=False, alternative=alt)[1]


        else:

          print("Error:Marginal types should be Continuous (Normal) or Binary (binomial)")
          break

        p_values[sim,i]=p_value

  else:

    print("Error: Number of endpoints mentioned through dist_type should be same as number of hypotheses")

  return p_values


def var_alpha_generation(alpha_var, nhypo, fixed_sum,w_count):
    '''
    Input:
    alpha_var: A data frame with one row indicating initial alpha allocation to all hypotheses of interests
    nhypo: Number of hypotheses
    fixed_sum: Sum of fixed numeric entries in alpha initial i.e. alpha_var
    w_count: Number of variable entries in alpha initial i.e alpha_var
    '''
    #Initiate a blank array
    arr=np.empty(nhypo)
    # Generate random values for variable elements such that their sum is (1 - fixed_sum)
    random_values = np.random.rand(w_count)
    random_values *= (1 - fixed_sum) / random_values.sum()

    # Replace w values with the generated random values
    random_index=0
    for i in range(alpha_var.shape[1]):
        if 'w' in str(alpha_var.iloc[0,i]):
            arr[i]=random_values[random_index]
            random_index += 1
        else:
            arr[i]=alpha_var.iloc[0,i]
 

    return arr


def normalize_random_corr_mat(transition_df,remaining_sum, w_count):
    '''
    Input:
    transition_df: Transition Matrix in a data frame
    remining_sum: List or 1D array - (1- sum of fixed elements) in each row
    w_count: Number of variable entries in each row

    Output:
    Transition matrix - 2D numpy array 
    Each variable entry is replaced by a random number between 0-1 , keeping each row sum as 1
    '''
    temp_df=np.empty(transition_df.shape)
    for i in range(transition_df.shape[0]):
            
        # Generate random values for elements containing 'w' such that their sum equals remaining_sum
        random_values = np.random.rand(w_count[i])
        random_values = random_values / random_values.sum() * remaining_sum[i]
        

        random_index = 0
        for j in range(transition_df.shape[1]):
            if 'w' in str(transition_df.iloc[i, j]):
                temp_df[i, j] = random_values[random_index]
                random_index += 1
            else:
                temp_df[i,j] = transition_df.iloc[i,j]
    return temp_df


def gMCP(n_hypo, alpha_w, transition_w, p, alpha):
  """
  Graph Based multiple comparison
  Inputs:
  n_hypo: Number of Hypotheses to be tested, type: Integer
  alpha_w: Weights for alpha assignment, type:1D array of length as n_hypo
  transition_w: Transition matrix, type: 2D array of shape n_hypo*n_hypo
  p: 1D array of p-values corresponding to each hypothesis
  alpha: Level of significance type:fraction between 0 to 1

  Output:
  Decision from graph as specified by testing strategy(a,w).
  type: A 1D array of 0 and 1 of length n_hypo, 1: rejected 0: failed to reject.
  Orders of 0s and 1s are relative to the hypothesis as represented in p-value input (p)
  """
  alpha_w= np.asarray(alpha_w)*alpha
  h = np.zeros(n_hypo)

  crit = 0

  while crit==0:
    test = (p < alpha_w )
    if sum(test > 0):
      rej = np.argmax(test)
      h[rej] = 1
      w_new = np.zeros(shape=(n_hypo,n_hypo))
      for i in range(n_hypo):
        alpha_w[i] += alpha_w[rej]*transition_w[rej,i]
        if (transition_w[i,rej]*transition_w[rej,i])<1:
          for j in range(n_hypo):
            w_new[i,j] = (transition_w[i,j] + transition_w[i,rej]*transition_w[rej,j])/(1- transition_w[i,rej]*transition_w[rej,i])
        w_new[i,i] = 0
      transition_w = w_new
      transition_w[rej,:] = np.zeros(n_hypo)
      transition_w[:,rej] = np.zeros(n_hypo)
      alpha_w[rej] = 0
    else:
      crit = 1

  return h

def SimGainLoss(n_hypo, n_graph, n_sim, alpha_w, transition_w, sim_pvalues, gain_values, alpha=0.025):
  '''
  Calculate Gain and Loss value for simulate p-values under each graph
  Inputs:
  n_hypo: Number of hypothesis type:integer
  n_graph: Number of graphs to be simulated type:integer
  n_sim: Number of studies simulated under each graph type:integer
  alpha_w: Alpha weights for n_graph many designs type: 2D array , each row for one graph/design
  transition_w: Collection of n_graphs many transition matrices type: 3D array
  e.g. W[i,:,:] represents ith transition matrix
  sim_pvalues: Simulated p values for n_sim studies
  gain_values: Gain vector type: 1D array , 1 for each hypothesis
  alpha: Type I error for graph testing type:float, fixed to 0.025 for one sided testing

  Outputs:
  gain_sim: Expected gain values for each graph type: 1D array
  loss_sim: Expected loss values for each graph type: 1D array
  power_sim: Marginal power for each graph type: 2D array
  TrueGraph: Whether a graph is connected and exhausts all alpha type: 2D array
  alpha_w: Input for alpha weights type: 2D array same as input
  transition_w: Input for transition weights type: 3D array same as input

  '''

  gain_sim= np.empty(n_graph, dtype=float)
  power_sim=np.empty(shape=(n_graph,n_hypo), dtype=float)

  for i in stqdm(range(n_graph)):

    graph_reject=np.empty(shape=(n_sim,n_hypo), dtype=int)

    for sim in range(n_sim):
      rej= gMCP(n_hypo=n_hypo, alpha_w=alpha_w[i,:], transition_w=transition_w[i,:,:], p=sim_pvalues[sim,:], alpha=alpha)
      graph_reject[sim,:]=rej

    gain= np.multiply(graph_reject,gain_values).sum(axis=1).mean(dtype=float)
 
    marginal_power=graph_reject.mean(axis=0,dtype=float)

    gain_sim[i]=gain

    power_sim[i,:]= marginal_power

  return gain_sim, power_sim, alpha_w, transition_w


# Functions for NN module

def select_initializer(option):
            if option=='Random Normal':
                value=keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
            elif option=='Random Uniform':
                value=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
            elif option=='zeros':
                value=keras.initializers.Zeros()
            elif option=='Xavier Uniform':
                value=keras.initializers.GlorotUniform()
            else:
                value==keras.initializers.GlorotNormal()
            return value

@st.cache_data()
def build_model_1L(nparam,h_act, h_kern_init, h_bias_init,h1_nodes):
    model=keras.models.Sequential(
        [
        keras.layers.Dense(input_dim=nparam, 
              units=h1_nodes, 
              activation=h_act,
              kernel_initializer=select_initializer(h_kern_init),
              bias_initializer=select_initializer(h_bias_init)),
        keras.layers.Dense(1, activation='relu')
        ]
    )

    return model
@st.cache_data()
def build_model_2L(nparam,h_act, h_kern_init, h_bias_init,h1_nodes,h2_nodes):
    model=keras.models.Sequential(
        [
        keras.layers.Dense(input_dim=nparam, 
                           units=h1_nodes, 
                           activation=h_act,
                           kernel_initializer=select_initializer(h_kern_init),
                           bias_initializer=select_initializer(h_bias_init)),
        keras.layers.Dense(units=h2_nodes, 
                           activation=h_act,
                           kernel_initializer=select_initializer(h_kern_init),
                           bias_initializer=select_initializer(h_bias_init)),
        keras.layers.Dense(1,activation='relu')
        ]
    )

    return model
@st.cache_data()
def load_data(data, nhypo, test_prop):
        # Drop redundant columns
        off_diag = ['w'+ str(x) + str(x) for x in range(nhypo)]
        marg_powers =['H'+str(x)+'_power' for x in range(nhypo)]

        drop_cols= off_diag + marg_powers

        data = data.drop(drop_cols, axis=1)
        # Drop columns with zero variance
        data=data.loc[:, data.var() != 0]

        min_gain = data['gain'].min()
        max_gain = data['gain']. max()

        data['gain_scaled'] = (data['gain'] - min_gain) / (max_gain - min_gain)

        data['gain_scaled_alt']=data['gain_scaled']*0.4 + 0.3

        data_Y = data['gain_scaled_alt']

        
        data_X=data.drop(['gain','gain_scaled'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = test_prop/100)

        return X_train, X_test, y_train, y_test

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def optimal_graph_draw(alpha_weights, transition_weights, hypotheses_names, 
                       layout_seed=3, fig_height=8, fig_width=8, arrow_size=10,
                       node_size=400, node_font_size=10, node_font='sans-serif',node_color='blue',node_style="o",
                       edge_font_color='red',edge_font_size=8, edge_font_position=0.45, edge_font='sans-serif',
                       arrow_arc='arc3, rad = 0.2', arrow_gap=3):
    """"
    INPUT:
    alpha_weights: 1D numpy array of order K
    transition_weights: 2D numpy array of order K X K
    hypotheses_names: Names of hypotheses in a list of strings of length K
    OUTPUT:
    Graph
    """
    n_hypo=len(alpha_weights)
    node_names= np.array(range(n_hypo))

    node_labels=[names+'\n'+ str(weights) for names, weights in zip(hypotheses_names, alpha_weights)]
    node_dict=dict(list(enumerate(node_labels)))

    fig=plt.figure(figsize=(fig_height, fig_width))

    G=nx.DiGraph()

    for i in range(n_hypo):
        for j in range(n_hypo):
            if transition_weights[i][j]> 0:
                G.add_edge(node_names[i], node_names[j], weight=transition_weights[i][j])
    
    pos=nx.spring_layout(G, seed=layout_seed)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, node_shape=node_style)
    nx.draw_networkx_labels(G, pos, font_size=node_font_size, font_family=node_font, labels=node_dict)

    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos,edgelist=straight_edges, arrowsize= arrow_size)

    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle=arrow_arc, arrowsize=arrow_size)
    edge_labels_curved=dict([((u, v,), str(transition_weights[u][v])+ '\n'*arrow_gap  + str(transition_weights[v][u]))
                for (u, v) in curved_edges if pos[u][0]> pos[v][0]])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_curved, label_pos=edge_font_position,
                                 font_color=edge_font_color, font_size=edge_font_size, font_family=edge_font, bbox={'alpha':0})

    edge_labels_straight=dict([((u, v,), f'{transition_weights[u][v]}')
                for (u, v) in straight_edges ])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_straight, label_pos=edge_font_position,
				font_color=edge_font_color, font_size=edge_font_size, font_family=edge_font)
    return st.pyplot(fig)

