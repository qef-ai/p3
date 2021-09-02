#!/usr/bin/env python
# coding: utf-8

# ## Module 1 - Dogs: loglinear model for binary data
# 
# 
# **Background:** Solomon-Wynne in 1953 conducted an experiment on avoidance learning in dogs from traumatic experiences in past such as those from electric shocks.
# The apparatus of experiment holds a dog in a closed compartment with steel flooring, open on side with a small barrier for dog to jump over to the other side. A high-voltage electric shock is discharged into the steel floor intermittently to stimulate the dog; Thus the dog is effectively left with an option to either get the shock for that trial or jump over the barrier to other side & save himself. Several dogs were subjected to similar experiment for many consecutive trials.
# This picture elicits the appratus
# 
# <img src="data/avoidance_learning.png">
# 
# The elaborate details of the experiment can be found at
# http://www.appstate.edu/~steelekm/classes/psy5300/Documents/Solomon&Wynne%201953.pdf
# 
# The hypothesis is that most of the dogs learnt to avoid shocks by jumping over barrier to the other side after suffering the trauma of shock in previous trials. That inturn sustain dogs in future encounters with electric shocks.

# Since the experiment aims to study the avoidance learning in dogs from past traumatic experiences and reach a plausible model where dogs learn to avoid scenerios responsible for causing trauma, we describe the phenomenon using expression
# $$
# \pi_j   =   A^{xj} B^{j-xj}
# $$
# Where :
#    * $\pi_j$ is the probability of a dog getting shocked at trial $j$
#    * A & B both are random variables drawing values from Normal distribution
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
# In the subsequent 
# 
# The hypothesis is thus corroborated by Bayesian modelling and comprehensive analysis of dogs data available from Solomon-Wynne experiment in Pyro.
# 

# The data is analysed step by step in accordance with Bayesian workflow as described in "Bayesian Workflow", Prof. Andrew Gelman [http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf].
# 
# Import following dependencies.

# In[1]:


import os
import torch
import pyro
import random
import time
import numpy as np
import pandas as pd
import re
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict

from scipy import stats
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS


import plotly
import plotly.express as px
import plotly.figure_factory as ff

pyro.set_rng_seed(1)

# Uncomment following if pandas available with plotly backend

# pd.options.plotting.backend = "plotly"

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')


# ### 1. Model Specification: Dogs Model definition
# ________
# $$
# \pi_j   =   A^{xj} B^{j-xj}
# $$
# 
# We intend to find most probable values for parameters $\alpha$ & $\beta$ (dubbed as random variable A & B respectively) in the expression to compute likelihood ($\pi_j$) of dogs getting shocked.
# 
# **Generative model for resulting likelihood of shock:**
# 
# $\pi_j$  ~   $bern\ (\exp \ (\alpha.XAvoidance + \beta.XShocked)\ )$,  $prior\ \alpha$ ~ $N(0., 316.)$,  $\beta$ ~ $N(0., 316.)$
# 
# The above expression is used as a generalised linear model with log-link function in WinBugs implementation
# 
#   **BUGS model**
#   
# $\log\pi_j = \alpha\ x_j + \beta\ ( $j$-x_j )$
# 
#    **Here**
#    * $\log\pi_j$ is log probability of a dog getting shocked at trial $j$
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
#    *  $\alpha$ is the coefficient corresponding to number of success, $\beta$ is the coefficient corresponding to number of failures.
# 
#   
#   ____________________
# The same model when implemented in PyStan
#   
#   **Equivalent Stan model** 
#   
#       {
#   
#       alpha ~ normal(0.0, 316.2);
#   
#       beta  ~ normal(0.0, 316.2);
#   
#       for(dog in 1:Ndogs)
#   
#         for (trial in 2:Ntrials)  
# 
#           y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * 
#           xs[dog, trial]));
#       
#       }  
# 

# **Model implementation**
# 
# The model is defined using Pyro as per the expression of generative model for this dataset as follows

# In[2]:


# Dogs model with normal prior
def DogsModel(x_avoidance, x_shocked, y):
    """
    Input
    -------
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25)
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    
    Output
    --------
    Implements pystan model: {
              alpha ~ normal(0.0, 316.2);
              beta  ~ normal(0.0, 316.2);
              for(dog in 1:Ndogs)  
                for (trial in 2:Ntrials)  
                  y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}
    """
    alpha = pyro.sample("alpha", dist.Normal(0., 316.))
    beta = pyro.sample("beta", dist.Normal(0., 316))
    with pyro.plate("data"):
        pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)


# **Dogs data**
# 
# Following holds the Dogs data in the pystan modelling format

# In[3]:


dogs_data = {"Ndogs":30, 
             "Ntrials":25, 
             "Y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 
                  1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 
                  1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 
                  0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 
                  0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 
                  1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((30,25))}


# **Following processes target label `y` to obtain input data `x_avoidance` & `x_shocked` where:**
# * `x_avoidance` :  number of shock avoidances before current trial.
# * `x_shocked` :  number of shocks before current trial.

# In[4]:


def transform_data(Ndogs=30, Ntrials=25, Y= np.array([0, 0, 0, 0])):
    """
    Input
    -------
    Ndogs: Total number of Dogs i.e., 30
    Ntrials: Total number of Trials i.e., 25
    Y: Raw responses from data, example: np.array([0, 0, 0, 0])
    
    Outputs
    ---------
    xa: tensor holding avoidance count for all dogs & all trials
    xs: tensor holding shock count for all dogs & all trials
    y: tensor holding response observation for all dogs & all trials
    
    """
    y= np.zeros((Ndogs, Ntrials))
    xa= np.zeros((Ndogs, Ntrials))
    xs= np.zeros((Ndogs, Ntrials))

    for dog in range(Ndogs):
        for trial in range(1, Ntrials+1):
            xa[dog, trial-1]= np.sum(Y[dog, :trial-1]) #Number of successful avoidances uptill previous trial
            xs[dog, trial-1]= trial -1 - xa[dog, trial-1] #Number of shocks uptill previous trial
    for dog in range(Ndogs):
        for trial in range(Ntrials):
            y[dog, trial]= 1- Y[dog, trial]
    xa= torch.tensor(xa, dtype=torch.float)
    xs= torch.tensor(xs, dtype=torch.float)  
    y= torch.tensor(y, dtype=torch.float)

    return xa, xs, y


# **Here the py-stan format data (python dictionary) is passed to the function above, in order to preprocess it to tensor format required for pyro sampling**

# In[5]:


x_avoidance, x_shocked, y= transform_data(**dogs_data)
print("x_avoidance: %s, x_shocked: %s, y: %s"%(x_avoidance.shape, x_shocked.shape, y.shape))

print("\nSample x_avoidance: %s \n\nSample x_shocked: %s"%(x_avoidance[1], x_shocked[1]))


# ### 2. Prior predictive checking
# 
# These checks help to understand the implications of a prior distributions of underlying parameters (random variables) in the context of a given generative model by simulating from the model rather than observed data.
# 

# In[6]:


priors_list= [(pyro.sample("alpha", dist.Normal(0., 316.)).item(), 
               pyro.sample("beta", dist.Normal(0., 316.)).item()) for index in range(1100)]# Picking 1100 prior samples

prior_samples = {"alpha":list(map(lambda prior_pair:prior_pair[0], priors_list)), "beta":list(map(lambda prior_pair:prior_pair[1], priors_list))}


# Sampled output of prior values for alpha & beta is stored in `prior_samples` above, and is plotted on a KDE plot as follows:

# In[7]:


fig = ff.create_distplot(list(prior_samples.values()), list(prior_samples.keys()))
fig.update_layout(title="Prior distribution of parameters", xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Prior alpha Q(0.5) :%s | Prior beta Q(0.5) :%s"%(np.quantile(prior_samples["alpha"], 0.5), np.quantile(prior_samples["beta"], 0.5)))


# ### 3. Posterior estimation
# 
# In the parlance of probability theory, Posterior implies the probability of updated beliefs in regard to a quantity or parameter of interest, in the wake of given evidences and prior information.
# 
# $$Posterior = \frac {Likelihood x Prior}{Probability \ of Evidence}$$
# 
# 
#  
# For the parameters of interest $\alpha,\beta$ & evidence y; Posterior can be denoted as $P\ (\alpha,\beta\ /\ y)$.
# 
# 
# $$P\ (\alpha,\beta\ /\ y) = \frac {P\ (y /\ \alpha,\beta) P(\alpha,\beta)}{P(y)}$$
# 
# Posterior, $P\ (\alpha,\beta\ /\ y)$ in regard to this experiment is the likelihood of parameter values (i.e., Coefficient of instances avoided dubbed retention ability & Coefficient of instances shocked dubbed learning ability) given the observed instances `y` of getting shocked. Where $P(\alpha,\beta)$ is prior information/likelihood of parameter values.

# The following intakes a pyro model object with defined priors, input data and some configuration in regard to chain counts & chain length prior to launching a `MCMC NUTs sampler` and outputs MCMC chained samples in a python dictionary format.

# In[8]:


def get_hmc_n_chains(pyromodel, xa, xs, y, num_chains=4, base_count = 900):
    """
    Input
    -------
    pyromodel: Pyro model object with specific prior distribution
    xa: tensor holding avoidance count for all dogs & all trials
    xs: tensor holding shock count for all dogs & all trials
    y: tensor holding response observation for all dogs & all trials
    num_chains: Count of MCMC chains to launch, default 4
    base_count:Minimum count of samples in a MCMC chains , default 900
    
    Ouputs
    ---------
    hmc_sample_chains: a dictionary with chain names as keys & dictionary of parameter vs sampled values list as values 
    
    """
    hmc_sample_chains =defaultdict(dict)
    possible_samples_list= random.sample(list(np.arange(base_count, base_count+num_chains*100, 50)), num_chains)
    possible_burnin_list= random.sample(list(np.arange(100, 500, 50)), num_chains)

    t1= time.time()
    for idx, val in enumerate(list(zip(possible_samples_list, possible_burnin_list))):
        num_samples, burnin= val[0], val[1]
        nuts_kernel = NUTS(pyromodel)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin)
        mcmc.run(xa, xs, y)
        hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    print("\nTotal time: ", time.time()-t1)
    hmc_sample_chains= dict(hmc_sample_chains)
    return hmc_sample_chains


# In[9]:


hmc_sample_chains= get_hmc_n_chains(DogsModel, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# `hmc_sample_chains` holds sampled MCMC values as `{"Chain_0": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}, "Chain_1": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}. .}`

# ### 4. Diagnosing model fit
# 
# Model fit diagnosis consists of briefly obtaining core statistical values from sampled outputs and assess the convergence of various chains from the output, before moving onto inferencing or evaluating predictive power of model.

# Following plots **Parameter vs. Chain matrix** and optionally saves the dataframe.

# In[10]:


beta_chain_matrix_df = pd.DataFrame(hmc_sample_chains)
# beta_chain_matrix_df.to_csv("dogs_log_regression_hmc_sample_chains.csv", index=False)
beta_chain_matrix_df


# **Key statistic results as dataframe**
# 
# Following method maps the values of required statistics, given a list of statistic names

# In[11]:


all_metric_func_map = lambda metric, vals: {"mean":np.mean(vals), "std":np.std(vals), 
                                            "25%":np.quantile(vals, 0.25), 
                                            "50%":np.quantile(vals, 0.50), 
                                            "75%":np.quantile(vals, 0.75)}.get(metric)


# **Following outputs the summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, given a list of statistic names**

# In[12]:


key_metrics= ["mean", "std", "25%", "50%", "75%"]

summary_stats_df_= pd.DataFrame()
for metric in key_metrics:
    final_di = {}
    for column in beta_chain_matrix_df.columns:
        params_per_column_di = dict(beta_chain_matrix_df[column].apply(lambda x: all_metric_func_map(metric, x)))
        final_di[column]= params_per_column_di
    metric_df_= pd.DataFrame(final_di)
    metric_df_["parameter"]= metric
    summary_stats_df_= pd.concat([summary_stats_df_, metric_df_], axis=0)

summary_stats_df_.reset_index(inplace=True)
summary_stats_df_.rename(columns= {"index":"metric"}, inplace=True)
summary_stats_df_.set_index(["parameter", "metric"], inplace=True)
summary_stats_df_


# **Obtain 5 point Summary statistics (mean, Q1-Q4, Std, ) as tabular data per chain and save the dataframe.**
# 

# In[13]:


fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

fit_df.to_csv("data/dogs_classification_hmc_samples.csv", index=False)    
fit_df


# In[14]:


# Use/Uncomment following once the results from pyro sampling operation are saved offline
# fit_df= pd.read_csv("data/dogs_classification_hmc_samples.csv")

fit_df


# Following outputs the similar summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, **But in a slightly different format**, given a list of statistic names**

# In[15]:


summary_stats_df_2= pd.DataFrame()

for param in ["alpha", "beta"]:
    for name, groupdf in fit_df.groupby("chain"):
        groupdi = dict(groupdf[param].describe())

        values = dict(map(lambda key:(key, [groupdi.get(key)]), ['mean', 'std', '25%', '50%', '75%']))
        values.update({"parameter": param, "chain":name})
        summary_stats_df= pd.DataFrame(values)
        summary_stats_df_2= pd.concat([summary_stats_df_2, summary_stats_df], axis=0)
summary_stats_df_2.set_index(["parameter", "chain"], inplace=True)
summary_stats_df_2


# **Following plots sampled parameters values as Boxplots with `M parameters` side by side on x axis for each of the `N chains`**

# In[17]:


parameters= ["alpha", "beta"]# All parameters for given model
chains= fit_df["chain"].unique()# Number of chains sampled for given model


func_all_params_per_chain = lambda param, chain: (param, fit_df[fit_df["chain"]==chain][param].tolist())
func_all_chains_per_param = lambda chain, param: (f'{chain}', fit_df[param][fit_df["chain"]==chain].tolist())

di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, "chain_0"), parameters))
di_all_chains_per_param = dict(map(lambda chain: func_all_chains_per_param(chain, "beta"), chains))


def plot_parameters_for_n_chains(chains=["chain_0"], parameters=["beta0", "beta1", "beta2", "beta3", "sigma"], plotting_cap=[4, 5], plot_interactive=False):
    """
    Input
    --------
    chains: list of valid chain names, example - ["chain_0"].
    
    parameters: list of valid parameters names, example -["beta0", "beta1", "beta2", "beta3", "sigma"].
    
    plotting_cap: list of Cap on number of chains & Cap on number of parameters to plot, example- [4, 5] 
                  means cap the plotting of number of chains upto 4 & number of parameters upto 5 ONLY,
                  If at all the list size for Chains & parameters passed increases.
    
    plot_interactive: Flag for using Plotly if True, else Seaborn plots for False.
    
    
    output
    -------
    Plots box plots for each chain from list of chains with parameters on x axis.
    
    """
    try:
        chain_cap, param_cap = plotting_cap#
        assert len(chains)<=chain_cap, "Cannot plot Number of chains greater than %s!"%chain_cap
        assert len(parameters)<=param_cap, "Cannot plot Number of parameters greater than %s!"%param_cap
        
        for chain in chains:
            di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, chain), parameters))
            df_all_params_per_chain = pd.DataFrame(di_all_params_per_chain)
            if df_all_params_per_chain.empty:
#                 raise Exception("Invalid chain number in context of model!")
                print("Note: Chain number [%s] is Invalid in context of this model!"%chain)
                continue
            if plot_interactive:
                df_all_params_per_chain= df_all_params_per_chain.unstack().reset_index(level=0)
                df_all_params_per_chain.rename(columns={"level_0":"parameters", 0:"values"}, inplace=True)
                fig = px.box(df_all_params_per_chain, x="parameters", y="values")
                fig.update_layout(height=600, width=900, title_text=f'{chain}')
                fig.show()
            else:
                df_all_params_per_chain.plot.box()
                plt.title(f'{chain}')
    except Exception as error:
        if type(error) is AssertionError:
            print("Note: %s"%error)
            chains = np.random.choice(chains, chain_cap, replace=False)
            parameters=np.random.choice(parameters, param_cap, replace=False)
            plot_parameters_for_n_chains(chains, parameters)
        else: print("Error: %s"%error)


# **Pass the list of `M parameters` and list of `N chains`, with `plot_interactive` as `True or False` to choose between Plotly or Seaborn**

# In[18]:


# Use plot_interactive=False for Normal seaborn plots offline

plot_parameters_for_n_chains(chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=True)


# **Following plots the `joint distribution` of `pair of each parameter` sampled values for all chains**

# In[19]:


all_combination_params = list(itertools.combinations(parameters, 2))

for param_combo in all_combination_params:
    param1, param2= param_combo
    print("\nPyro -- %s"%(f'{param1} Vs. {param2}'))
    sns.jointplot(data=fit_df, x=param1, y=param2, hue= "chain")
    plt.title(f'{param1} Vs. {param2}')
    plt.show()
    


# **Following plots the `Pairplot distribution` of each parameter with every other parameter's sampled values**

# In[20]:


sns.pairplot(data=fit_df, hue= "chain");


# **Following intakes the list of parameters say `["alpha", "beta"]` and plots hexbins for each interaction pair for all possible combinations of parameters `alpha & beta`.**

# In[21]:


def hexbin_plot(x, y, x_label, y_label):
    """
    
    Input
    -------
    x: Pandas series or list of values to plot on x axis.
    y: Pandas series or list of values to plot on y axis.
    x_label: variable name x label. 
    y_label: variable name x label. 
    
    
    Output
    -------
    Plot Hexbin correlation density plots for given values.
    
    
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    min_x = min(list(x)+list(y)) - 0.1
    max_x = max(list(x)+list(y)) + 0.1
    ax.plot([min_x, max_x], [min_x, max_x])
    
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_x, max_x])
    
    ax.set_title('{} vs. {} correlation scatterplot'.format(x_label, y_label))
    hbin= ax.hexbin(x, y, gridsize=25, mincnt=1, cmap=plt.cm.Reds)
    cb = fig.colorbar(hbin, ax=ax)
    cb.set_label('occurence_density')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

def plot_interaction_hexbins(fit_df, parameters=["alpha", "beta"]):
    """
    
    Input
    -------
    fit_df: Pandas dataframe containing sampled values across columns with parameter names as column headers
    parameters: List of parameters for which all combination of hexbins are to be plotted, defaults to ["alpha", "beta"]
    
    
    Output
    -------
    Plots hexbin correlation density plots for each pair of parameter combination.
        
    """
    all_combination_params = list(itertools.combinations(parameters, 2))
    for param1, param2 in all_combination_params:#Plots interaction between each of two parameters
        hexbin_plot(fit_df[param1], fit_df[param2], param1, param2)


# **Here parameters `["alpha", "beta"]` are passed to plot all possible interaction pair Hexbin plots in between**

# In[22]:



plot_interaction_hexbins(fit_df, parameters=parameters)


# ### 5. Model evaluation: Posterior predictive checks
# 
# Posterior predictive checking helps examine the fit of a model to real data, as the parameter drawn for simulating conditions & regions of interests come from the posterior distribution.

# **Pick samples from one particular chain of HMC samples say `chain_3`**

# In[23]:


for chain, samples in hmc_sample_chains.items():
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))


# **Plot density for parameters from `chain_3` to visualise the spread of sample values from that chain**

# In[24]:


title= "parameter distribution for : %s"%(chain)
fig = ff.create_distplot(list(map(lambda x:x.numpy(), samples.values())), list(samples.keys()))
fig.update_layout(title=title, xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))


# **Plot density & contours for both parameters from `chain_3` to visualise the joint distribution & region of interest**

# In[25]:



fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

#Choosing samples from chain 3
chain_samples_df= fit_df[fit_df["chain"]==chain].copy()# chain is 'chain_3' 

alpha= chain_samples_df["alpha"].tolist()
beta= chain_samples_df["beta"].tolist()
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(alpha, beta, colorscale=colorscale, hist_color='rgb(255, 255, 150)', point_size=4, title= "alpha beta joint density plot")
fig.update_layout( xaxis_title="x (alpha)", yaxis_title="y (beta)")

fig.show()


# **Note:** The distribution of alpha values are significantly offset to the left from beta values, by almost 13 times; Thus for any given input observation of avoidances or shocked, the likelihood of getting shocked is more influenced by small measure of avoidance than by getting shocked.

# **Observations:**
# 
# **On observing the spread of alpha & beta values, the parameter beta being less negative & closer to zero can be interpreted as `learning ability`, i.e., the ability of dog to learn from shock experiences. The increase in number of shocks barely raises the probability of non-avoidance (value of 𝜋𝑗) with little amount. Unless the trials & shocks increase considerably large in progression, it doesn't mellow down well and mostly stays around 0.9.**
# 
# **Whereas its not the case with alpha, alpha is more negative & farthest from zero. It imparts a significant decline in non-avoidance (𝜋𝑗) even for few instances where dog avoids the shock; therefore alpha can be interpreted as `retention ability` i.e., the ability to retain the learning from previous shock experiences.**

# In[26]:


print(chain_samples_df["alpha"].describe(),"\n\n", chain_samples_df["beta"].describe())


# **From the contour plot above following region in posterior distribution seems highly plausible for parameters:**
# 1. For alpha, `-0.2 < alpha < -0.19`
# 2. For beta `-0.0075 < beta < -0.0055`
# 
# Following selects all the pairs of `alpha, beta` values between the range mentioned above.

# In[27]:


select_sample_df= chain_samples_df[(chain_samples_df["alpha"]<-0.19)&(chain_samples_df["alpha"]>-0.2)&(chain_samples_df["beta"]<-0.0075)&(chain_samples_df["beta"]<-0.0055)]

# print(select_sample_df.set_index(["alpha", "beta"]).index)
print("Count of alpha-beta pairs of interest, from mid region with high desnity in contour plot above (-0.2 < alpha < -0.19, -0.0075 < beta < -0.0055): ", select_sample_df.shape[0])

select_sample_df.head(3)


# **Picking a case of 3 trials with Y [0,1,1], i.e. Dog is shocked in 1st, Dogs avoids in 2nd & thereafter, effectively having an experience of 1 shock & 1 avoidance. `Considering all values of alpha & beta in range -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055`**

# In[28]:


Y_li= []
Y_val_to_param_dict= defaultdict(list)

# Value -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055
for rec in select_sample_df.iterrows():# for -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055
    a,b = float(rec[1]["alpha"]), float(rec[1]["beta"])
    res= round(np.exp(a+b), 4)
    Y_li.append(res)
    Y_val_to_param_dict[res].append((round(a,5),round(b,5)))# Sample-- {0.8047: [(-0.18269378, -0.034562342), (-0.18383412, -0.033494473)], 0.8027: [(-0.18709463, -0.03263992), (-0.18464606, -0.035114493)]}


# In above `Y_val_to_param` is a dictionary that holds value $\exp^{\alpha +\beta}$ as key and tuple of corresponding $(\alpha, \beta)$ as value.
# 
# The following plots the histogram of $\exp^{\alpha +\beta}$ values obtained as an interaction of selected $\alpha$ and $\beta$ values from region of interest.

# In[29]:


Y_for_select_sample_df = pd.DataFrame({"Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055": Y_li})
fig = px.histogram(Y_for_select_sample_df, x= "Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055")
title= "observed values distribution for params Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055"

fig.update_layout(title=title, xaxis_title="observed values", yaxis_title="count", legend_title="dogs")
fig.show()
print("Mean: %s | Median: %s"%(np.mean(Y_li), np.quantile(Y_li, 0.5)))
print("Sorted observed values: \n", sorted(Y_li))


# **For given experiment of 3 trials, from all the `Ys` with corresponding alpha-beta pairs of interest, pick 3  lower most values of `Y` for instance; Thus selecting its corresponding alpha-beta pairs**
# 
# **Note:** Can add multiple observed values from histogram for comparison.

# Corresponding to `lowest_obs` values of `Y`, obtain `select_pairs` as list of correspoding alpha, beta pairs from  `Y_val_to_param_dict`.

# In[30]:


lowest_obs = sorted(Y_li)[:3]#[0.8085, 0.8094, 0.8095]# Pick values from above histogram range or sorted list

selected_pairs= list(itertools.chain.from_iterable(map(lambda obs: Y_val_to_param_dict.get(obs), lowest_obs)))
selected_pairs


# **Following stores a dictionary of `observed y` values for pair of alpha-beta parameters**

# In[31]:


def get_obs_y_dict(select_pairs, x_a, x_s):
    """
    
    Input
    -------
    select_pairs: pairs of (alpha, beta) selected
    x_a: array holding avoidance count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
    x_s: array holding shock count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
    
    Output
    -------
    
    Outputs a dictionary with tuple of alpha, beta as key & observerd values of y corresponding to alpha, beta in key
    
    """
    y_dict = {}
    for alpha, beta in select_pairs:# pair of alpha, beta
        y_dict[(alpha, beta)] = torch.exp(alpha*x_a + beta* x_s)
    
    return y_dict


obs_y_dict= get_obs_y_dict(selected_pairs, x_avoidance, x_shocked)

print("Alpha-beta pair values as Keys to access corresponding array of inferred observations: \n", list(obs_y_dict.keys()))


# **Following plots scatterplots of `observed y` values for all 30 dogs for each alpha-beta pair of interest**

# In[32]:


def plot_observed_y_given_parameters(observations_list, selected_pairs_list, observed_y, chain, original_obs= []):
    """
    
    Input
    -------
    observations_list:list of observated 'y' values from simulated 3 trials experiment computed corresponding 
                      to selected pairs of (alpha, beta)
    selected_pairs_list: list of alpha, beta pair tuples, example :  [(-0.225, -0.01272), (-0.21844, -0.01442)]
    
    observed_y: dict holding observed values correspodning to pair of alpha, beta tuple as key, 
                example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])}
    chain: name of the chain from sampler
    original_obs: original observations/ labels from given data

    returns  plotly scatter plots with number of trials on X axis & corresponding probability of getting
    shocked for each pair of (alpha, beta) passed in 'selected_pairs_list'.
    
    Output
    --------
    Plots scatter plot of all observed values of y corresponding to each given pair of alpha, beta
    
    """
    obs_column_names = [f'Dog_{ind+1}'for ind in range(dogs_data["Ndogs"])]
    
    for record in zip(observations_list, selected_pairs_list):
        sim_y, select_pair = record
        print("\nFor simulated y value: %s & Selected pair: %s"%(sim_y, select_pair))

        obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=obs_column_names)
        if not original_obs is plot_observed_y_given_parameters.__defaults__[0]:
            original_obs_column_names = list(map(lambda name:f'*{name}', obs_column_names))
            
            original_obs_df= pd.DataFrame(original_obs.numpy().T, columns=original_obs_column_names)
            obs_y_df= pd.concat([obs_y_df, original_obs_df], axis=1)
            print("Note: Legend *Dog_X corresponds to 'y' i.e.,original observation values")
        
        obs_y_title= "Observed values distribution for all dogs given parameter %s from %s"%(select_pair, chain)
        fig = px.scatter(obs_y_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
        fig.show()


# **Also Optionally pass the `True observed y` values to `original_obs` argument for all 30 dogs to plot alongside the `observed y` from alpha-beta pairs of interest.**
# 
# **_Note_**: `True observed y` are marked with legends in format `*Dog_X`

# In[33]:


plot_observed_y_given_parameters(lowest_obs, selected_pairs, obs_y_dict, chain, original_obs= y)


# **Following plots a single scatterplots for comparison of `observed y` values for all alpha-beta pairs of interest from dense region in contourplot above, that is `-0.2 < alpha < -0.19`, `-0.0075 < beta < -0.0055`**
# 

# In[34]:


def compare_dogs_given_parameters(pairs_to_compare, observed_y, original_obs=[], alpha_by_beta_dict= {}):
    """
    
    Input
    --------
    
    pairs_to_compare: list of alpha, beta pair tuples to compare, 
                      example :  [(-0.225, -0.0127), (-0.218, -0.0144)]
    observed_y: dict holding observed values correspodning to pair of alpha,
                      beta tuple as key, example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])} 
    alpha_by_beta_dict: holds alpha, beta pair tuples as keys & alpha/beta as value, example:
                        {(-0.2010, -0.0018): 107.08}
    
    
    Output
    --------
    returns a plotly scatter plot with number of trials on X axis & corresponding probability of getting
    shocked for each pair of (alpha, beta) passed for comparison.
    
    """
    combined_pairs_obs_df= pd.DataFrame()
    title_txt = ""
    additional_txt = ""
    obs_column_names = [f'Dog_{ind+1}'for ind in range(dogs_data["Ndogs"])]
    for i, select_pair in enumerate(pairs_to_compare):
        i+=1
        title_txt+=f'Dog_X_m_{i} corresponds to {select_pair}, '

        obs_column_names_model_x =list(map(lambda name:f'{name}_m_{i}', obs_column_names))

        if alpha_by_beta_dict:
            additional_txt+=f'𝛼/𝛽 for Dog_X_m_{i} {round(alpha_by_beta_dict.get(select_pair), 2)}, '
        
        obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=obs_column_names_model_x)

        combined_pairs_obs_df= pd.concat([combined_pairs_obs_df, obs_y_df], axis=1)

    print(title_txt)
    print("\n%s"%additional_txt)

    if not original_obs is compare_dogs_given_parameters.__defaults__[0]:
        original_obs_column_names = list(map(lambda name:f'*{name}', obs_column_names))

        original_obs_df= pd.DataFrame(original_obs.numpy().T, columns=original_obs_column_names)
        combined_pairs_obs_df= pd.concat([combined_pairs_obs_df, original_obs_df], axis=1)
        print("\nNote: Legend *Dog_X_ corresponds to 'y' i.e.,original observation values")
        
    obs_y_title= "Observed values for all dogs given parameter for a chain"
    fig = px.scatter(combined_pairs_obs_df, title=obs_y_title)
    fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
    fig.show()


# **Also Optionally pass the `True observed y` values to `original_obs` argument for all 30 dogs to plot alongside the `observed y` from alpha-beta pairs of interest.**
# 
# **_Note_**: `True observed y` are marked with legends in format `*Dog_X`

# In[35]:


compare_dogs_given_parameters(selected_pairs, obs_y_dict, original_obs= y)


# **Observations:** The 3 individual scatter plots above correspond to 3 most optimum alpha-beta pairs from 3rd quadrant of contour plot drawn earlier; Also the scatterplot following them faciliates comparing obeserved y values for all 3 pairs at once:
# 
# Data for almost all dogs in the experiment favours m1 parameters (-0.19852, -0.0156), over m3 & m2; With exceptions of Dog 6, 7 showing affinity to m3 parameters (-0.19804, -0.01568), over m2 & m1 at all levels of 30 trials.

# **Plotting observed values y corresponding to pairs of alpha-beta with with `mean, minmum, maximum value of` $\frac{alpha}{beta}$**

# **Following computes $\frac{alpha}{beta}$ for each pair of alpha, beta and outputs pairs with `mean, maximum & minimum values`; that can therefore be marked on a single scatterplots for comparison of observed y values for all alpha-beta pairs of interest**

# In[36]:


def get_alpha_by_beta_records(chain_df, metrics=["max", "min", "mean"]):
    """
    
    Input
    --------
    chain_df: dataframe holding sampled parameters for a given chain
    
    returns an alpha_by_beta_dictionary with alpha, beta pair tuples as keys & alpha/beta as value,
    example: {(-0.2010, -0.0018): 107.08}
    
    Output
    -------
    Return a dictionary with values corresponding to statistics/metrics asked in argument metrics, computed
    over alpha_by_beta column of passed dataframe.
    
    
    """
    alpha_beta_dict= {}
    
    chain_df["alpha_by_beta"] = chain_df["alpha"]/chain_df["beta"]
    min_max_values = dict(chain_df["alpha_by_beta"].describe())
    alpha_beta_list= list(map(lambda key: chain_df[chain_df["alpha_by_beta"]<=min_max_values.get(key)].sort_values(["alpha_by_beta"]).iloc[[-1]].set_index(["alpha", "beta"])["alpha_by_beta"].to_dict(), metrics))

    [alpha_beta_dict.update(element) for element in alpha_beta_list];
    return alpha_beta_dict


alpha_by_beta_dict = get_alpha_by_beta_records(chain_samples_df, metrics=["max", "min", "mean"])# outputs a dict of type {(-0.2010, -0.0018): 107.08}
print("Alpha-beta pair with value as alpha/beta: ", alpha_by_beta_dict)


alpha_by_beta_selected_pairs= list(alpha_by_beta_dict.keys())
alpha_by_beta_obs_y_dict = get_obs_y_dict(alpha_by_beta_selected_pairs, x_avoidance, x_shocked)# Outputs observed_values for given (alpha, beta)


# **Following is the scatter plot for `observed y` values corresponding to pairs of `alpha, beta` yielding `minimum, maximum & mean` value for $\frac{alpha}{beta}$.**
# 
# **_Note_**: The y i.e., original observations are simultaneously plotted side by side.

# In[37]:


compare_dogs_given_parameters(alpha_by_beta_selected_pairs, alpha_by_beta_obs_y_dict, 
                              original_obs= y, alpha_by_beta_dict= alpha_by_beta_dict)


# **Observations:** The scatter plots above corresponds to 3 pairs of alpha-beta values from contour plot drawn earlier, which correspond to maxmimum, minimum & mean value of 𝛼/𝛽. Plot faciliates comparing `obeserved y` values for all pairs with `True observed y` at once:
# 
#     1. Data for for first 7 dogs in the experiment favours m1 parameters (-0.184, -0.0015) with highest 𝛼/𝛽 around 116, followed by m3 & m2 at all levels of 30 trials. Avoidance learning in these 7 dogs is captured suitablely by model 2 but most of the instances for which they are shocked, are modelled well with m1 parameters.
#     
#     2. Data for for rest 23 dogs in the experiment showed affinity for m2 parameters (-0.197, -0.016) with lowest 𝛼/𝛽 around 11, followed by m3 & m1 at all levels of 30 trials; Likewise Avoidance learning in these 23 dogs is captured suitablely by model 2 but most of the instances for which they are shocked, are modelled well with m1 parameters only.
#     
#     3. Data for Dogs 18-20 fits model 2 increasingly well after 10th trial; Whereas for Dogs 21-30 model 2 parameters fit the original data exceptionally well after 6th trial only.

# ### 6. Model Comparison
# **Compare Dogs model with Normal prior & Uniform prior using Deviance Information Criterion (DIC)**

# **DIC is computed as follows**
# 
# $D(\alpha,\beta) = -2\ \sum_{i=1}^{n} \log P\ (y_{i}\ /\ \alpha,\beta)$
# 
# $\log P\ (y_{i}\ /\ \alpha,\beta)$ is the log likehood of shocks/avoidances observed given parameter $\alpha,\beta$, this expression expands as follows:
# 
# $$D(\alpha,\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\alpha Xa_{i}\ +\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\alpha Xa_{i}\ +\beta\ Xs_{i})})]$$
# 
# 
# #### Using $D(\alpha,\beta)$ to Compute DIC
# 
# $\overline D(\alpha,\beta) = \frac{1}{T} \sum_{t=1}^{T} D(\alpha,\beta)$
# 
# $\overline \alpha = \frac{1}{T} \sum_{t=1}^{T}\alpha_{t}\\$
# $\overline \beta = \frac{1}{T} \sum_{t=1}^{T}\beta_{t}$
# 
# $D(\overline\alpha,\overline\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i})})]$
# 
# 
# **Therefore finally**
# $$
# DIC\ =\ 2\ \overline D(\alpha,\beta)\ -\ D(\overline\alpha,\overline\beta)
# $$
# 
# 

# **Following method computes deviance value given parameters `alpha & beta`**

# In[38]:


def calculate_deviance_given_param(parameters, x_avoidance, x_shocked, y):
    """

    Input
    -------
    parameters : dictionary containing sampled values of parameters alpha & beta
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25)
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    

    Output
    -------
    Computes deviance as D(Bt)
    D(Bt)   : Summation of log likelihood / conditional probability of output, 
              given param 'Bt' over all the 'n' cases.

    Returns deviance value for a pair for parameters, alpha & beta.
    
    """

    D_bt_ = []
    p = parameters["alpha"]*x_avoidance + parameters["beta"]*x_shocked# alpha * Xai + beta * Xsi
    p=p.double()
    p= torch.where(p<-0.0001, p, -0.0001).float()
    
    Pij_vec = p.flatten().unsqueeze(1)# shapes (750, 1)
    Yij_vec= y.flatten().unsqueeze(0)# shapes (1, 750)
    
    # D_bt = -2 * Summation_over_i-30 (yi.(alpha.Xai + beta.Xsi)+ (1-yi).log (1- e^(alpha.Xai + beta.Xsi)))
    D_bt= torch.mm(Yij_vec, Pij_vec) + torch.mm(1-Yij_vec, torch.log(1- torch.exp(Pij_vec)))
    D_bt= -2*D_bt.squeeze().item()
    return D_bt

def calculate_mean_deviance(samples, x_avoidance, x_shocked, y):
    """
    
    Input
    -------
    samples : dictionary containing mean of sampled values of parameters alpha & beta.
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25).
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).

    
    Output
    -------
    Computes mean deviance as D(Bt)_bar
    D(Bt)_bar: Average of D(Bt) values calculated for each 
                   Bt (Bt is a single param value from chain of samples)
    Returns mean deviance value for a pair for parameters, alpha & beta.
    
    
    """
    samples_count = list(samples.values())[0].size()[0]
    all_D_Bts= []
    for index in range(samples_count):# pair of alpha, beta
        samples_= dict(map(lambda param: (param, samples.get(param)[index]), samples.keys()))
        
        D_Bt= calculate_deviance_given_param(samples_, x_avoidance, x_shocked, y)
        all_D_Bts.append(D_Bt)
    
    D_Bt_mean = torch.mean(torch.tensor(all_D_Bts))
    
    D_Bt_mean =D_Bt_mean.squeeze().item()
    
    return D_Bt_mean
        


# **Following method computes `deviance information criterion` for a given bayesian model & chains of sampled parameters `alpha & beta`**

# In[39]:


def DIC(sample_chains, x_avoidance, x_shocked, y):
    """
        
    Input
    -------
    sample_chains : dictionary containing multiple chains of sampled values, with chain name as
                    key and sampled values of parameters alpha & beta.
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25).
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
            
    Output
    -------
    Computes DIC as follows
    D_mean_parameters: 𝐷(𝛼_bar,𝛽_bar), Summation of log likelihood / conditional probability of output, 
                   given average of each param 𝛼, 𝛽, over 's' samples, across all the 'n' cases.
    D_Bt_mean: 𝐷(𝛼,𝛽)_bar, Summation of log likelihood / conditional probability of output, 
                   given param 𝛼, 𝛽, across all the 'n' cases.
        
    𝐷𝐼𝐶 is computed as 𝐷𝐼𝐶 = 2 𝐷(𝛼,𝛽)_bar − 𝐷(𝛼_bar,𝛽_bar)
    
    returns Deviance Information Criterion for a chain alpha & beta sampled values.


    """
    dic_list= []
    for chain, samples in sample_chains.items():
        samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors

        mean_parameters = dict(map(lambda param: (param, torch.mean(samples.get(param))), samples.keys()))
        D_mean_parameters = calculate_deviance_given_param(mean_parameters, x_avoidance, x_shocked, y)

        D_Bt_mean = calculate_mean_deviance(samples, x_avoidance, x_shocked, y)
        dic = round(2* D_Bt_mean - D_mean_parameters,3)
        dic_list.append(dic)
        print(". . .DIC for %s: %s"%(chain, dic))
    print("\n. .Mean Deviance information criterion for all chains: %s\n"%(round(np.mean(dic_list), 3)))

def compare_DICs_given_model(x_avoidance, x_shocked, y, **kwargs):
    """
    
    Input
    --------
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25).
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    kwargs: dict of type {"model_name": sample_chains_dict}
    
    Output
    --------
    Compares Deviance Information Criterion value for a multiple bayesian models.
    
    
    """
    for model_name, sample_chains in kwargs.items():
        print("%s\n\nFor model : %s"%("_"*30, model_name))
        DIC(sample_chains, x_avoidance, x_shocked, y)


# **Define alternate model with different prior such as uniform distribution**
# 
# The following model is defined in the same manner using Pyro as per the following expression of generative model for this dataset, just with modification of prior distribution to `Uniform` rather than `Normal` as follows:
# 
# $\pi_j$  ~   $bern\ (\exp \ (\alpha.XAvoidance + \beta.XShocked)\ )$,  $prior\ \alpha$ ~ $U(0., 316.)$,  $\beta$ ~ $U(0., 316.)$

# In[40]:


# Dogs model with uniform prior
def DogsModelUniformPrior(x_avoidance, x_shocked, y):
        """
    Input
    -------
    x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                 30 dogs & 25 trials, shaped (30, 25)
    x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    y:           tensor holding response for all dogs & trials, example for 30 dogs
                 & 25 trials, shaped (30, 25).
    
    Output
    --------
    Implements pystan model: {
              alpha ~ uniform(0.0, 316.2);
              beta  ~ uniform(0.0, 316.2);
              for(dog in 1:Ndogs)  
                for (trial in 2:Ntrials)  
                  y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}
    
    """
    alpha = pyro.sample("alpha", dist.Uniform(-10, -0.00001))
    beta = pyro.sample("beta", dist.Uniform(-10, -0.00001))
    with pyro.plate("data"):
        pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)

hmc_sample_chains_uniform_prior= get_hmc_n_chains(DogsModelUniformPrior, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# **compute & compare `deviance information criterion` for a multiple bayesian models**

# In[41]:


compare_DICs_given_model(x_avoidance, x_shocked, y, Dogs_normal_prior= hmc_sample_chains, Dogs_uniform_prior= hmc_sample_chains_uniform_prior)


# _______________
