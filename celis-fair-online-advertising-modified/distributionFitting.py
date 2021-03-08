#!/usr/bin/python
import numpy as np
from scipy.stats import uniform
import scipy.interpolate as interpolate
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import mixture
import pickle, os, time, sys

N_ADV = 2
M_KEY = 2

## Fit a mixture of gaussians to the bidding pattern of given agent for the given keyword
def distribution_fitting(bids,agents,key,adv):
    #count: number of bids processed
    count = 0

    ## Number of Gaussians in the mixture model
    num_gaussian = 5

    ##################################################
    ## Distribution characteristics for each agent
    ## range_phi_min : list of min of range of virtual valuation of all advertisers
    ## range_phi_max : list of max of range of virtual valuation of all advertisers
    ## cdf : list of cdf of all advertisers
    ## pdf : list of pdf of all advertisers
    ## inv_cdf : list of inverse cdf of all advertisers
    ## inv_phi : list of inverse pdf of all advertisers
    ##################################################
    cdf = []
    pdf = []
    inv_cdf = []
    inv_phi = []
    range_phi_min = []
    range_phi_max = []

    ## b: Hash list of (list of number of bids by top 50 advertisers bidding on the keyword)
    ## b[str(i)+"count"][0][j]: number of bids on keyword i by jth largest advertiser
    ## b[str(i)+"agent"][0][j]: id of jth largest advertiser on keyword i
    b = sio.loadmat('data/find_agent_keyword')
    print("on agent:", adv,flush=True)

    # Top agents (in terms of number of bids) selected for keyword
    # adv = b[str(key)+'agent'][0][i]
    ## Form list of bids by agent on the keyword
    agent_bids = []

    for j in range(len(bids)):
        ## Only take bids under $100
        if agents[j] == adv and bids[j]<100:
            agent_bids.append(bids[j])
    #if len(agent_bids)<1000:
    #    print("Error!!!",flush=True)
    #    return [cdf,pdf,inv_cdf,inv_phi,range_phi_min,count,-1]

    ##################################################
    ## Model fitting here
    ##################################################
    x = np.asarray(agent_bids)
    x = np.reshape(x,[len(agent_bids),1])

    # UNIFORM
    loc, scale = uniform.fit(x)
    scale = np.round(scale)
    #
    
    ## Fit GaussianMixture with num_gaussian components
    # g = mixture.GaussianMixture(n_components=num_gaussian)
    # g.fit(x)

    ## Weights, means and variances of the gaussians
    # w = np.reshape(g.weights_,[num_gaussian,1])
    # m = g.means_
    # var = g.covariances_

    ## Eliminating low variance agents
    ## Hard to approximate distributions for these.
    ## Incrementally reduce number of gausians if variance is low
    # if np.min(var) < 0.003:
    #     num_gaussian = 3
    #     g = mixture.GaussianMixture(n_components=num_gaussian)
    #     g.fit(x)
    #     w = np.reshape(g.weights_,[num_gaussian,1])
    #     m = g.means_
    #     var = g.covariances_
    #     if np.min(var) < 0.003:
    #         num_gaussian = 1
    #         g = mixture.GaussianMixture(n_components=num_gaussian)
    #         g.fit(x)
    #         w = np.reshape(g.weights_,[num_gaussian,1])
    #         m = g.means_
    #         var = g.covariances_
    #         if np.min(var) < 0.003:
    #             ## Too low variance cannot include this bidder.
    #             print("Excluding advertiser: ",adv,"Variance for 1 gaussian is: ",np.min(var),flush=True)
    #            return [cdf,pdf,inv_cdf,inv_phi,range_phi_min,count,-1]

    count += len(agent_bids)
    min_x = -2
    max_x = 4
    samples = 6001
    x_range=np.linspace(min_x,max_x,samples)

    ## Calculating pdf of distribution
    # y = []
    # for j in range(num_gaussian):
    #     y.append(np.exp(-(x_range[:-1]-m[j])**2/(2*var[j][0][0]))/(2*np.pi*var[j][0][0])**0.5)

    # pdf_fitted = np.dot(np.transpose(w),y)[0]
    pdf_fitted = uniform.pdf(x_range, loc, scale)
    # # pdf_prime = np.dot(np.transpose(w),yprime)[0]
    
    ## Calculates the CDF
    cum_values = np.zeros(x_range.shape)
    cum_values[1:] = np.cumsum(pdf_fitted[:-1:]*np.diff(x_range))
    ## Hard code normalization
    # cum_values[-1]=1
    # cum_old = 0

    for j in range(1, len(cum_values)):#lower break the distribution
        if cum_values[j] > 0.0001:
            cum_values = cum_values[j-1:]
            cum_values[0] = 0
            x_range = x_range[j-1:]
            pdf_fitted = pdf_fitted[j-1:]
            break

    ## Hard code normalisation
    # cum_values[-1]=1

    ## Helpful plotting functions
    # import matplotlib.pyplot as plt
    # print(cum_values)
    # plt.scatter(x_range,pdf_fitted,s=1)
    # plt.plot(x_range,pdf_prime)
    # plt.scatter(x_range,cum_values,s=1)
    # plt.show()


    ##################################################
    ## distributions
    ## x_range: linear range of valuations
    ## y: values of valuations corresponding to x_range
    ##    y[i] =  valuations for x_range[i]
    ## pdf_fitted: pdf of valuations
    ## cum_values: cdf of valuations
    ## cdfh: interpolated cdf of valuations, returns a function
    ## inv_cdfh: interpolated inverse of cdf of valuations, returns a function
    ## pdfh: interpolated pdf of valuations, returns a function
    ## phis: virtual valuations of for fitted pdf and cdf
    ## inv_phih: interpotaled inverse of phi
    ##################################################
    cdfh = interpolate.interp1d(x_range,cum_values,fill_value=(0,1), bounds_error=False)
    cum_values=sorted(cum_values)
    inv_cdfh = interpolate.interp1d(cum_values, x_range)
    pdfh = interpolate.interp1d(x_range, pdf_fitted,fill_value=(0,0), bounds_error=False)
    # pdfprimeh = interpolate.interp1d(bin_edges2, pdf_prime)

    ## Calculate virtual valuation
    def vval(i):
        if cum_values[i] == 0:
            return 0
        if cum_values[i] == 1:
            return x_range[i]
        return x_range[i] - (1 - cum_values[i] / pdf_fitted[i])

    phis = [vval(i) for i in range(len(x_range))]

    # plt.scatter(x_range, pdfh(x_range), s=1)
    # plt.scatter(x_range, phis, s=1)
    # plt.title(f"pdf and vval key {key} adv {adv}")
    # plt.xlim(0,4)
    # plt.ylim(-4,4)
    # plt.show()
    for j in range(len(phis)-1):
        ## Enforce virtual valuation to be strictly monotonic
        if phis[j+1] < phis[j]:
            phis[j+1] = phis[j]+0.0001

    ## The functions are not invertible
    inv_phih = interpolate.interp1d(phis, x_range)
    range_phi_min.append(phis[0])
    range_phi_max.append(phis[-1])

    cdf.append(cdfh)
    pdf.append(pdfh)
    inv_cdf.append(inv_cdfh)
    inv_phi.append(inv_phih)

    return [cdf,pdf,inv_cdf,inv_phi,range_phi_min,count,1]

def pickle_fitted_distribution(key,adv,low_var_adv):
    ##################################################
    ## Model fitting parameters
    ## n_samples: Number of samples to draw from each distribution
    ##################################################
    scale = 1
    n_samples = 5000
    threshold_count = 200

    ##Load keyword "keyword", 1-keyword.mat is a hash table
    ## 1bids has all the bids for keyword 1
    ## 1agents has the corresponding agents.
    # Loading the agent and bids
    with open('data/key-'+str(key), 'rb') as f:
        tmp = pickle.load(f)

    bids=tmp["bid"]
    agents=tmp["advertiser"]

    print("calling distribution fitting",flush=True)
    ## Get distributions of all agents for the keyword
    [cdf,pdf,inv_cdf,inv_phi,range_phi_min,count,no_error] = distribution_fitting(bids,agents,key,adv)

    if no_error == -1: low_var_adv.append([adv,key])

    print("saving fs",flush=True)
    folder="data/keys-"+str(key)+"-adv"+str(adv)+"/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder+"cdf", 'wb') as f:
        pickle.dump(cdf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"pdf", 'wb') as f:
        pickle.dump(pdf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"inv_cdf", 'wb') as f:
        pickle.dump(inv_cdf, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"inv_phi", 'wb') as f:
        pickle.dump(inv_phi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"range_phi_min", 'wb') as f:
        pickle.dump(range_phi_min, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"count", 'wb') as f:
        pickle.dump(count, f, protocol=pickle.HIGHEST_PROTOCOL)

    return



####################################################################################
# Driving function
####################################################################################

if __name__ == '__main__' :
    start_time = time.time();
    arg=sys.argv

    # load the correlation matrix between keywords
    with open("data/corr_all_key", 'rb') as f:
        corr = pickle.load(f)

    # low_var_adv: stores [adv,key] of low variance advertisers
    low_var_adv=[]

    for key in range(M_KEY):
        for adv in range(N_ADV):
            print("key:",key,", adv:",adv,flush=True)
            pickle_fitted_distribution(key,adv,low_var_adv)
            
    # adv_key = [set() for i in range(M_KEY)]
    # for key1 in range(M_KEY):
    #     if key1%100==0: print(key1,flush=True)
    #     for key2 in range(M_KEY):
    #         if(key2<=key1): continue
    #         MIN_CORR = -1
    #         if corr[key1][key2] > MIN_CORR:
    #             with open("data/keys-"+str(key1)+"-"+str(key2)+"/advertiser", 'rb') as f:
    #                 shared_adv = pickle.load(f)
    #             for adv in shared_adv:
    #                 adv_key[key1].add(adv)
    #                 adv_key[key2].add(adv)

    # for key in range(len(adv_key)):
    #     if len(adv_key[key])<2: continue;
    #     for adv in adv_key[key]:
    #         print("key:",key,", adv:",adv,flush=True)
    #         pickle_fitted_distribution(key,adv,low_var_adv)

    with open("data/low_var_adv", 'wb') as f:
        pickle.dump(low_var_adv, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(low_var_adv,flush=True)

    print("Time taken : %s seconds" % (time.time() - start_time),flush=True)
