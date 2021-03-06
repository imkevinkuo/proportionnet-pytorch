import pickle
from advertiser import *

####################################################################################
# Helper Functions
####################################################################################
def report_error(msg,key1,key2):
    os.system("echo \""+str(key1)+"-"+str(key2)+": "+msg+"\">> errorsExperiment")

def to_relative(res,num_adv):
    ## Convert "probability of advertiser winning, given usertype" to
    ## "probability of advertiser winning on a particular user type given he won"
    res_rel=copy.deepcopy(res)
    print(res_rel)
    for i in range(num_adv):
        tmp=res_rel[i][0]+res_rel[i][1];
        res_rel[i][0]/=tmp;res_rel[i][1]/=tmp
    res_rel=np.around(res_rel, decimals=10)
    return res_rel

####################################################################################
# Functions to gather data
####################################################################################
corr=0 # correlation between different keys
def get_key_pair(number_of_keys,number_of_cores,corr_local):
    global corr
    corr = corr_local

    key_pair=[]
    for key1 in range(number_of_keys):
        for key2 in range(number_of_keys):
            # if key2<=key1 or corr[key1][key2]<2: continue;
            if key2 <= key1: continue;
            key_pair.append([key1,key2])

    def get_key(key_pair):
        global corr
        return int(corr[key_pair[0]][key_pair[1]])

    key_pair=sorted(key_pair,key=get_key,reverse=True)
    print("number of key pairs found", len(key_pair), flush=True)
    return key_pair

def get_adv(key1,key2):
    folder="data/keys-"+str(key1)+"-"+str(key2)+"/"
    ## Maximum number of advertisers
    shared_adv = pickle.load(open(folder+"/advertiser", 'rb'))
    low_var_adv_tmp = pickle.load(open("data/low_var_adv", 'rb'))
    low_var_adv  = set()
    for pair in low_var_adv_tmp:
        if pair[1]==key1 or pair[1]==key2:
            low_var_adv.add(pair[0])

    shared_adv = shared_adv.difference(low_var_adv)
    num_adv=len(shared_adv);
    # if(len(shared_adv)<2):
    #     print("num_adv:",num_adv,flush=True)
    #     reportError("Only "+str(num_adv)+ " advertiser",key1,key2);
    #     return -1,-1,-1

    ## Folder to load pdf and cdf
    folder="data/keys-"+str(key1)+"-"+str(key2)+"/"

    ##################################################
    ## Get distributions of advertisers
    ##################################################
    cdf=[[] for i in range(num_adv)];
    pdf=[[] for i in range(num_adv)];
    inv_cdf=[[] for i in range(num_adv)];
    inv_phi=[[] for i in range(num_adv)];
    range_phi_min=[[] for i in range(num_adv)]

    pdf_arr = [[] for a in range(numAttr)]
    cdf_arr = [[] for a in range(numAttr)]

    adv_id = []

    i=0
    for adv in shared_adv:
        if i >= num_adv: break
        folder="data/keys-"+str(key1)+"-adv"+str(adv)+"/"
        tmpcdf = pickle.load(open(folder+"cdf", 'rb'))
        tmpinv_cdf = pickle.load(open(folder+"inv_cdf", 'rb'))
        tmppdf = pickle.load(open(folder+"pdf", 'rb'))
        tmpinv_phi = pickle.load(open(folder+"inv_phi", 'rb'))
        tmprange_phi_min = pickle.load(open(folder+"range_phi_min", 'rb'))
        cdf[i].append(tmpcdf[0])
        pdf[i].append(tmppdf[0])
        inv_cdf[i].append(tmpinv_cdf[0])
        inv_phi[i].append(tmpinv_phi[0])
        range_phi_min[i].append(tmprange_phi_min[0])

        pdf_arr[0].append(pickle.load(open(folder+"pdf_array_virtual_valuation", 'rb')))
        cdf_arr[0].append(pickle.load(open(folder+"cdf_array_virtual_valuation", 'rb')))

        adv_id.append(adv)

        i+=1
    i=0
    for adv in shared_adv:
        if i >= num_adv: break
        folder="data/keys-"+str(key2)+"-adv"+str(adv)+"/"
        tmpcdf = pickle.load(open(folder+"cdf", 'rb'))
        tmpinv_cdf = pickle.load(open(folder+"inv_cdf", 'rb'))
        tmppdf = pickle.load(open(folder+"pdf", 'rb'))
        tmpinv_phi = pickle.load(open(folder+"inv_phi", 'rb'))
        tmprange_phi_min = pickle.load(open(folder+"range_phi_min", 'rb'))
        cdf[i].append(tmpcdf[0])
        pdf[i].append(tmppdf[0])
        inv_cdf[i].append(tmpinv_cdf[0])
        inv_phi[i].append(tmpinv_phi[0])
        range_phi_min[i].append(tmprange_phi_min[0])

        pdf_arr[1].append(pickle.load(open(folder+"pdf_array_virtual_valuation", 'rb')))
        cdf_arr[1].append(pickle.load(open(folder+"cdf_array_virtual_valuation", 'rb')))
        i+=1

    adv = []
    for i in range(num_adv):
        adv.append(Advertiser(cdf[i],inv_cdf[i],inv_phi[i],pdf[i],range_phi_min[i],adv_id[i]))

    return adv,pdf_arr,cdf_arr

def getPdfCdf(a,i):
    iter=100*samples
    bid = np.linspace(min_x,max_x,iter)

    cdf_list = adv[i].cdf[a](bid)
    one=np.ones(int(iter))
    virBid = bid-(one-cdf_list)/(adv[i].pdf[a](bid)+(one/100000.0))

    min_x = -2
    max_x = 4
    samples = 6001
    bins = np.linspace(min_x, max_x, samples)
    digitized = np.digitize(virBid, bins)
    pdf = np.array([0 for i in range(len(bins))])
    for i in range(len(digitized)):
        if digitized[i] != samples: pdf[digitized[i]]+=1
    pdf[0]=0 # remove lower tail
    pdf = (pdf / np.sum(pdf)) * (201/2.0)
    cdf=np.zeros(len(pdf))
    cdf[1:]=np.cumsum(pdf[1:]*np.diff(bins))
    pdf=pdf/cdf[-1]
    cdf=cdf/cdf[-1]
    return pdf,cdf


def get_revenue_array(adv,shift,uT):
    iter = 10000;

    if len(adv) != len(shift):
        print("Error! len(adv) != len(shift)"+str(len(adv))+"!="+str(len(shift)), flush=True)

    num_adv=len(adv)
    bids = [ [] for i in range(num_adv)]
    virB = [ [] for i in range(num_adv)]

    # Getting bids
    for i in range(num_adv):
        bids[i],virB[i]=adv[i].bid2(uT,10+int(iter))
    bids=np.array(bids);virB=np.array(virB);

    # Calculate the winner
    abc = virB+shift.reshape((-1,1))
    winner = np.argmax(abc,axis=0)
    runUp = np.array([-1 for i in range(int(iter))])
    for j in range(int(iter)):
        run=-1;se=-100;
        for i in range(num_adv):
            if i == winner[j]: continue;
            if se<abc[i][j]:
                se=abc[i][j];
                runUp[j]=i;

    # Calculating payment
    pay=[]
    query = [[] for i in range(num_adv)]
    when  = [[] for i in range(num_adv)]
    who   = []
    revonue_array=[];

    j=0;i=0;
    for it in range(int(iter)):
        j=runUp[it];i=winner[it]
        value=0
        value=virB[j][it]-shift[i]+shift[j]
        value=max(value,adv[i].range_phi_min[uT])
        if type(value) != int and type(value) != float and type(value)!= np.float64:
            value=value[0]
        ## Bluk queries are much faster
        query[i].append(value)
        when[i].append(it)
        who.append(i)

    ## Get answers to bulk queries
    try:
        ans = [ adv[i].inv_phi[uT](query[i]) for i in range(num_adv)]
    except:
        return -1
    cnt = [0 for i in range(num_adv)]
    for it in range(int(iter)):
        j=runUp[it];i=who[it]
        tmp=ans[i][cnt[i]]
        tmp=min(tmp,bids[j][it])
        revonue_array.append(tmp)
        cnt[i]+=1

    return revonue_array

# Runs the \alpha-shifted auction and returns coverage
def shiftedMyer(adv,shift,uT,stats=0):
    iter=10000
    num_adv=len(adv)

    ## Get virtual bids
    virB = [ [] for i in range(num_adv) ]
    for i in range(num_adv):
        virB[i]=adv[i].bid(uT,10+int(iter))
    virB=np.array(virB)

    # calculate the winner
    print("virb",virB)
    print("shift",shift)
    tmp = virB+shift.reshape((-1,1))
    winner = np.argmax(tmp,axis=0)

    results=np.zeros(num_adv)
    for w in winner: results[w]+=1
    results /= iter*1.0

    #Scaling according to user distribution
    results*=0.5

    return results
