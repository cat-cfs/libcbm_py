#!/usr/bin/env python
# coding: utf-8

# # LibCBM versus CBM-CFS3 growth testing #
# This notebook is a automated test of the growth curve implementation in CBM-CFS3 versus that in LibCBM.  The objective is to ensure that LibCBM results match the established CBM-CFS3 model very closely. 
# 
# The script automatically generates randomized merchantable volume growth curves in various configurations
#  * softwood, hardwood or mixed
#  * random spatial unit (which is associated with biomass conversion parameters)
#  * one of several random age/volume curve generators with varying amplitudes and start points (impulse, ramp, step, and exp curve)
#  
# It then compares the results and sorts them by largest different for plotting.

# In[1]:


import os, json, math
import numpy as np
import pandas as pd



# libCBM related imports

# In[2]:


from libcbmwrapper import LibCBMWrapper
import libcbmconfig
import cbmconfig
import cbm_defaults


# cbm3 related imports

# In[3]:



# scenario set-up

# In[4]:


db_path = 'C:\dev\cbm_defaults\cbm_defaults.db'
n_steps = 225

age_interval = 10 #required by cbm3
num_age_classes = 20 #required by cbm3

def get_classifier_name(id):
    return str(id)

def create_growth_curve(id, admin_boundary, eco_boundary, softwood_species=None, 
                        softwood_age_volume_pairs=None, hardwood_species=None, 
                        hardwood_age_volume_pairs=None):
    return {
        "id":id,
        "admin_boundary": admin_boundary,
        "eco_boundary": eco_boundary,
        "softwood_species": softwood_species,
        "softwood_age_volume_pairs": softwood_age_volume_pairs,
        "hardwood_species": hardwood_species,
        "hardwood_age_volume_pairs": hardwood_age_volume_pairs
    }

def generate_random_yield(size, age_step, ndigits ):
    
    #return y for a single value of x: nx
    def get_impulse_func():
        y = np.random.rand(1)[0] * 500
        nx = round(np.random.rand(1)[0] * size)
        def impulse(x):
            if x==nx:
                return y
            else:
                return 0
        return impulse

    #return a step of value y for the range minx to maxX
    def get_step_func():
        y = np.random.rand(1)[0] * 500
        minX = round(np.random.rand(1)[0] * size)
        maxX = round(np.random.rand(1)[0] * (size-minX)) + minX
        def step(x):
            if x == 0:
                return 0
            if x>=minX and x<=maxX:
                return y
            else:
                return 0
        return step

    def get_ramp_func():
        rate = np.random.rand(1)[0] * 5
        def ramp(x):
            return x*rate
        return ramp

    def get_expCurve_func():
        yMax = np.random.rand(1)[0] * 500
        def expCurve(x):
            return yMax - math.exp(-x) * yMax
        return expCurve
    
    func = np.random.choice([get_impulse_func,get_step_func,get_ramp_func,get_expCurve_func],1)[0]()
    return [(x*age_step,round(func(x),2)) for x in range(0, size)]
        
    
def generate_cases(random_seed, num_curves, dbpath, ndigits):
    '''
    ndigits is here because the CBM-CFS3 toolbox rounds yield curve volumes to 2 decimal places
    '''
    np.random.seed(random_seed)
    
    stand_type = np.random.choice(["softwood_only", "hardwood_only", "mixed"], num_curves)
    
    species = cbm_defaults.load_species_reference(dbpath, "en-CA")
    sw_species = [x for x in species if species[x]["forest_type_id"]==1 and len(x)<50] #exclude species names that are too long for the project database schema
    hw_species = [x for x in species if species[x]["forest_type_id"]==3 and len(x)<50]
    
    
    random_sw_species = np.random.choice(list(sw_species), num_curves)
    random_hw_species = np.random.choice(list(hw_species), num_curves)
    
    spatial_units = cbm_defaults.get_spatial_unit_ids_by_admin_eco_name(dbpath, "en-CA")
    random_spus = np.random.choice([",".join(x) for x in spatial_units.keys()], num_curves)

    cases = []
    for i in range(num_curves):
        spu = random_spus[i].split(',')        
        cases.append(create_growth_curve(
            id = i+1,
            admin_boundary=spu[0],
            eco_boundary=spu[1],
            softwood_species = random_sw_species[i] if stand_type[i] in ["softwood_only", "mixed"] else None,
            softwood_age_volume_pairs = generate_random_yield(num_age_classes,age_interval, ndigits)
                if stand_type[i] in ["softwood_only", "mixed"] else None,
            hardwood_species = random_hw_species[i] if stand_type[i] in ["hardwood_only", "mixed"] else None,
            hardwood_age_volume_pairs = generate_random_yield(num_age_classes,age_interval, ndigits) 
                if stand_type[i] in ["hardwood_only", "mixed"] else None,           
             ))
    return cases


# run test cases on libCBM

# In[5]:


def run_libCBM(dbpath, cases, nsteps):
    
    dllpath = r'C:\dev\LibCBM\LibCBM\x64\Debug\LibCBM.dll'

    dlldir = os.path.dirname(dllpath)
    cwd = os.getcwd()
    os.chdir(dlldir)
    dll = LibCBMWrapper(dllpath)
    os.chdir(cwd)
    
    dll.Initialize(libcbmconfig.to_string(
        {
            "pools": cbm_defaults.load_cbm_pools(dbpath),
            "flux_indicators": cbm_defaults.load_flux_indicators(dbpath)
        }))
    
    #create a single classifier/classifier value for the single growth curve
    classifiers_config = cbmconfig.classifier_config([
        cbmconfig.classifier("growth_curve", [
            cbmconfig.classifier_value(get_classifier_name(c["id"])) 
            for c in cases
        ])
    ])


    transitions_config = []
    species_reference = cbm_defaults.load_species_reference(dbpath, "en-CA")
    spatial_unit_reference = cbm_defaults.get_spatial_unit_ids_by_admin_eco_name(dbpath, "en-CA")
    curves = []
    for c in cases:
        classifier_set = [get_classifier_name(c["id"])]
        merch_volumes = []

        if c["softwood_species"]:
            merch_volumes.append({
                "species_id": species_reference[c["softwood_species"]]["species_id"],
                "age_volume_pairs": c["softwood_age_volume_pairs"]
            })

        if c["hardwood_species"]:
            merch_volumes.append({
                "species_id": species_reference[c["hardwood_species"]]["species_id"],
                "age_volume_pairs": c["hardwood_age_volume_pairs"]
            })

        curve = cbmconfig.merch_volume_curve(
            classifier_set = classifier_set,
            merch_volumes = merch_volumes)
        curves.append(curve)

    merch_volume_to_biomass_config = cbmconfig.merch_volume_to_biomass_config(
        dbpath, curves)

    dll.InitializeCBM(libcbmconfig.to_string({
        "cbm_defaults": cbm_defaults.load_cbm_parameters(dbpath),
        "merch_volume_to_biomass": merch_volume_to_biomass_config,
        "classifiers": classifiers_config["classifiers"],
        "classifier_values": classifiers_config["classifier_values"],
        "transitions": []
    }))
                      
    

    nstands = len(cases)
    age = np.zeros(nstands,dtype=np.int32)
    classifiers = np.zeros((nstands,1),dtype=np.int32)
    classifiers[:,0]=[classifiers_config["classifier_index"][0][get_classifier_name(c["id"])] for c in cases]
         
    spatial_units = np.array(
        [spatial_unit_reference[(c["admin_boundary"],c["eco_boundary"])]
            for c in cases],dtype=np.int32)

    pools = np.zeros((nstands,len(config["pools"])))
    pools[:,0] = 1.0

    op = dll.AllocateOp(nstands)

    result = pd.DataFrame()
    for i in range(0, nsteps):
        dll.GetMerchVolumeGrowthOps(
            op, 
            classifiers=classifiers,
            pools=pools,
            ages=age,
            spatial_units=spatial_units,
            last_dist_type=None,
            time_since_last_dist=None,
            growth_multipliers=None)

        #since growth in CBM3 is split into 2 phases per timestep, 
        #we need to apply the operation 2 times in order to match it
        dll.ComputePools([op, op], pools)

        iteration_result = pd.DataFrame({x["name"]: pools[:,x["index"]] for x in config["pools"]})
        iteration_result.insert(0, "age", age+1)
        iteration_result.reset_index(level=0, inplace=True)
        result = result.append(iteration_result)

        age += 1

    result = result.reset_index(drop=True)
    return result


# run test cases on cbm-cfs3. uses [StandardImportToolPlugin](https://github.com/cat-cfs/StandardImportToolPlugin) and [cbm3-python](https://github.com/cat-cfs/cbm3_python) to automate cbm-cfs3 functionality

# In[6]:



# generate randomized growth curve test cases

# In[7]:


cases = generate_cases(random_seed=1, num_curves=1, dbpath=db_path, ndigits=2)


# run the test cases on libCBM

# In[8]:


libCBM_result = run_libCBM(db_path, cases, n_steps)
libCBM_result["identifier"] = (libCBM_result["index"]+1).apply(get_classifier_name)


# run the test cases on cbm-cfs3

# In[9]:

