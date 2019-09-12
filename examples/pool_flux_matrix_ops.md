---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# LibCBM Pool, and Flux Matrix Operations #
This notebook demonstrates LibCBM matrix functionality.  These functions can be used to inject flows into a LibCBM application directly from python.


It also performs tests by running equivalent matrix operations using the numpy matmul function, and comparing the output.
<!-- #endregion -->

```python
import os, json
import numpy as np
import scipy.sparse
```
```python
#helpers for integrating this notebook with libcbm
import notebook_startup
```

```python
from libcbm.wrapper.libcbm_wrapper import LibCBMWrapper
from libcbm.wrapper.libcbm_handle import LibCBMHandle
```

```python
settings = notebook_startup.load_settings()
dllpath = settings["libcbm_path"]
```

```python

def load_dll(config):
    dll = LibCBMWrapper(LibCBMHandle(dllpath, json.dumps(config)))
    return dll

def create_pools(names):
    return [{'name': x, 'id': i+1, 'index': i} for i,x in enumerate(names)]

def create_pools_by_name(pools):
    return {x["name"]: x for x in pools}

def to_coordinate(matrix):
    '''convert the specified matrix to a matrix of coordinate triples'''
    coo = scipy.sparse.coo_matrix(matrix)
    return np.column_stack((coo.row, coo.col, coo.data))
    
```

## ComputePools ##

This demonstrates the ComputePools core function of LibCBM.  


Given a matrix of pool values (where the columns represent specific pool, and the rows represent land units) Perform an arbitrary number of recursive vector matrix products.


```python
def ComputePools(pools, ops, op_indices):
    '''
    Runs the ComputePools libCBM function based on the specified numpy pool matrix, and the specified matrix ops.
    @param pools a matrix of pool values of dimension nstands by npools
    @param ops list of list of numpy matrices, the major dimension is nops, and the minor dimension may be jagged.
           Each matrix is of dimension npools by npools
    @param op_indices An nstands by nops matrix, where each column is a vector of indices to the jagged minor 
           dimension of the ops parameter
    '''
    pools = pools.copy()
    pooldef = create_pools([str(x) for x in range(pools.shape[1])])
    dll = load_dll({
        "pools": pooldef,
        "flux_indicators": []
    })
    op_ids = []
    for i,op in enumerate(ops):
        op_id = dll.AllocateOp(pools.shape[0])
        op_ids.append(op_id)
        #The set op function accepts a matrix of coordinate triples.  
        #In LibCBM matrices are stored in a sparse format, so 0 values can be omitted from the parameter
        dll.SetOp(op_id, [to_coordinate(x) for x in op], 
                  np.ascontiguousarray(op_indices[:,i]))
        
    dll.ComputePools(op_ids, pools)
    
    return pools
        
```

Introduction: A single pool/matrix operation 


```python
pools = np.ones((1,5))

mat = np.array(
    [[1,0.5,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

op_indices = np.array([[0]], dtype=np.uintp)
pools_test = ComputePools(pools,[[mat]], op_indices)

#create the expected result using the numpy implementation
pools_expected = np.matmul(pools, mat)

print("summed difference: {}".format((pools_expected-pools_test).sum()))
print("max difference: {}".format((pools_expected-pools_test).max()))

```

Scale up to a couple of pool vectors.  Multiply the single matrix by each pool vector.

```python
nstands = 10
npools = 5
nops = 1
pools = np.ones((nstands,npools))

#required to be a square mtrix of order n-pools
mat = np.array(
    [[1,0.5,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

op_indices = np.zeros((nops,nstands), dtype=np.uintp)
pools_test = ComputePools(pools,[[mat]], op_indices)


#create the expected result using the numpy implementation
pools_expected = np.zeros((10,5))
for i in range(nstands):
    pools_expected[i,:] = np.matmul(pools[i,:], mat)

print("summed difference: {}".format((pools_expected-pools_test).sum()))
print("max difference: {}".format((pools_expected-pools_test).max()))
```

Now scale up to a couple of matrices.

```python
nstands = 10
npools = 5
nops = 1
pools = np.ones((nstands,npools))

#required to be a square mtrix of order n-pools
mat0 = np.array(
    [[1,0.5,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mat1 = np.array(
    [[1,1,0,0,0],
     [0,1.0,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mats = [[mat0,mat1]]

op0_indices = [0,0,0,0,0,1,1,1,1,1] #the first 5 stands use mat0, and the sencond 5 use mat1
op_indices = np.transpose(np.array([op0_indices], dtype=np.uintp))
pools_test = ComputePools(pools, mats, op_indices)


#create the expected result using the numpy implementation
pools_expected = np.zeros((10,5))
for i in range(nstands):
    mat = mats[0][op_indices[i,0]]
    pools_expected[i,:] = np.matmul(pools[i,:], mat)

print("summed difference: {}".format((pools_expected-pools_test).sum()))
print("max difference: {}".format((pools_expected-pools_test).max()))
```

now expand to multiple ops

```python
nstands = 10
npools = 5
nops = 2
pools = np.ones((nstands,npools))

#matrices here are named mat_i_j where i is the op index, and j is the mat index
mat_0_0 = np.array(
    [[1,0.5,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mat_0_1 = np.array(
    [[1,1,0,0,0],
     [0,1.0,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mat_1_0 = np.array(
    [[1,0.5,0,0,0],
     [0,1,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mat_1_1 = np.array(
    [[1,1,0,0,0],
     [0,1.0,0,0,0],
     [0,0,1,0,0],
     [0,0,0,1,0],
     [0,0,0,0,1]])

mats = [[mat_0_0, mat_0_1],
        [mat_1_0, mat_1_1]]

op0_indices = [0,0,0,0,0,1,1,1,1,1] 
op1_indices = [0,0,0,0,0,1,1,1,1,1]
op_indices = np.transpose(np.array([op0_indices,op1_indices], dtype=np.uintp))
pools_test = ComputePools(pools, mats, op_indices)


#create the expected result using the numpy implementation
pools_expected = np.zeros((10,5))
pools_working = pools.copy() #working variable required
for i in range(nops):
    for k in range(nstands):
        mat = mats[i][op_indices[k,i]]
        pools_working[k,:] = np.matmul(pools_working[k,:], mat)
        
pools_expected = pools_working
print("summed difference: {}".format((pools_expected-pools_test).sum()))
print("max difference: {}".format((pools_expected-pools_test).max()))
```

randomized test versus np.matmul

```python
nstands = np.random.randint(1,1000+1)
npools = np.random.randint(3,25)
nops = np.random.randint(1,20)
pools = (np.random.rand(nstands,npools)-0.5)*1e15

mats = []
op_indices = np.zeros((nstands,nops), dtype=np.uintp)
for i in range(nops):
    n_op_mats = int(np.random.rand(1)[0]*nstands)
    if n_op_mats == 0:
        n_op_mats = 1
    op_indices[:,i] = np.floor((np.random.rand(nstands)*n_op_mats)).astype(np.uintp)
    op_mats = []
    for j in range(n_op_mats):
        op_mats.append(np.random.rand(npools,npools)) #create a random square matrix
    mats.append(op_mats)

pools_test = ComputePools(pools, mats, op_indices)

#create the expected result using the numpy implementation
pools_working = pools.copy() #working variable required
for i in range(nops):
    for k in range(nstands):
        mat = mats[i][op_indices[k,i]]
        pools_working[k,:] = np.matmul(pools_working[k,:], mat)

pools_expected = pools_working
print("mean difference: {}".format((pools_expected-pools_test).mean()))
print("summed difference: {}".format((pools_expected-pools_test).sum()))
print("max difference: {}".format((pools_expected-pools_test).max()))
print("all close [rtol=1e-12, atol=1e-15]: {}".format(np.allclose(pools_expected,pools_test,rtol=1e-12, atol=1e-15)))
```

## Flux Indicators ##

```python
def create_flux_indicator(pools_by_name, process_id, sources, sinks):
    return {
       'id': None,
       'index': None,
       'process_id': process_id,
       'source_pools': [pools_by_name[x]["id"] for x in sources],
       'sink_pools': [pools_by_name[x]["id"] for x in sinks]
   }

def append_flux_indicator(collection, flux_indicator):
    flux_indicator["index"] = len(collection)
    flux_indicator["id"] = len(collection)+1
    collection.append(flux_indicator)

def ComputeFlux(pools, poolnames, ops, op_indices, op_processes, flux_indicators):
    pools = pools.copy()
    flux = np.zeros((pools.shape[0], len(flux_indicators)))
    pooldef = create_pools([poolnames[x] for x in range(pools.shape[1])])
    pools_by_name = create_pools_by_name(pooldef)
    fi_collection = []
    for f in flux_indicators:
        fi = create_flux_indicator(pools_by_name, f["process_id"], f["sources"], f["sinks"])
        append_flux_indicator(fi_collection, fi)
    dll = load_dll({
        "pools": pooldef,
        "flux_indicators": fi_collection
    })
    op_ids = []
    for i,op in enumerate(ops):
        op_id = dll.AllocateOp(pools.shape[0])
        op_ids.append(op_id)
        dll.SetOp(op_id, [to_coordinate(x) for x in op], 
                  np.ascontiguousarray(op_indices[:,i]))
        
    dll.ComputeFlux(op_ids, op_processes, pools, flux)
    return pools, flux

```

```python

fi = [
    #with this flux indicator, we are capturing all flows from the "a" pool to any of the other pools
    {"process_id": 1, "sinks": ["b","c","d","e"], "sources": ["a"]},
    #and with this one, we are capturing all flows to the "a" pool from any of the other pools
    {"process_id": 2, "sinks": ["a"], "sources": ["b","c","d","e"]},
    {"process_id": 2, "sinks": ["a"], "sources": ["a"]},
    {"process_id": 3, "sinks": ["d","e"], "sources": ["b","c"]},
]
unique_process_ids = {x["process_id"] for x in fi} 
poolnames = ["a","b","c","d","e"]
pool_index = {x:i for i,x in enumerate(poolnames)}
nstands = np.random.randint(1,1000+1)
npools = len(poolnames)
nops = np.random.randint(1,20)
pools = (np.random.rand(nstands,npools)-0.5)*0.1

I = np.identity(npools)
mats = []
op_indices = np.zeros((nstands,nops), dtype=np.uintp)
for i in range(nops):
    n_op_mats = int(np.random.rand(1)[0]*nstands)
    if n_op_mats == 0:
        n_op_mats = 1
    op_indices[:,i] = np.floor((np.random.rand(nstands)*n_op_mats)).astype(np.uintp)
    op_mats = []
    for j in range(n_op_mats):
        op_mats.append(np.random.rand(npools,npools)) #create a random square matrix
    mats.append(op_mats)

#evenly assigns ops to the defined process ids
op_processes = [x%len(unique_process_ids)+1 for x in range(nops)] 
pools_test,flux_test = ComputeFlux(pools, poolnames, mats, op_indices, op_processes, fi)

#create the expected result using the numpy implementation
#this fully emulates the ComputeFlux function, and computes an 
#independent result against which we check differences

pools_working = pools.copy() #working variable required
flux_expected = np.zeros((nstands,len(fi)))
for i in range(nops):
    for k in range(nstands):
        mat = mats[i][op_indices[k,i]]
        flux = np.matmul(np.diag(pools_working[k,:]), (mat-I))
        for i_f, f in enumerate(fi):
            process_id = op_processes[i]
            if(f["process_id"] != process_id):
                continue
            for src in f["sources"]:
                for sink in f["sinks"]:
                    flux_expected[k,i_f] += flux[pool_index[src],pool_index[sink]]
        pools_working[k,:] = np.matmul(pools_working[k,:], mat)

pools_expected = pools_working
print("pool mean difference: {}".format((pools_expected-pools_test).mean()))
print("pool summed difference: {}".format((pools_expected-pools_test).sum()))
print("pool max difference: {}".format((pools_expected-pools_test).max()))
print("pool allclose[rtol=1e-12, atol=1e-15]: {}".format(np.allclose(pools_expected,pools_test,rtol=1e-12, atol=1e-15)))

print("flux mean difference: {}".format((flux_expected-flux_test).mean()))
print("flux summed difference: {}".format((flux_expected-flux_test).sum()))
print("flux max difference: {}".format((flux_expected-flux_test).max()))
print("flux allclose[rtol=1e-12, atol=1e-15]: {}".format(np.allclose(flux_expected,flux_test,rtol=1e-12, atol=1e-15)))
```
