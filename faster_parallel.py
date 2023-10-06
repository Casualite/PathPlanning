import numba as nb
import numpy as np
import cupy as cp
@nb.njit('uint32(int32)')
def hash_32bit_4k(value):
    return (np.uint32(value) * np.uint32(27_644_437)) & np.uint32(0x0FFF)

@nb.njit(['int32[:](int32[:], int32[:])', 'int32[:](int32[::1], int32[::1])'], parallel=True)
def set_difference(arr1, arr2):
    # Pre-computation of the bloom-filter
    bloomFilter = np.zeros(4096, dtype=np.uint8)
    for j in range(arr2.size):
        bloomFilter[hash_32bit_4k(arr2[j])] = True

    chunkSize = 1024 # To tune regarding the kind of input
    chunkCount = (arr1.size + chunkSize - 1) // chunkSize

    # Find for each item of `arr1` if the value is in `arr2` (parallel)
    # and count the number of item found for each chunk on the fly.
    # Note: thanks to page fault, big parts of `found` are not even written in memory if `arr2` is small
    found = np.zeros(arr1.size, dtype=nb.bool_)
    foundCountByChunk = np.empty(chunkCount, dtype=nb.uint16)
    for i in nb.prange(chunkCount):
        start, end = i * chunkSize, min((i + 1) * chunkSize, arr1.size)
        foundCountInChunk = 0
        for j in range(start, end):
            val = arr1[j]
            if bloomFilter[hash_32bit_4k(val)] and val in arr2:
                found[j] = True
                foundCountInChunk += 1
        foundCountByChunk[i] = foundCountInChunk

    # Compute the location of the destination chunks (sequential)
    outChunkOffsets = np.empty(chunkCount, dtype=nb.uint32)
    foundCount = 0
    for i in range(chunkCount):
        outChunkOffsets[i] = i * chunkSize - foundCount
        foundCount += foundCountByChunk[i]

    # Parallel chunk-based copy
    out = np.empty(arr1.size-foundCount, dtype=arr1.dtype)
    for i in nb.prange(chunkCount):
        srcStart, srcEnd = i * chunkSize, min((i + 1) * chunkSize, arr1.size)
        cur = outChunkOffsets[i]
        # Optimization: we can copy the whole chunk if there is nothing found in it 
        if foundCountByChunk[i] == 0:
            out[cur:cur+(srcEnd-srcStart)] = arr1[srcStart:srcEnd]
        else:
            for j in range(srcStart, srcEnd):
                if not found[j]:
                    out[cur] = arr1[j]
                    cur += 1
    return out

@nb.njit(['int32[:](int32[:])','uint32[:](uint32[:,:])'])
def unique(arr):
    return(np.unique(arr))

