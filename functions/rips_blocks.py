import numpy as np
from ripser import ripser
import functions.nexusIO as nexusIO
import subprocess

# vectors = m*n array, where each row is a vector in R^n
# returns = m*m matrix dm_inf where dm_inf[i,j] = L^inf distance between
# rows i and j of vectors
def L_inf_distance(vectors):
    if vectors.shape[0] == 0:
        return np.zeros((0,0))
    
    vec_reshape = np.transpose(
        np.expand_dims(vectors, axis=2),
        (0,2,1)
    )
    
    # vectors: m*n (broadcast to 1*m*n)
    # vec_reshape: m*1*n
    # Then:
    # diff[i,j,:] = vectors[i,:] - vectors[j,:]
    diff = vec_reshape - vectors

    # dm_inf[i,j] = L^infty(cut_vertices[i,:] - cut_vertices[j,:])
    dm_inf = np.linalg.norm(diff, ord=np.inf, axis=2)

    return dm_inf

def block_dec(infile, folder='.', outfile=None, verbose=False, taxa=False,
              inputmetric=False, blocksplits=False, cutpoints=True,
              blocks=False, partialmetrics=False, blockmetrics=False):
    if outfile is None:
        outfile = f'{infile}_blocks'
    command = f'java -client -jar functions/BloDec.jar infile={folder}/{infile}.nex outfile={folder}/{outfile}.nex'

    if verbose:
        command += ' verbose=YES'
    if taxa:
        command += ' taxa=YES'
    if inputmetric:
        command += ' inputmetric=YES'
    if blocksplits:
        command += ' blocksplits=YES'
    if cutpoints:
        command += ' cutpoints=YES'
    else:
        command += ' cutpoints=NO'
    if blocks:
        command += ' blocks=YES'
    if partialmetrics:
        command += ' partialmetrics=YES'
    if blockmetrics:
        command += ' blockmetrics=YES'
    
    subprocess.run(command, shell=True)

def read_block_nexus(file_name, folder='.'):
    cut_vertices = []
    blocks = []

    # Create file
    full_name = f'{folder}/{file_name}_blocks.nex'

    loading_cuts = False
    loading_blocks = False

    with open(full_name) as file:
        for line in file:
            if line.rstrip()=='BEGIN CUTPOINTS;':
                loading_cuts = True
            
            if line.rstrip()=='BEGIN BLOCKS;':
                loading_blocks = True

            if loading_cuts:
                # Reached the end of the cutpoints
                if line.rstrip() == 'END;':
                    loading_cuts = False
                
                # Extract the number of real and virtual cutpoints from the
                # line with size information
                if line.startswith('DIMENSIONS'):
                    # print(line.rstrip())
                    
                    # line format:
                    # DIMENSIONS NTAX=# NCUTPOINTS=#;
                    # line_split = ['DIMENSIONS', 'NTAX=#', 'NCUTPOINTS=#']
                    line_split = line.rstrip()[:-1].split(' ')
                    taxa_info = line_split[1].split('=')
                    N = int(taxa_info[1])

                    cut_info = line_split[2].split('=')
                    total_size = int(cut_info[1])

                # For rows that have numerical values
                if line.startswith('['):
                    # print(line.rstrip())

                    # line format:
                    # [i]   d1 d2 d3 ... dn,\n
                    # where i is the index of the point and d1, ... are distances
                    # line_split = ['i', '   d1 d2 d3 ... dn,']
                    line_split = line[1:].rstrip().split(']')

                    # Load cut vertices that didn't exist in the metric space
                    cut_idx = int(line_split[0])
                    if cut_idx > N:
                        line_num = line_split[1]

                        # Remove whitespace from start and comma from end of line
                        line_num = line_num.lstrip()
                        line_num = line_num[:-1]

                        # Convert line into vector
                        cut_vertices.append(line_num.split(' '))
            
            if loading_blocks:
                # Reached the end of the blocks
                if line.rstrip() == 'END;':
                    loading_blocks = False
                # print(line.rstrip())

                # For rows that have numerical values
                if line.startswith('['):
                    # line format:
                    # [i size=#]   p1 p2 p3 ... pn,\n
                    # where i is the index of the block and d1, ... are distances
                    # line_split = ['i size=#', '   d1 d2 d3 ... dn,']
                    line_split = line.split(']')
                    line_num = line_split[1]

                    # Remove whitespace and comma from end of line
                    line_num = line_num.strip()
                    line_num = line_num[:-1]
                    
                    # Convert line_num to a np.array
                    line_num = np.array(line_num.split(' '), dtype=int)

                    # Change to 0-indexing
                    line_num -= 1
                    
                    # Add block
                    blocks.append(line_num)

    # Convert cut_vertices into an array
    cut_vertices = np.array(cut_vertices, dtype=float)

    return blocks, cut_vertices

def separate_virtual_points(N, n_cut_vertices, blocks):
    # Divide blocks into real points and virtual cut vertices
    cut_indices = np.arange(N, N+n_cut_vertices)
    # print(cut_indices)
    # print()

    B_real = []
    B_virtual = []

    for B in blocks:
        I_cuts = np.in1d(B, cut_indices)
        # print(B)
        # print(B[~I_cuts])
        # print(B[I_cuts])
        # print()

        B_real.append(B[~I_cuts])

        # Change cut vertices to have 0-indexing
        B_virtual.append(B[I_cuts]-N)
    
    # print('B_real, B_virtual done')
    # print()

    return B_real, B_virtual

def block_cut_dict(B_virtual, n_cut_vertices):
    # Create a dict of the blocks that contain any given cut vertex
    blocks_with_cut = {}
    for cut in range(n_cut_vertices):
        blocks_with_c = []
        for idb, B in enumerate(B_virtual):
            if cut in B:
                blocks_with_c.append(idb)

        blocks_with_cut[cut] = blocks_with_c
    
    return blocks_with_cut

def replace_virtual_points_recursive(N, cut_0, min_dist_prev, cuts_prev, blocks_prev, B_real, B_virtual, cut_vertices, blocks_with_cut, dm_cuts):
    # print('Start recursion (cuts, blocks)')
    # print(cuts_prev)
    # print(blocks_prev)
    
    n_cuts_prev = len(cuts_prev)

    # For every pair (c_prev,B_prev), look for all real points in the blocks B
    # that contain c_prev and are different from B_prev
    real_new = []
    for idc in range(n_cuts_prev):
        cut_prev = cuts_prev[idc]
        B_prev = blocks_prev[idc]

        # Join the real vertices of all blocks B2 that contain cut_prev
        # and are different from B_prev
        for B in blocks_with_cut[cut_prev]:
            if B != B_prev:
                # Join all real vertices
                real_new.extend(B_real[B])

    # If real_new is not empty, find a point that minimizes the distance
    # to the original cut vertex
    min_dist = min_dist_prev
    if len(real_new) > 0:
        real_dists = cut_vertices[cut_0, real_new]
        min_dist = np.min(real_dists)
        
        # Any point that realizes the minimum will do
        real_idx = np.where(real_dists == min_dist)[0][0]
        vertex_final = real_new[real_idx]
    # If real_new is empty, we don't update min_dist
    else:
        min_dist = np.inf
    # Update min_dist
    min_dist = np.minimum(min_dist, min_dist_prev)
    
    # Now look for new virtual points that are closer than min_dist to
    # the original cut vertex
    # Remember: for every (cut_prev, B_prev) combination, we search on
    # blocks that contain cut_prev that are not B_prev
    virtual_new = []
    blocks_new = []
    for idc in range(n_cuts_prev):
        cut_prev = cuts_prev[idc]
        B_prev = blocks_prev[idc]

        for B in blocks_with_cut[cut_prev]:
            if B != B_prev:
                B_virt_idx = [c for c in B_virtual[B] if c != cut_prev]
                B_dists = dm_cuts[cut_prev, B_virt_idx]
                
                # Keep cuts with distance < min_dist
                B_search = np.where(B_dists < min_dist)[0]

                virtual_new.extend([B_virt_idx[t] for t in B_search])
                blocks_new.extend([B]*B_search.shape[0])

    # Repeat the search over the next set of virtual points (if there is any)
    if len(virtual_new)>0:
        vertex_recursive, min_dist_recursive = replace_virtual_points_recursive(N, cut_0, min_dist, virtual_new, blocks_new, B_real, B_virtual, cut_vertices, blocks_with_cut, dm_cuts)
    # If not, set a flag
    else:
        vertex_recursive = -1

    # Note that min_dist_recursive <= min_dist by construction. If the
    # recursion found a point that realizes the minimum, we don't need
    # to keep searching
    if vertex_recursive != -1:
        return vertex_recursive, min_dist_recursive
    
    # If the recursion didn't find a suitable point (vertex_recursive == -1)
    # and we didn't find a good candidate in the first search, we return -1
    elif vertex_recursive == -1 and len(real_new) == 0:
        return -1, min_dist
    # But if we did find a suitable point, we return it
    else:
        return vertex_final, min_dist

def replace_virtual_points(N, cut_vertices, B_real, B_virtual, blocks_with_cut, dm_cuts):
    n_blocks = len(B_real)

    # print('Cut vertices:', cut_indices)
    # print('Number of cut vertices:', cut_vertices.shape[0])
    # print('Length of entries:', cut_vertices.shape[1])
    
    # For each cut-point c of a block B, use the Kuratowski embedding of c
    # to find x_{B,c}.
    # x_{B,c} is the closest point to c in any block B' that is separated
    # from B by c
    B_cut_real = []
    for B_0 in range(n_blocks):
        B_cut_real_row = []
        for cut_0 in B_virtual[B_0]:
            # print(idb, cut_0)
            
            # Search recursively over the blocks that are separated from B
            # by cut
            new_vertex, _ = replace_virtual_points_recursive(N, cut_0, np.inf, [cut_0], [B_0], B_real, B_virtual, cut_vertices, blocks_with_cut, dm_cuts)
            B_cut_real_row.append(new_vertex)
        
        B_cut_real.append(B_cut_real_row)
    
    return B_cut_real

# Returns a list of connected components. Each element is a list of the blocks
# in a single connected component of T(X) - cut_0. Optionally, we may exclude
# the component with a block B_exclude
def components_without_cut(cut_0, blocks_with_cut, B_virtual, B_exclude=None):
    comps_all = []

    # Search over all blocks that contain cut_0
    for B in blocks_with_cut[cut_0]:
        if B == B_exclude:
            continue

        # Search in all cut vertices of B
        comps_B = [B]
        for cut in B_virtual[B]:
            if cut == cut_0:
                continue

            # Look for the connected components in further blocks and join them all
            comps = components_without_cut(cut, blocks_with_cut, B_virtual, B_exclude=B)
            for C in comps:
                comps_B.extend(C)
        
        # Add the connected component to the global list
        comps_all.append(comps_B)

    return comps_all

def replace_virtual_points_v2(cut_vertices, B_real, B_virtual, blocks_with_cut):
    n_blocks = len(B_real)
    
    # For each cut-point c of a block B, use the Kuratowski embedding of c
    # to find x_{B,c}.
    # x_{B,c} is the closest point to c in any block B' that is separated
    # from B by c
    B_cut_real = []
    for B_0 in range(n_blocks):
        B_cut_real_row = []
        for cut_0 in B_virtual[B_0]:
            # Get the connected components of T(X) - cut_0
            comps_cut = components_without_cut(cut_0, blocks_with_cut, B_virtual, B_exclude=B_0)

            # Search over the connected components without B_0
            # I need to join all points in these blocks
            candidates = []
            for comp in comps_cut:
                for B in comp:
                    candidates.extend(B_real[B])
            
            # Select the candidate with lowest distance to the cut vertex
            # We can pick any point that minimizes the distance
            dist_cut_0 = cut_vertices[cut_0, candidates]
            min_dist = np.min(dist_cut_0)
            I = np.where(dist_cut_0 == min_dist)[0][0]
            
            B_cut_real_row.append(candidates[I])
        
        B_cut_real.append(B_cut_real_row)
    
    return B_cut_real

# Create the distance matrices of \overline{X}_B for every block B
# For every block B and cut vertex c \in B, I need to find the elements
# x_{B,c}. There are two cases.
# 1. If the cut vertex comes from the original metric space, then x_{B,c}
#    is itself
# 2. If not, then I look for all blocks that contain c. In those blocks,
#    I find the point that has the smallest distance to B, and set x_{B,c}
#    as that point.
def find_dm_of_blocks(dm, B_real, B_cut_real):
    dm_blocks = []
    for idb in range(len(B_real)):
        I_block = B_real[idb].tolist()
        I_block.extend(B_cut_real[idb])
        
        dm_B = dm[I_block,:]
        dm_B = dm_B[:,I_block]
        dm_blocks.append(dm_B)
    
    return dm_blocks

# Applies a consistent ordering to a 2D vector with persistence diagrams
def sort_diagrams(diagrams):
    # Sort diagrams
    I_sort = np.lexsort((diagrams[:,1],-diagrams[:,0]))
    return diagrams[I_sort, :]

def ripser_with_blocks(dm, file_name='dm', folder='temp', maxdim=1, **kwargs):
    # block_start = time()

    N = dm.shape[0]

    # Save distance matrix to a nexus file
    # start = time()
    nexusIO.dm_to_nexus(dm, file_name, folder=folder)
    # end = time()
    # print('Saving nexus:', display_time(end-start))

    # Compute block decomposition by calling BloDec
    # start = time()
    block_dec(file_name, folder=folder, blocks=True, cutpoints=True)
    # end = time()
    # print('Compute block decomposition:', display_time(end-start))
    
    # Read block information from the nexus file
    # start = time()
    blocks, cut_vertices = read_block_nexus(file_name, folder=folder)
    n_cut_vertices = cut_vertices.shape[0]
    # end = time()
    # print('Reading block decomposition:', display_time(end-start))

    # Compute distance matrix of cut vertices
    # start = time()
    # dm_cuts = L_inf_distance(cut_vertices)
    # end = time()
    # print('Distance between cut vertices:', display_time(end-start))

    # Find the virtual cut points in every block
    # start = time()
    B_real, B_virtual = separate_virtual_points(N, n_cut_vertices, blocks)
    # end = time()
    # print('Separate virtual and real points:', display_time(end-start))
    
    # Create a dictionary {c:[blocks that contain c] for every cut-vertex c}
    # start = time()
    blocks_with_cut = block_cut_dict(B_virtual, n_cut_vertices)
    # end = time()
    # print('Block-cut dict:', display_time(end-start))

    # Replace virtual cut points with x_{B,c}
    # sep_start = time()
    # B_cut_real = replace_virtual_points_v2(N, cut_vertices, B_real, B_virtual, blocks_with_cut, dm_cuts)
    # sep_end = time()
    # print('Replace virtual points:', display_time(sep_end-sep_start))

    B_cut_real = replace_virtual_points_v2(cut_vertices, B_real, B_virtual, blocks_with_cut)
    
    # Construct distance matrix of every space \overline{X}_B
    # start = time()
    dm_blocks = find_dm_of_blocks(dm, B_real, B_cut_real)
    # end = time()
    # print('Construct dm of blocks:', display_time(end-start))

    # block_end = time()
    # print('Block decomposition time:', display_time(block_end-block_start))
    # print()

    # ph_start = time()
    # Compute the persistent homology of every block
    dgms_blocks = []
    for M in dm_blocks:
        diagrams = ripser(M, distance_matrix=True, maxdim=maxdim, **kwargs)['dgms']
        dgms_blocks.append(diagrams)

    # Compute 0-dim persistence
    dgm0 = ripser(dm, maxdim=0, distance_matrix=True, **kwargs)['dgms']
    dgm0 = dgm0[0]

    # Join and sort diagrams
    dgms_all = [dgm0]
    for dim in range(1,maxdim+1):
        dgm_all = np.zeros((0,2))

        for dgm_B in dgms_blocks:
            dgm_all = np.concatenate((dgm_all, dgm_B[dim]), axis=0)

        # Append sorted diagrams
        dgms_all.append(sort_diagrams(dgm_all))
    
    # ph_end = time()
    # print('PH:', display_time(ph_end - ph_start))
    # print()

    return dgms_all