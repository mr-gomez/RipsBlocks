import numpy as np

# Functions to write a numpy matrix into a nexus file

# Create a format label to display a non-negative matrix dm
# with a fixed number of decimal places (digits)
# and pads smaller numbers to match the number of characters
# in a larger number
# Expected result with default paraters if dm.type==float and
# np.max(dm) < 100: format_str='{:.2f}'
def format_matrix(dm, digits=2, int_digits='none'):
    if int_digits == 'auto':
        # Find the number of digits of the largest entry
        M = np.max(dm)
        int_digits = np.floor(np.log10(M))

        # Need to add 1 because np.log10(100)=2
        # but 100 has 3 digits
        int_digits = int(int_digits)+1
    else:
        int_digits = 0

    # If dm has integer type, we don't care about decimals
    if dm.dtype == int:
        digits = 0
    
    
    # Write format string
    total_digits = int_digits+digits+1
    format_str='{:'+ str(total_digits) + '.' + str(digits) + 'f}'
    return format_str

def dm_to_nexus(dm, file_name, digits=2, folder='.'):
    nNodes = dm.shape[0]
    
    # Create file
    file = open(f'{folder}/{file_name}.nex', 'w')

    # Headers
    file.write('#nexus\n\n')
    file.write('BEGIN TAXA;\n')
    file.write(f'DIMENSIONS NTAX={nNodes};\n')

    # List of taxa
    # I just write the index for now
    file.write('TAXLABELS\n')
    for i in range(nNodes):
        file.write(f'[{i+1}] \'{i+1}\'\n')
    file.write(';\nEND; [TAXA]\n\n')

    # Distances headers
    file.write('BEGIN DISTANCES;\n')
    file.write(f'DIMENSIONS NTAX={nNodes};\n')
    file.write('FORMAT TRIANGLE=both LABELS=no;\n')
    file.write('MATRIX\n')

    # Write dm as text
    format_str = format_matrix(dm, digits=digits)
    for idx in range(nNodes):
        # Write one row at a time
        text = f'[{idx+1}] '
        for jdx in range(nNodes):
            text += format_str.format(dm[idx,jdx])
            text += ' '
        
        # Write row to file
        file.write(text[:-1]+'\n')
    
    # Close file
    file.write(';\nEND; [DISTANCES]')
    file.close()