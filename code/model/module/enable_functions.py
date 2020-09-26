
__all__ = ['sparse_output','q_enable']

def sparse_output(m):
    ''' output sparse ration function apply to module
    '''
    if hasattr(m, 'running_zero_ratio'):
        print("    aSparse Ratio: ", m.running_zero_ratio)


def q_enable(m):
    ''' enable function to apply to module
    '''
    if hasattr(m, 'quantize'):
        m.quantize = True
        print(" @  Enable quantize of")
        print(m)

