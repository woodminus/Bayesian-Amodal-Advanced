import numpy as np

def vc_dis(inst1, inst2, deform):
    inst1 = inst1[:,:,0:14]
    inst2 = inst2[:,:,0:14]
    hh1 = inst1.shape[1]
    hh2 = inst2.shape[1]
    if hh1 > hh2:
        diff = hh1 - hh2
        diff_top = int(diff/2)
        diff_bottom = diff - diff_top
        inst2 = np.concatenate([np.zeros((inst2.shape[0], diff_top, inst2.shape[2])), inst2], axis=1)
        inst2 = np.concatenate([inst2, np.zeros((inst2.shape[0], diff_bottom, inst2.shape[2]))], axis=1)
    elif hh1 < hh2:
        diff = hh2 - hh1
        diff_top = int(diff/2)
        diff_bottom = hh2 - (diff - diff_top)
        inst2 = inst2[:,diff_top: diff_bottom,:]
        
    vc_dim = (inst1.shape[1], inst1.shape[2])
    dis_cnt = 0
