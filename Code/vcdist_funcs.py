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
    where_f = np.where(inst2==1)
    
    for ii in range(len(where_f[0])):
        nn1 = where_f[0][ii]
        nn2 = where_f[1][ii]
        nn3 = where_f[2][ii]
        if deform:
            ww_min = max(0,nn2-1)
            ww_max = min(vc_dim[0],nn2+1)
            hh_min = max(0,nn3-1)
            hh_max = min(vc_dim[1],nn3+1)
            
            if inst1[nn1, ww_min:ww_max+1, hh_min:hh_max+1].sum()==0:
                dis_cnt += 1
        else:
            if inst1[nn1,nn2,nn3]==0:
                dis_cnt += 1
    
    return (len(where_f[0]), dis_cnt)

def comp_one_to_ls(inst1, ls2, deform):
    inst_num2 = len(ls2)
    rst = np.zeros(inst_num2)
    for inst_nn2 in range(inst_num2):
        inst2 = ls2[inst_nn2]
        n1, n2 = vc_dis(inst1,inst2, deform)
        rst1 = n2/n1
        n1, n2 = vc_dis(inst2,inst1, deform)
        rst2 = n2/n1
        rst[inst_nn2] = min(rst1, rst2)
        
    return rst


def comp_two_ls(ls1, ls2, deform):
    inst_num1 = len(ls1)
    inst_num2 = len(ls2)
    mat = np.zeros((inst_num1,inst_num2))
    for inst_nn1 in range(inst_num1):
        mat[inst_nn1, :] = comp_one_to_ls(ls1[inst_nn1], ls2, deform)
       