


def skyhook_calculator(upper_vel,delta_vel):
    Cmax = 4000
    Cmin = 300
    epsilon = 0.0001
    alpha = 0.5
    sat_limit = 800
    if upper_vel * delta_vel >= 0:
        C = (alpha * Cmax * upper_vel + (1 - alpha) * Cmax * upper_vel)/(delta_vel + epsilon)
        C = min(C,Cmax)
        u = C * delta_vel
    else:
        u = Cmin*delta_vel
    
    if u >= 0:
        if u > sat_limit:
            u_ = sat_limit
        else:
            u_ = u
    else:
        if u < -sat_limit:
            u_ = -sat_limit
        else:
            u_ = u
    return u_
    
def skyhook(env):
                
    # env.state_SH = [fl2,tfl2,fr2,tfr2,rl2,trl2,rr2,trr2]
    dz_fl = env.state_SH[0]
    vel_fl = dz_fl - env.state_SH[1]

    u_fl = skyhook_calculator(dz_fl,vel_fl)

    if len(env.state_SH) > 2:

        dz_fr = env.state_SH[2]
        dz_rl = env.state_SH[4]
        dz_rr = env.state_SH[6]

        vel_fr = dz_fr - env.state_SH[3]
        vel_rl = dz_rl - env.state_SH[5]
        vel_rr = dz_rr - env.state_SH[7]

        
        u_fr = skyhook_calculator(dz_fr,vel_fr)
        u_rl = skyhook_calculator(dz_rl,vel_rl)
        u_rr = skyhook_calculator(dz_rr,vel_rr)
        
        return [u_fl,u_fr,u_rl,u_rr]
    else:
        return u_fl