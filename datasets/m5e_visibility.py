"""Offline candidate planar-background audit. NEVER authorizes negative labels."""
import numpy as np


def planar_patch(depth_patch, semantic_patch, instance_patch, *, tolerance_m=.05):
    """Perspective plane: inverse optical depth affine in pixel coordinates.

    Require one identity AND semantic across 5x5, all finite/in-range, and
    <=5cm max reconstructed-depth residual. Both rings and central pixel enter
    the max residual: discontinuities cannot be hidden by an average fit.
    A coplanar same-ID unobserved thin surface remains unprovable: Candidate only.
    """
    d=np.asarray(depth_patch,float)
    if d.shape[-2:]!=(5,5):raise ValueError('Expected [...,5,5]')
    shape=d.shape[:-2];flat=d.reshape(-1,25)
    s=np.asarray(semantic_patch).reshape(-1,25);i=np.asarray(instance_patch).reshape(-1,25)
    finite=np.isfinite(flat).all(-1)&(flat>=1.5).all(-1)&(flat<80).all(-1)
    identity=(s==s[:,12,None]).all(-1)&(i==i[:,12,None]).all(-1)
    y,x=np.mgrid[-2:3,-2:3];A=np.stack((x.ravel(),y.ravel(),np.ones(25)),1)
    inv=1/np.where(np.isfinite(flat)&(flat>0),flat,1)
    beta=inv@np.linalg.pinv(A).T;predinv=beta@A.T
    pred=1/np.maximum(predinv,1e-9)
    residual=np.max(np.abs(pred-flat),-1)
    accepted=finite&identity&(predinv>0).all(-1)&(residual<=tolerance_m)
    return dict(candidate=accepted.reshape(shape),identity_uniform=identity.reshape(shape),finite=finite.reshape(shape),residual_m=residual.reshape(shape),range_m=np.ptp(flat,-1).reshape(shape),inverse_plane_coefficients=beta.reshape(*shape,3))


def patches(array,uv):
    uv=np.asarray(uv);u,v=uv.T;offset=np.arange(-2,3)
    return array[np.clip(v[:,None,None]+offset[None,:,None],0,array.shape[0]-1),np.clip(u[:,None,None]+offset[None,None,:],0,array.shape[1]-1)]
