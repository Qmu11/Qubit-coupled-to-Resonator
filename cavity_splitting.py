"""
In this we simuate the steady state response of resonator under a constant microwave drive.
when we sweep the drive frequency and detuning of resonator and qubit, we see that the resonant 
peak splits due to the coupling betwen the systems  

"""

from mpi4py import MPI
import numpy as np
from qutip import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N   = 10          
wr = 6.0*2*np.pi
g1 = 0.1 * 2 * np.pi       
  
N = 15                     
tl1= 1e6                                 
tp1 = 1e3                                    
k = 0.001 * 2 * np.pi        
e = 0.35*2*np.pi
ei = e/(np.sqrt(2))
eq = e/(np.sqrt(2))


sm = tensor(destroy(2), qeye(N))
sz = tensor(sigmaz(),   qeye(N))
a  = tensor(qeye(2),    destroy(N))

c_ops = [
    np.sqrt(1/tl1)    * sm,
    np.sqrt(1/(2*tp1)) * sz,
    np.sqrt(k)         * a,
]

a_dag_a = a.dag() * a   

def H_build(w1, wd):
    return (
          (w1 - wd) * sm.dag() * sm
        + (wr - wd) * a_dag_a
        + g1 * (sm * a.dag() + sm.dag() * a)
        + (ei / 2)      * (a + a.dag())
        + 1j*(eq / 2)   * (a.dag() - a)
    )


w1_values = np.linspace(5.8, 6.4, 100)*2*np.pi
wd_values = np.linspace(5.8, 6.4, 100)*2*np.pi

local_w1_indices = np.array_split(np.arange(len(w1_values)), size)[rank]


local_rows = {}
for i in local_w1_indices:
    w1  = w1_values[i]
    row = []
    for wd in wd_values:
        H2  = H_build(w1, wd)
        rho = steadystate(H2, c_ops)
        n   = expect(a_dag_a, rho)
        row.append(n)
    local_rows[i] = row
    if rank == 0:
        print(f"Row {i+1}/{len(w1_values)} done", flush=True)


all_rows = comm.gather(local_rows, root=0)


if rank == 0:
    matrix = [None] * len(w1_values)
    for rank_dict in all_rows:
        for i, row in rank_dict.items():
            matrix[i] = row
    matrix = np.array(matrix)
    np.savetxt("matrix.csv", matrix, delimiter=",")

   
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
    matrix.T,                    
    origin="lower",
    aspect="auto",
    extent=[
        w1_values[0]  / (2*np.pi),
        w1_values[-1] / (2*np.pi),
        wd_values[0]  / (2*np.pi),
        wd_values[-1] / (2*np.pi),
    ],
    cmap="viridis",
    vmax=0.01
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\langle a^\dagger a \rangle$")
    ax.set_xlabel(r"$\omega_1 / 2\pi$ (GHz)")
    ax.set_ylabel(r"$\omega_d / 2\pi$ (GHz)")
    ax.set_title("Steady-state photon number")
    fig.tight_layout()
    fig.savefig("matrix.png", dpi=150)
    print("Saved matrix.png")
    

    