

import os
import torch
from module import *
import sqlite3
import itertools
from datetime import datetime, timedelta

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

cur.execute('''Select model_file_path, Nz, 
    Nlayer, Nz, Nlayer, Nh, Nbatch, Nitr from Result
    where id >= 65
    ''')
for (model_file_path, Nz, Nlayer, Nz, Nlayer, Nh, Nbatch, Nitr) in cur.fetchall():

    x_generator = XGenerator(Nz, Nx_trg, Nlayer, Nh)
    x_generator.load_state_dict(torch.load(model_file_path))

    z_generator_ = z_generator(2**12, Nz)
    
    Z = z_generator_.__next__()
    _Z = torch.from_numpy(Z) # (*, Nz)

    _X0 = x_generator(_Z) # (*, Nx)
    X0 = _X0.data.numpy() 
    X1 = operate_poincare_map(X0)
    _X1 = torch.from_numpy(X1) # (*, Nx)
    X1 = _X1.data.numpy()

    model_name = os.path.splitext(os.path.basename(model_file_path))[0]
    print(model_name)

    plt.figure()
    plt.plot(X0[:,0], X0[:,1], '.', label = "X0")
    plt.plot(X1[:,0], X1[:,1], '.', label = "X1")
    plt.legend()
    plt.savefig("./tmp/%s_poincaremap.png" % model_name)
    plt.close()


