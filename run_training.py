
from module import *
import sqlite3
import itertools
from datetime import datetime, timedelta

conn = sqlite3.connect('db.sqlite')
cur = conn.cursor()

lst_Nz  = [4,] # 4
lst_Nlayer= [4,] # 5
lst_Nh= [2**3, ] # 6
lst_Nbatch= [2**6,] # 3
lst_Nitr= [2**14,]
lst_eps = [1e-1, 1e-2, ] # 1
lst_max_sink_itr = [2**5, 2**7, 2**9,] # 2

t_bgn = datetime.now()
t_wait = 2 * 60 * 60 
Nitr_phase = 2**10

for Nz, Nlayer, Nh, Nbatch, Nitr, eps, max_sink_itr in \
    itertools.cycle(itertools.product(lst_Nz, lst_Nlayer, \
    lst_Nh, lst_Nbatch, lst_Nitr, lst_eps, lst_max_sink_itr)):
    print(datetime.now() - t_bgn)

    if datetime.now() - t_bgn > timedelta(seconds = t_wait):
        break

    x_generator = XGenerator(Nz, Nx_trg, Nlayer, Nh)
    z_generator_ = z_generator(Nbatch, Nz)
    optimizer = torch.optim.Adam(x_generator.parameters())

    Nitr_cum = 0
    while True:
        if Nitr_cum >= Nitr:
            break
        call_training(x_generator, z_generator_, Nitr_phase, 
            eps_given = eps, max_itr=max_sink_itr, optimizer = optimizer)
        Nitr_cum += Nitr_phase

        cur.execute('''Select count(id) From Result''')
        cnt = cur.fetchone()[0]
        model_file_path = "./tmp/model_%04d.pt" % cnt

        torch.save(x_generator.state_dict(), model_file_path)

        cur.execute('''Insert or Ignore into Result (model_file_path, 
            Nz, Nlayer, Nz, Nlayer, Nh, Nbatch, Nitr, eps, max_sink_itr) 
            Values (?,?,?,?,?,?,?,?,?,?) ''', (model_file_path, Nz, Nlayer, 
            Nz, Nlayer, Nh, Nbatch, Nitr_cum, eps, max_sink_itr))
        conn.commit()
conn.close()
