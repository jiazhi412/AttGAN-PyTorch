from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('errors'):
    os.makedirs('errors')

def grid(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, command_template):
    for p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 in product(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
        command = command_template.format(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)
        job_name = f'{p1}-{p2}-{p3}-{p4}-{p5}-{p6}'
        bash_file = '{}.sh'.format(job_name)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('#!/bin/bash\n')
            OUT.write('#SBATCH --job-name={} \n'.format(job_name))
            OUT.write('#SBATCH --ntasks=1 \n')
            OUT.write('#SBATCH --account=other \n')
            OUT.write('#SBATCH --qos=premium \n')
            OUT.write('#SBATCH --partition=ALL \n')
            OUT.write('#SBATCH --cpus-per-task=4 \n')
            OUT.write('#SBATCH --gres=gpu:1 \n')
            OUT.write('#SBATCH --mem=16G \n')
            OUT.write('#SBATCH --time=5-00:00:00 \n')
            OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
            OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
            OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
            OUT.write('source ~/.bashrc\n')
            OUT.write('conda activate pytorch\n')
            OUT.write(command)

        qsub_command = 'sbatch {}'.format(bash_file)
        os.system( qsub_command )
        os.system('rm -f {}'.format(bash_file))
        print( qsub_command )
        print( 'Submitted' )

# parameters
command_template = 'python train_CelebA_our.py --experiment {} --name {} --gr {} --gc {} --dc {} --ga {} --mi {} --num_ganp1 {} --num_ganp2 {} --num_dis {}'
# p1 = ['label_mse']
# p1 = ['label_mse_G1pretrain']
# p1 = ['label']
# p1 = ['label_mse_MI']
# p1 = ['label_mse_MI_Pscratch']
p1 = ['label_mse_MI_PDshare']
p2 = ['927']
p3 = [100] # gr
p4 = [5] # gc
p5 = [1] # dc
p6 = [5] # ga
p7 = [1] # mi
p8 = [1]
p9 = [1]
p10 = [1]

grid(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, command_template)
for i in range(7):
    if i == 0:
        p4 = [10, 20] # gc
    if i == 1:
        p4 = [5] # gc
        p5 = [5, 10] # dc
    if i == 2:
        p5 = [1] # dc
        p6 = [10, 20] # ga
    if i == 3:
        p6 = [5] # ga
        p7 = [5, 10] # mi
    if i == 4:
        p7 = [1] # mi
        p8 = [5, 10]
    if i == 5:
        p8 = [1]
        p9 = [5, 10]
    if i == 6:
        p9 = [1]
        p10 = [5, 10]
    grid(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, command_template)
