from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')

# parameters
command_template = 'python train_sex_label.py --experiment_name {} --inject_layers {} --gc {} --dim_per_attr {} '
p1 = ['913_2']
p2 = [0,4]
p3 = [10,100,1000]
p4 = [1,10,100]
# p4 = [4, 8, 16, 32]


for p1, p2, p3, p4 in product(p1, p2, p3, p4):
    command = command_template.format(p1, p2, p3, p4)
    job_name = f'{p1}-{p2}-{p3}-{p4}'
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
        OUT.write('#SBATCH --mem=15G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('conda activate pytorch\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )
