Official code of the NeurIPS 2022 paper - Transformer-based Working Memory for Multiagent Reinforcement Learning with Action Parsing

To switch between ATM and RNN agents, motidy src/config/default.yaml to enable RNN Agent parameters or Memory Agent parameters.

To run experiments on SMAC, use command like "python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=4m_vs_5m env_args.seed=1 > out_4m_vs_5m_atm_qmix_1.log 2>&1 &".

To run experiments on LBF, use command like "python3 src/main.py --config=maa2c --env-config=gymma with env_args.key="lbforaging:Foraging-5s-10x10-3p-3f-v2" t_max=5000000 > out_lbf_10x10_3p3f_atm_maa2c_1.log 2>&1 &".

Note that, on LBF, the parameter "num_foods" in src/config/default.yaml should correspond with each scenario such as "3f" means "num_foods" is 3.
