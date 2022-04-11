import os
import argparse

from utils import *

PROJECT_PATH = os.path.abspath("..")
DATASET_PATH = PROJECT_PATH + "/R3-Driving-Dataset/dataset/expert/"

if __name__ == '__main__':
    print("Initialize script")

    # Define hyperparameter
    parser = argparse.ArgumentParser(description='data processor for R3D dataset')

    parser.add_argument('--obs_rollout', default=5, help='rollout length of observation')
    parser.add_argument('--act_rollout', default=10, help='rollout length of action')
    parser.add_argument('--n_scenario', default=13, help='number of scenario')

    args = parser.parse_args()
    print("==============================Setting==============================")
    print(args)
    print("===================================================================")

    n_data = 0
    # Load R3 Driving Dataset
    for i_scenario in range(1, args.n_scenario + 1):
        print("#############################################################################")
        FILE_PATH = DATASET_PATH + "scenario_" + str(i_scenario).zfill(3) + "/data/"
        SAVE_PATH = DATASET_PATH + "scenario_" + str(i_scenario).zfill(3) + "/result/"
        make_dir(SAVE_PATH)
        n_frames = count_files(FILE_PATH)
        print("# of frames in expert scenario %03d: %d" %(i_scenario, n_frames))

        raw_data = []
        for seq in range(1, n_frames + 1):
            FILE_NAME = FILE_PATH + str(seq).zfill(6) + ".json"
            raw_data.append(load_json(FILE_NAME))
    
        # build state
        obs_data = []
        obs_seq_data = []
        for seq in range(n_frames):
            obs_data.append(get_obs(raw_data[seq]))
        
        for seq in range(args.obs_rollout, n_frames - args.act_rollout):
            obs = []
            for i in range(args.obs_rollout):
                obs.append(obs_data[seq - i - 1])
            obs_seq_data.append(obs)
        print("# of obs data: %d" %(len(obs_seq_data)))

        # build action
        act_seq_data = []
        for seq in range(args.obs_rollout, n_frames - args.act_rollout):
            act_seq_data.append(get_act(obs_data[seq - 1  : seq + args.act_rollout]))
        print("# of act data: %d" %(len(act_seq_data)))

        # save the files
        for seq in range(0, n_frames - args.obs_rollout - args.act_rollout):
            n_data += 1
            save_file = SAVE_PATH + str(seq).zfill(6) + ".json"
            with open(save_file, 'w') as jf:
                json.dump({'obs': obs_seq_data[seq], 'act': act_seq_data[seq]}, jf, indent=4)
        print("Complete to save about expert scenario %03d" %(i_scenario))
        print("# of data, current scenario: %d, total: %d" %(n_frames - args.obs_rollout - args.act_rollout, n_data))

    print("Finish script")