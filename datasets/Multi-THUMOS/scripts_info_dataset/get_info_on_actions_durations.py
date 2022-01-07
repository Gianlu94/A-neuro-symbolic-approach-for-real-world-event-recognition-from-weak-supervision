import os
import pandas as pd


if __name__ == '__main__':
    path_se = "../filtered_se/"
    se_files = os.listdir(path_se)
    
    for se_file in se_files:
        se_name = se_file.split(".")[0]
        se_df = pd.read_csv(path_se + se_file)
        se_df = se_df.iloc[:, 1:]
        se_df = se_df[se_df["video"].str.contains("validation")]
        action_statistics = {}
        if se_name == "HighJump":
            action_statistics["HighJump"] = [
                (se_df["end_s_hj"] - se_df["begin_s_hj"]).mean(),
                (se_df["end_s_hj"] - se_df["begin_s_hj"]).var(),
                (se_df["end_s_hj"] - se_df["begin_s_hj"]).std(),
                # see min and max durations of actions
                #(se_df["end_s_hj"] - se_df["begin_s_hj"]).min(), (se_df["end_s_hj"] - se_df["begin_s_hj"]).max()
            ]
            action_statistics["Run"] = [
                (se_df["end_s_run"] - se_df["begin_s_run"]).mean(),
                (se_df["end_s_run"] - se_df["begin_s_run"]).var(),
                (se_df["end_s_run"] - se_df["begin_s_run"]).std(),
                #(se_df["end_s_run"] - se_df["begin_s_run"]).min(), (se_df["end_s_run"] - se_df["begin_s_run"]).max(),
            ]
            action_statistics["Jump"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
                #(se_df["end_s"] - se_df["begin_s"]).min(), (se_df["end_s"] - se_df["begin_s"]).max()
            ]
            action_statistics["Fall"] = [
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).mean(),
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).var(),
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).std(),
                #(se_df["end_s_fall"] - se_df["begin_s_fall"]).min(), (se_df["end_s_fall"] - se_df["begin_s_fall"]).max()
            ]
        elif se_name == "LongJump":
            continue
            # action_statistics["LongJump"] = [
            #     (se_df["end_s_lj"] - se_df["begin_s_lj"]).mean(),
            #     (se_df["end_s_lj"] - se_df["begin_s_lj"]).min(), (se_df["end_s_lj"] - se_df["begin_s_lj"]).max()
            # ]
            # action_statistics["Run"] = [
            #     (se_df["end_s_run"] - se_df["begin_s_run"]).mean(),
            #     (se_df["end_s_run"] - se_df["begin_s_run"]).min(), (se_df["end_s_run"] - se_df["begin_s_run"]).max(),
            # ]
            # action_statistics["Jump"] = [
            #     (se_df["end_s"] - se_df["begin_s"]).mean(),
            #     (se_df["end_s"] - se_df["begin_s"]).min(), (se_df["end_s"] - se_df["begin_s"]).max()
            # ]
            # action_statistics["Sit"] = [
            #     (se_df["end_s_sit"] - se_df["begin_s_sit"]).mean(),
            #     (se_df["end_s_sit"] - se_df["begin_s_sit"]).min(), (se_df["end_s_sit"] - se_df["begin_s_sit"]).max(),
            # ]
        elif se_name == "HammerThrow":
            action_statistics["HammerThrow"] = [
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).mean(),
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).var(),
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).std(),
                #(se_df["end_s_ht"] - se_df["begin_s_ht"]).min(), (se_df["end_s_ht"] - se_df["begin_s_ht"]).max()
            ]
            action_statistics["HammerThrowWindUp"] = [
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).mean(),
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).var(),
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).std(),
                #(se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).min(), (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).max(),
            ]
            action_statistics["HammerThrowSpin"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
                #(se_df["end_s"] - se_df["begin_s"]).min(), (se_df["end_s"] - se_df["begin_s"]).max()
            ]
            action_statistics["HammerThrowRelease"] = [
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).mean(),
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).var(),
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).std(),
                #(se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).min(), (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).max(),
            ]
        
        print("\nSE: {}:\n".format(se_name))
        for action, info in action_statistics.items():
            print("action: {} -- info =  {}".format(action, info))
        
    
    
    
