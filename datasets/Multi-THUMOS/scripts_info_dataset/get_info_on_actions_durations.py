import os
import pandas as pd


if __name__ == '__main__':
    path_se = "../changed_file/"
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
            ]
            action_statistics["Run"] = [
                (se_df["end_s_run"] - se_df["begin_s_run"]).mean(),
                (se_df["end_s_run"] - se_df["begin_s_run"]).var(),
                (se_df["end_s_run"] - se_df["begin_s_run"]).std(),
            ]
            action_statistics["Jump"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
            ]
            action_statistics["Fall"] = [
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).mean(),
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).var(),
                (se_df["end_s_fall"] - se_df["begin_s_fall"]).std(),
            ]
        elif se_name == "LongJump":
            action_statistics["LongJump"] = [
                (se_df["end_s_lj"] - se_df["begin_s_lj"]).mean(),
                (se_df["end_s_lj"] - se_df["begin_s_lj"]).min(), (se_df["end_s_lj"] - se_df["begin_s_lj"]).max()
            ]
            action_statistics["Run"] = [
                (se_df["end_s_run"] - se_df["begin_s_run"]).mean(),
                (se_df["end_s_run"] - se_df["begin_s_run"]).min(), (se_df["end_s_run"] - se_df["begin_s_run"]).max(),
            ]
            action_statistics["Jump"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).min(), (se_df["end_s"] - se_df["begin_s"]).max()
            ]
            # action_statistics["Sit"] = [
            #     (se_df["end_s_sit"] - se_df["begin_s_sit"]).mean(),
            #     (se_df["end_s_sit"] - se_df["begin_s_sit"]).min(), (se_df["end_s_sit"] - se_df["begin_s_sit"]).max(),
            # ]
        elif se_name == "HammerThrow":
            action_statistics["HammerThrow"] = [
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).mean(),
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).var(),
                (se_df["end_s_ht"] - se_df["begin_s_ht"]).std(),
            ]
            action_statistics["HammerThrowWindUp"] = [
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).mean(),
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).var(),
                (se_df["end_s_ht_wu"] - se_df["begin_s_ht_wu"]).std(),
            ]
            action_statistics["HammerThrowSpin"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
            ]
            action_statistics["HammerThrowRelease"] = [
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).mean(),
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).var(),
                (se_df["end_s_ht_r"] - se_df["begin_s_ht_r"]).std(),
            ]
        elif se_name == "CleanAndJerk":
            action_statistics["CleanAndJerk"] = [
                (se_df["end_s_cj"] - se_df["begin_s_cj"]).mean(),
                (se_df["end_s_cj"] - se_df["begin_s_cj"]).var(),
                (se_df["end_s_cj"] - se_df["begin_s_cj"]).std(),
            ]
            action_statistics["WeightliftingClean"] = [
                (se_df["end_s_wlc"] - se_df["begin_s_wlc"]).mean(),
                (se_df["end_s_wlc"] - se_df["begin_s_wlc"]).var(),
                (se_df["end_s_wlc"] - se_df["begin_s_wlc"]).std(),
            ]
            action_statistics["WeightliftingJerk"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
            ]
        elif se_name == "ThrowDiscus":
            action_statistics["ThrowDiscus"] = [
                (se_df["end_s_td"] - se_df["begin_s_td"]).mean(),
                (se_df["end_s_td"] - se_df["begin_s_td"]).var(),
                (se_df["end_s_td"] - se_df["begin_s_td"]).std(),
            ]
            action_statistics["DiscusWindUp"] = [
                (se_df["end_s_td_wu"] - se_df["begin_s_td_wu"]).mean(),
                (se_df["end_s_td_wu"] - se_df["begin_s_td_wu"]).var(),
                (se_df["end_s_td_wu"] - se_df["begin_s_td_wu"]).std(),
            ]
            action_statistics["DiscusRelease"] = [
                (se_df["end_s"] - se_df["begin_s"]).mean(),
                (se_df["end_s"] - se_df["begin_s"]).var(),
                (se_df["end_s"] - se_df["begin_s"]).std(),
            ]
        print("\nSE: {}:\n".format(se_name))
        for action, info in action_statistics.items():
            print("action: {} -- info =  {}".format(action, info))
        
    
    
    
