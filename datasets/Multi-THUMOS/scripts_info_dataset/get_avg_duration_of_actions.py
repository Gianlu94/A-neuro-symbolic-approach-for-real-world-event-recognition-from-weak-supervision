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
            action_statistics["HighJump"] = round((se_df["end_f_hj"] - se_df["begin_f_hj"] + 1).mean())
            action_statistics["Run"] = round((se_df["end_f_run"] - se_df["begin_f_run"] + 1).mean())
            action_statistics["Jump"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
            action_statistics["Fall"] = round((se_df["end_f_fall"] - se_df["begin_f_fall"] + 1).mean())
        elif se_name == "LongJump":
            action_statistics["LongJump"] = round((se_df["end_f_lj"] - se_df["begin_f_lj"] + 1).mean())
            action_statistics["Run"] = round((se_df["end_f_run"] - se_df["begin_f_run"] + 1).mean())
            action_statistics["Jump"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
            action_statistics["Sit"] = round((se_df["end_f_sit"] - se_df["begin_f_sit"] + 1).mean())
        elif se_name == "PoleVault":
            action_statistics["PoleVault"] = round((se_df["end_f_pv"] - se_df["begin_f_pv"] + 1).mean())
            action_statistics["Run"] = round((se_df["end_f_run"] - se_df["begin_f_run"] + 1).mean())
            action_statistics["PoleVaultPlantPole"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
            action_statistics["Jump"] = round((se_df["end_f_jump"] - se_df["begin_f_jump"] + 1).mean())
            action_statistics["Fall"] = round((se_df["end_f_fall"] - se_df["begin_f_fall"] + 1).mean())
        elif se_name == "HammerThrow":
            action_statistics["HammerThrow"] = round((se_df["end_f_ht"] - se_df["begin_f_ht"] + 1).mean())
            action_statistics["HammerThrowWindUp"] = round((se_df["end_f_ht_wu"] - se_df["begin_f_ht_wu"] + 1).mean())
            action_statistics["HammerThrowSpin"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
            action_statistics["HammerThrowRelease"] = round((se_df["end_f_ht_r"] - se_df["begin_f_ht_r"] + 1).mean())
        elif se_name == "ThrowDiscus":
            action_statistics["ThrowDiscus"] = round((se_df["end_f_td"] - se_df["begin_f_td"] + 1).mean())
            action_statistics["DiscusWindUp"] = round((se_df["end_f_td_wu"] - se_df["begin_f_td_wu"] + 1).mean())
            action_statistics["DiscusRelease"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
        elif se_name == "Shotput":
            action_statistics["Shotput"] = round((se_df["end_f_sp"] - se_df["begin_f_sp"] + 1).mean())
            action_statistics["ShotPutBend"] = round((se_df["end_f_spb"] - se_df["begin_f_spb"] + 1).mean())
            action_statistics["Throw"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
        elif se_name == "JavelinThrow":
            action_statistics["JavelinThrow"] = round((se_df["end_f_jt"] - se_df["begin_f_jt"] + 1).mean())
            action_statistics["Run"] = round((se_df["end_f_run"] - se_df["begin_f_run"] + 1).mean())
            action_statistics["Throw"] = round((se_df["end_f"] - se_df["begin_f"] + 1).mean())
        
        print("\nSE: {}:\n".format(se_name))
        for action, info in action_statistics.items():
            print("action: {} -- avg_duration =  {}".format(action, info))
        
    
    
    
