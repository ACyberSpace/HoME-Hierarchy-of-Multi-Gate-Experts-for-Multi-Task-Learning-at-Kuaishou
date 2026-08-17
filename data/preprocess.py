import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

try:
    from recall.data.feature_column import RECALL_USER_FEATURES, RECALL_ITEM_FEATURES
except ImportError:
    RECALL_USER_FEATURES = [
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    RECALL_ITEM_FEATURES = [
        "video_id",
        "author_id",
        "music_id",
    ]


def convert_date(date: int):
    date_str = str(date)
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:]
    return datetime(int(year), int(month), int(day))


def compute_last_k_clicked_history(
    df: pd.DataFrame, k: int = 20, pad_value: int = 0
) -> pd.DataFrame:
    """
    计算每个用户在每个日期点击的最后k个视频id，基于当前日期之前的点击
    无数据泄露：序列只包含当前日期之前的历史点击

    Args:
        df (pd.DataFrame): 输入的DataFrame，包含列['user_id', 'video_id', 'date', 'time_ms', 'is_click']
        k (int): 要获取的最后点击的个数
        pad_value (int): 用于填充较短序列的值

    Returns:
        pd.DataFrame: 包含'last_k_clicked_items'列的原始数据
    """
    if not all(
        col in df.columns
        for col in ["user_id", "video_id", "date", "time_ms", "is_click"]
    ):
        raise ValueError(
            "输入的DataFrame必须包含列: 'user_id', 'video_id', 'date', 'time_ms', 'is_click'"
        )

    if k <= 0:
        df["last_k_clicked_items"] = [[] for _ in range(len(df))]
        return df

    df_processed = df.copy()
    df_processed["date"] = df_processed["date"].apply(lambda x: convert_date(x))
    df_processed["is_click"] = df_processed["is_click"].astype(bool)
    df_processed = df_processed.sort_values(
        by=["user_id", "date", "time_ms"], ascending=True
    )
    df_processed.reset_index(drop=True, inplace=True)

    clicked_df = df_processed[df_processed["is_click"]].copy()

    if clicked_df.empty:
        print("Warning: No click interactions found in the data.")
        df_processed["last_k_clicked_items"] = [[pad_value] * k] * len(df_processed)
        return df_processed

    daily_clicks = (
        clicked_df.groupby(["user_id", "date"])["video_id"]
        .apply(list)
        .reset_index()
    )
    daily_clicks = daily_clicks.sort_values(by=["user_id", "date"])

    daily_clicks["cumulative_history"] = daily_clicks.groupby("user_id")[
        "video_id"
    ].transform(lambda s: s.cumsum())

    daily_clicks["prev_days_history"] = daily_clicks.groupby("user_id")[
        "cumulative_history"
    ].shift(1)

    daily_clicks["prev_days_history"] = daily_clicks["prev_days_history"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    history_map = daily_clicks[["user_id", "date", "prev_days_history"]]

    df_processed = pd.merge(
        df_processed, history_map, on=["user_id", "date"], how="left"
    )

    df_processed = df_processed.sort_values(
        by=["user_id", "date", "time_ms"], ascending=True
    )

    df_processed["propagated_history"] = df_processed.groupby("user_id")[
        "prev_days_history"
    ].ffill()

    df_processed["propagated_history"] = df_processed["propagated_history"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    def pad_and_truncate(history_list, target_k: int, pad_val: int) -> list:
        if not isinstance(history_list, list):
            return [pad_val] * target_k
        last_k = history_list[-target_k:]
        pad_len = target_k - len(last_k)
        return ([pad_val] * pad_len) + last_k

    df_processed["last_k_clicked_items"] = df_processed["propagated_history"].apply(
        lambda hist: pad_and_truncate(hist, k, pad_value)
    )

    return df_processed.drop(columns=["prev_days_history", "propagated_history"])


def compute_session_id(
    df: pd.DataFrame, time_threshold_minutes: int = 30
) -> pd.DataFrame:
    """
    根据用户活动间隙计算每个交互的会话ID。

    Args:
        df (pd.DataFrame): 输入的DataFrame，包含列['user_id', 'time_ms', ...]
        time_threshold_minutes (int): 会话开始前的最大空闲时间，以分钟为单位

    Returns:
        pd.DataFrame: 原始的DataFrame，包含一个添加的列'session_id'
    """
    if not all(col in df.columns for col in ["user_id", "time_ms"]):
        raise ValueError("输入的DataFrame必须包含列: 'user_id', 'time_ms'")
    if df.empty:
        df["session_id"] = pd.Series(dtype="str")
        return df

    df_processed = df.copy()
    df_processed["_position"] = np.arange(len(df_processed))
    df_processed["time_ms"] = pd.to_numeric(df_processed["time_ms"])

    df_processed = df_processed.sort_values(
        by=["user_id", "time_ms"], ascending=True
    )

    time_threshold_ms = time_threshold_minutes * 60 * 1000

    prev_time_ms = df_processed.groupby("user_id")["time_ms"].shift(1)
    time_diff_ms = df_processed["time_ms"] - prev_time_ms

    is_new_session = (time_diff_ms > time_threshold_ms) | (prev_time_ms.isna())
    session_numeric_id = is_new_session.cumsum()

    session_ids = (
        df_processed["user_id"].astype(str) + "_" + session_numeric_id.astype(str)
    )

    return pd.Series(session_ids.to_numpy(), index=df_processed["_position"]).sort_index()


def preprocess(input_path: Path, output_path: Path) -> dict:
    print("加载数据...")

    log_df = pd.read_csv(input_path / "data" / "log_standard_4_22_to_5_08_1k.csv")
    user_features = pd.read_csv(input_path / "data" / "user_features_1k.csv")
    video_features_basic = pd.read_csv(
        input_path / "data" / "video_features_basic_1k.csv"
    )
    video_features_statistics = pd.read_csv(
        input_path / "data" / "video_features_statistic_1k.csv"
    )

    select_log_columns = [
        "user_id",
        "video_id",
        "date",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "tab",
    ]
    new_log_df = log_df[select_log_columns]

    print("处理用户特征...")
    user_sparse_feature_columns = [
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
        "onehot_feat4",
        "onehot_feat5",
        "onehot_feat6",
        "onehot_feat7",
        "onehot_feat8",
        "onehot_feat9",
        "onehot_feat10",
        "onehot_feat11",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
        "onehot_feat15",
        "onehot_feat16",
        "onehot_feat17",
    ]
    new_user_feature_df = user_features[user_sparse_feature_columns]

    new_user_feature_df["is_live_streamer"] = new_user_feature_df[
        "is_live_streamer"
    ].apply(lambda x: 0 if x == -124 else x)

    user_label_encode_feature_columns = [
        "user_id",
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]

    for feat_name in user_label_encode_feature_columns:
        label_encoder = LabelEncoder()
        new_user_feature_df[feat_name + "_encode"] = (
            label_encoder.fit_transform(new_user_feature_df[feat_name]) + 1
        )
        if feat_name not in ("user_id"):
            new_user_feature_df[feat_name] = new_user_feature_df[feat_name + "_encode"]
            del new_user_feature_df[feat_name + "_encode"]

    onehot_feat_columns = [x for x in user_sparse_feature_columns if "onehot_" in x]
    for feat_name in onehot_feat_columns:
        max_val = new_user_feature_df[feat_name].max()
        new_user_feature_df[feat_name].fillna(value=max_val + 1, inplace=True)
        new_user_feature_df[feat_name] = new_user_feature_df[feat_name].astype(np.int32)

    print("处理视频基础特征...")
    select_video_basic_feature_columns = [
        "video_id",
        "author_id",
        "video_type",
        "upload_type",
        "visible_status",
        "music_id",
        "music_type",
        "tag",
    ]

    new_video_features_basic_df = video_features_basic[
        select_video_basic_feature_columns
    ]

    for feat_name in ["visible_status", "music_type"]:
        max_val = new_video_features_basic_df[feat_name].max()
        new_video_features_basic_df[feat_name].fillna(value=max_val + 1, inplace=True)
        new_video_features_basic_df[feat_name] = new_video_features_basic_df[
            feat_name
        ].astype(np.int32)

    video_sparse_feature_columns = [
        x for x in select_video_basic_feature_columns if x not in ("tag")
    ]

    for feat_name in video_sparse_feature_columns:
        label_encoder = LabelEncoder()
        new_video_features_basic_df[feat_name + "_encode"] = (
            label_encoder.fit_transform(new_video_features_basic_df[feat_name]) + 1
        )
        if feat_name not in ("video_id"):
            new_video_features_basic_df[feat_name] = new_video_features_basic_df[
                feat_name + "_encode"
            ]
            del new_video_features_basic_df[feat_name + "_encode"]

    new_video_features_basic_df["tag"].fillna(value="-1", inplace=True)
    tag_set = set([])
    for x in new_video_features_basic_df["tag"].values:
        tag_list = x.split(",")
        for tag in tag_list:
            tag_set.add(tag)
    tag_map_dict = {}
    for i, tag in zip(range(len(tag_set)), tag_set):
        tag_map_dict[tag] = i + 1
    new_video_features_basic_df["tag"] = new_video_features_basic_df["tag"].apply(
        lambda x: [tag_map_dict[tag] for tag in x.split(",")]
    )

    print("处理视频统计特征...")
    video_features_statistics_columns = [
        "video_id",
        "counts",
        "show_cnt",
        "show_user_num",
        "play_cnt",
        "play_user_num",
        "play_duration",
        "complete_play_cnt",
        "complete_play_user_num",
        "valid_play_cnt",
        "valid_play_user_num",
        "long_time_play_cnt",
        "long_time_play_user_num",
        "short_time_play_cnt",
        "short_time_play_user_num",
        "play_progress",
        "comment_stay_duration",
        "like_cnt",
        "like_user_num",
        "click_like_cnt",
        "double_click_cnt",
        "cancel_like_cnt",
        "cancel_like_user_num",
        "comment_cnt",
        "comment_user_num",
        "direct_comment_cnt",
        "reply_comment_cnt",
        "delete_comment_cnt",
        "delete_comment_user_num",
        "comment_like_cnt",
        "comment_like_user_num",
        "follow_cnt",
        "follow_user_num",
        "cancel_follow_cnt",
        "cancel_follow_user_num",
        "share_cnt",
        "share_user_num",
        "download_cnt",
        "download_user_num",
        "report_cnt",
        "report_user_num",
        "reduce_similar_cnt",
        "reduce_similar_user_num",
        "collect_cnt",
        "collect_user_num",
        "cancel_collect_cnt",
        "cancel_collect_user_num",
        "direct_comment_user_num",
        "reply_comment_user_num",
        "share_all_cnt",
        "share_all_user_num",
        "outsite_share_all_cnt",
    ]

    video_dense_feature_columns = [
        x for x in video_features_statistics_columns if x not in ("video_id", "counts")
    ]
    new_video_features_statistics_df = video_features_statistics[
        ["video_id"] + video_dense_feature_columns
    ]

    for feat_name in video_dense_feature_columns:
        new_video_features_statistics_df[feat_name].fillna(value=0.0, inplace=True)

    print("合并特征...")
    df_merged = new_log_df.merge(new_user_feature_df, on="user_id", how="left")
    df_merged = df_merged.merge(new_video_features_basic_df, on="video_id", how="left")
    df_merged = df_merged.merge(
        new_video_features_statistics_df, on="video_id", how="left"
    )

    df_merged["user_id"] = df_merged["user_id_encode"]
    df_merged["video_id"] = df_merged["video_id_encode"]
    del df_merged["user_id_encode"]
    del df_merged["video_id_encode"]

    df_merged["tag"] = df_merged["tag"].apply(
        lambda x: [0] if not isinstance(x, list) or len(x) == 0 else x
    )
    df_merged["tag"] = df_merged["tag"].apply(lambda x: x[0])
    
    non_datetime_cols = [col for col in df_merged.columns if col not in ("date", "time_ms")]
    df_merged[non_datetime_cols] = df_merged[non_datetime_cols].fillna(value=0)

    for feat_name in tqdm(video_dense_feature_columns):
        df_merged[feat_name + "_binned"] = pd.qcut(
            df_merged[feat_name], q=10, labels=False, duplicates="drop"
        )
        df_merged[feat_name] = df_merged[feat_name + "_binned"]
        del df_merged[feat_name + "_binned"]

    columns = [x for x in df_merged.columns if x not in ("tag", "date", "time_ms")]
    for feat_name in columns:
        df_merged[feat_name] = df_merged[feat_name].astype(np.int32)

    main_tab_set = set([1, 0, 4, 2, 6])
    df_merged = df_merged[df_merged["tab"].isin(main_tab_set)]

    label_encoder = LabelEncoder()
    df_merged["tab_encode"] = label_encoder.fit_transform(df_merged["tab"])
    df_merged["tab"] = df_merged["tab_encode"]
    del df_merged["tab_encode"]

    print("计算序列特征...")
    SHORT_LEN = 50
    LONG_LEN = 200

    df_merged_with_long = compute_last_k_clicked_history(
        df_merged[["user_id", "video_id", "date", "time_ms", "is_click"]].copy(),
        k=LONG_LEN,
        pad_value=0,
    )
    df_merged["long_seq"] = df_merged_with_long["last_k_clicked_items"].values

    print("生成序列mask...")
    long_seq_array = np.asarray(df_merged["long_seq"].tolist(), dtype=np.int32)
    short_seq_array = long_seq_array[:, -SHORT_LEN:]
    short_mask_array = (short_seq_array != 0).astype(np.int8)
    long_mask_array = (long_seq_array != 0).astype(np.int8)
    print("计算会话id...")
    df_merged["session_id"] = compute_session_id(df_merged[["user_id", "time_ms"]]).to_numpy()

    print("生成特征词典...")
    not_feat_dict_columns = [
        "date",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "session_id",
        "short_seq",
        "long_seq",
        "short_mask",
        "long_mask",
    ]
    total_columns = [
        x for x in list(df_merged.columns) if x not in not_feat_dict_columns
    ]

    final_feature_dict = {}
    for feat_name in total_columns:
        final_feature_dict[feat_name] = df_merged[feat_name].max() + 1

    save_path = output_path / "kuairand_feature_dict.pkl"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    with open(save_path, "wb") as f:
        pickle.dump(final_feature_dict, f)

    print("划分训练和测试集（按日期）...")
    df_merged["date"] = df_merged["date"].apply(lambda x: convert_date(x))
    test_mask = df_merged["date"].dt.strftime('%Y%m%d') == '20220508'
    train_mask = ~test_mask
    test_df = df_merged[test_mask]
    train_eval_df = df_merged[train_mask]

    test_index = np.flatnonzero(test_mask.to_numpy())
    train_index = np.flatnonzero(train_mask.to_numpy())
    split_seq_arrays = {
        "train": {
            "short_seq": short_seq_array[train_index],
            "long_seq": long_seq_array[train_index],
            "short_mask": short_mask_array[train_index],
            "long_mask": long_mask_array[train_index],
        },
        "test": {
            "short_seq": short_seq_array[test_index],
            "long_seq": long_seq_array[test_index],
            "short_mask": short_mask_array[test_index],
            "long_mask": long_mask_array[test_index],
        },
    }

    train_eval_df_no_date = train_eval_df.drop(columns=["date", "time_ms"])
    test_df_no_date = test_df.drop(columns=["date", "time_ms"])

    print("保存训练和测试集（按日期，用于评估）...")
    train_eval_dict = {}
    for data_type, data_df in zip(["train", "test"], [train_eval_df_no_date, test_df_no_date]):
        train_eval_dict[data_type] = {}
        for feat_name in data_df.columns:
            if feat_name in ("short_seq", "long_seq"):
                train_eval_dict[data_type][feat_name] = split_seq_arrays[data_type][feat_name]
            else:
                train_eval_dict[data_type][feat_name] = np.array(
                    data_df[feat_name].values, dtype=np.int32
                )
        train_eval_dict[data_type]["short_seq"] = split_seq_arrays[data_type]["short_seq"]
        train_eval_dict[data_type]["long_seq"] = split_seq_arrays[data_type]["long_seq"]
        train_eval_dict[data_type]["short_mask"] = split_seq_arrays[data_type]["short_mask"]
        train_eval_dict[data_type]["long_mask"] = split_seq_arrays[data_type]["long_mask"]

    save_path = output_path / "kuairand_train_eval.pkl"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    joblib.dump(train_eval_dict, save_path, compress=3)

    print("召回训练数据将直接使用按日期划分的 kuairand_train_eval.pkl")

    print("构建用户序列数据（用于召回模型）...")

    def build_user_sequences(df: pd.DataFrame) -> dict:
        all_users = df["user_id"].unique()
        user_ids = []
        short_seqs = []
        long_seqs = []
        short_masks = []
        long_masks = []
        short_seq_lens = []
        long_seq_lens = []
        full_sequences = []

        for uid in tqdm(all_users, desc="构建用户序列"):
            user_ids.append(uid)

            user_data = df[df["user_id"] == uid]
            if len(user_data) == 0:
                short_seqs.append([0] * SHORT_LEN)
                short_masks.append([0] * SHORT_LEN)
                short_seq_lens.append(0)
                long_seqs.append([0] * LONG_LEN)
                long_masks.append([0] * LONG_LEN)
                long_seq_lens.append(0)
                full_sequences.append([])
                continue

            last_row = user_data.iloc[-1]
            short_seq = last_row["long_seq"][-SHORT_LEN:]
            short_seqs.append(short_seq)
            short_mask = [1 if item != 0 else 0 for item in short_seq]
            long_mask = [1 if item != 0 else 0 for item in last_row["long_seq"]]
            short_masks.append(short_mask)
            short_seq_lens.append(sum(short_mask))
            long_seqs.append(last_row["long_seq"])
            long_masks.append(long_mask)
            long_seq_lens.append(sum(long_mask))
            
            click_data = user_data[user_data["is_click"] == 1].sort_values("date")
            clicked_videos = []
            seen = set()
            for vid in click_data["video_id"].tolist():
                if vid not in seen:
                    seen.add(vid)
                    clicked_videos.append(vid)
            full_sequences.append(clicked_videos)

        short_seqs_np = np.array(short_seqs, dtype=np.int32)
        long_seqs_np = np.array(long_seqs, dtype=np.int32)
        
        return {
            "user_id": np.array(user_ids, dtype=np.int32),
            "short_seq": short_seqs_np,
            "long_seq": long_seqs_np,
            "short_mask": np.array(short_masks, dtype=np.int32),
            "long_mask": np.array(long_masks, dtype=np.int32),
            "short_seq_len": np.array(short_seq_lens, dtype=np.int32),
            "long_seq_len": np.array(long_seq_lens, dtype=np.int32),
            "full_sequences": full_sequences,
            "user_id_max": np.max(user_ids),
            "video_id_max": final_feature_dict.get("video_id", int(max(short_seqs_np.max(), long_seqs_np.max()))) - 1,
            "author_id_max": final_feature_dict.get("author_id", 50000) - 1,
            "music_id_max": final_feature_dict.get("music_id", 20000) - 1,
        }

    user_sequences = build_user_sequences(train_eval_df)

    save_path = output_path / "user_sequences.pkl"
    joblib.dump(user_sequences, save_path, compress=3)
    print(f"  保存用户序列: {save_path}")

    print("保存视频信息...")
    video_info = new_video_features_basic_df.copy()
    video_info["video_id"] = video_info["video_id_encode"]
    del video_info["video_id_encode"]

    def parse_upload_dt(upload_dt):
        if pd.isna(upload_dt):
            return 0.0
        try:
            date_str = str(int(upload_dt))
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            dt = datetime(year, month, day)
            return dt.timestamp()
        except:
            return 0.0

    if "upload_dt" in video_info.columns:
        video_info["upload_timestamp"] = video_info["upload_dt"].apply(parse_upload_dt)

    save_path = output_path / "video_info.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(video_info, f)
    print(f"  保存视频信息: {save_path}")

    print("\n预处理完成！")
    print(f"输出文件:")
    print(f"  1. kuairand_train_eval.pkl  - 训练/测试数据（含序列特征）")
    print(f"  2. kuairand_feature_dict.pkl - 特征字典")
    print(f"  3. user_sequences.pkl       - 用户序列")
    print(f"  4. video_info.pkl           - 视频信息")


if __name__ == '__main__':
    input_path = Path("../KuaiRand-1K")
    save_path = Path("../KuaiRand-1K")
    preprocess(input_path, save_path)
