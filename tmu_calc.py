import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv('movement_log.csv')

# 基本的なデータ処理
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')

# 差分計算
df['diff_X'] = df.groupby('Part')['X'].diff().fillna(0)
df['diff_Y'] = df.groupby('Part')['Y'].diff().fillna(0)
df['time_diff'] = df.groupby('Part')['Timestamp'].diff().fillna(0)

# 各キーポイントの変位計算
df['total_diff'] = (df['diff_X'].abs() + df['diff_Y'].abs())

# カテゴリごとの重み付け設定
weights = {
    'Pose_Landmark': 1.0,
    'Face_Landmark': 0.7,
    'Hand_Landmark': 0.9
}

# 重み付けとTMU係数の適用
def apply_tmu(row):
    part_type = row['Part'].split('_')[0] + '_Landmark'  # 'Pose_Landmark'などの抽出
    weight = weights.get(part_type, 1.0)  # 未定義の場合は重みを1とする
    return row['total_diff'] * weight * 0.036  # TMU係数の適用

df['weighted_tmu'] = df.apply(apply_tmu, axis=1)

# 各カテゴリごとに集計
summary = df.groupby(df['Part'].str.extract(r'(\D+)')[0])[['weighted_tmu', 'time_diff']].sum()
summary.columns = ['Total TMU', 'Total Time']

# 結果のCSV出力
summary.to_csv('tmu_summary_updated.csv')

print("更新されたTMU計算が完了し、tmu_summary_updated.csvに保存されました。")
