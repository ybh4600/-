

# -*- coding: utf-8 -*-  
"""  
Created on Tue Apr  8 16:49:25 2025  

@author: ybh4600  
"""  

import pandas as pd  
import numpy as np  
import gc  
from sklearn.linear_model import LogisticRegression  
from sklearn.feature_selection import RFE  
import lightgbm as lgb  
from sklearn.model_selection import train_test_split  

#==========================================================  
# 第一部分：文本字段整数映射处理  
#==========================================================  
print("\n===== 第一部分：文本字段整数映射处理 =====")  
input_file_path = r"C:\Users\ybh4600\Desktop\第二届瑞智杯建模大赛\1缺失值处理\out_trn_data_cleaned.csv"  
output_file_path = r"C:\Users\ybh4600\Desktop\第二届瑞智杯建模大赛\1缺失值处理\out_trn_data_cleaned_int.csv"  

# 设置较大的块大小以加快处理速度，但不会过度消耗内存  
chunk_size = 50000  
chunks = []  

# 先读取一小块，推断出所有列的类型 -- 只是为了拿到完整的列名和 object 列  
preview_chunk = pd.read_csv(input_file_path, nrows=1)  
object_cols = preview_chunk.select_dtypes(include=['object']).columns.tolist()  
print("文本字段列名:", object_cols)  

# 为每个文本列建立一个映射词典(col_dict)和当前最大编号(col_max_code)  
col_dict = {col: {} for col in object_cols}  
col_max_code = {col: 1 for col in object_cols}  

# 分块读取并逐块转换  
for chunk_idx, chunk in enumerate(pd.read_csv(input_file_path, chunksize=chunk_size, low_memory=True)):  
    print(f"正在处理第 {chunk_idx + 1} 块数据，大小: {chunk.shape}")  

    # 对 chunk 中每个 object 列进行映射  
    for col in object_cols:  
        # 取当前列，查看未见过的新字符串  
        col_series = chunk[col].astype(str)  # 若有空值或混合类型，用 str() 统一  
        unique_vals = col_series.unique()  

        for val in unique_vals:  
            if val not in col_dict[col]:  
                # 若是新的字符串，分配新的整数编码  
                col_dict[col][val] = col_max_code[col]  
                col_max_code[col] += 1  
        
        # 用映射字典替换  
        chunk[col] = col_series.map(col_dict[col]).astype('Int64')  

    chunks.append(chunk)  
    gc.collect()  

# 拼接并写出到文件  
print("所有数据块处理完成，正在合并并写入文件...")  
df = pd.concat(chunks, ignore_index=True)  
del chunks  
gc.collect()  

df.to_csv(output_file_path, index=False)  
print(f"文本类型字段已替换为全局一致的整数值，文件已保存到：{output_file_path}")  

#==========================================================  
# 第二部分：特征选择与排序  
#==========================================================  
print("\n===== 第二部分：特征选择与排序 =====")  
print("===== 程序开始：数据读取 =====")  

# 直接使用上一步输出的文件作为输入  
df = pd.read_csv(output_file_path)  
print("数据读取完成，数据形状：", df.shape)  

# 特征与标签  
features = [col for col in df.columns if col.startswith("x")]  
X = df[features]  
y = df["y"]  
print(f"共读取到特征 {len(features)} 个, 标签列 'y' 读取完成.")  

print("===== 区分数值型与类别型字段 =====")  
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()  
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()  
print(f"数值型字段数: {len(numeric_cols)}, 类别型字段数: {len(categorical_cols)}")  

print("===== LightGBM 数据准备（不预处理缺失值） =====")  
X_lgb = X.copy()  
for c in categorical_cols:  
    X_lgb[c] = X_lgb[c].astype('category')  
print(f"LightGBM数据转换完成, 形状 {X_lgb.shape}")  

print("===== 针对非 LightGBM 的算法，做缺失值预处理 =====")  
# 1) 数值型字段：缺失率 <= 30% 的保留并用中位数填充  
missing_rate_num = X[numeric_cols].isnull().mean()  
numeric_keep = [c for c in numeric_cols if missing_rate_num[c] <= 0.3]  
X_numeric = X[numeric_keep].copy()  
X_numeric = X_numeric.fillna(X_numeric.median())  

# 2) 类别型字段：get_dummies（含 dummy_na）  
X_cate = X[categorical_cols].copy()  
if len(categorical_cols) > 0:  
    X_cate_dummies = pd.get_dummies(X_cate, dummy_na=True)  
else:  
    X_cate_dummies = pd.DataFrame(index=X.index)  

X_proc = pd.concat([X_numeric, X_cate_dummies], axis=1)  
features_proc = X_proc.columns.tolist()  
print(f"经缺失值预处理后, 保留的特征数: {len(features_proc)}")  

# 类别不平衡参数  
n_pos = (y == 1).sum()  
n_neg = (y == 0).sum()  
scale_pos_weight = n_neg / n_pos if n_pos != 0 else 1  
print(f"正例数: {n_pos}, 负例数: {n_neg}, scale_pos_weight: {scale_pos_weight:.2f}")  

print("===== 使用 LightGBM 进行特征重要性排序 =====")  
# ------------ LightGBM -----------  
lgb_model = lgb.LGBMClassifier(n_jobs=-1,  
                               random_state=42,  
                               scale_pos_weight=scale_pos_weight)  
lgb_model.fit(X_lgb, y)  
print("LightGBM 模型训练完成")  

# LightGBM 非0重要性特征  
lgb_importances_all = pd.Series(lgb_model.feature_importances_, index=features)  
# 与预处理特征集对齐  
lgb_importances = lgb_importances_all.loc[lgb_importances_all.index.intersection(features_proc)]  

lgb_nonzero_feats = lgb_importances[lgb_importances > 0].index  
print(f"LightGBM重要性>0 的特征数: {len(lgb_nonzero_feats)}")  

# 将全部特征的LightGBM得分导出Excel  
lgb_imp_df = pd.DataFrame({'Feature': lgb_importances.index, 'LGB_Importance': lgb_importances.values})  
lgb_imp_df.sort_values('LGB_Importance', ascending=False, inplace=True)  
lgb_imp_file = "lgb_feature_importances.xlsx"  
lgb_imp_df.to_excel(lgb_imp_file, index=False)  
print(f"【已导出】LightGBM全部特征重要性到: {lgb_imp_file}")  

# 选择LightGBM非零特征作为候选特征  
candidate_features = set(lgb_nonzero_feats)  
print(f"==== 候选特征数量（LightGBM>0）: {len(candidate_features)}")  

X_candidate = X_proc[list(candidate_features)]  

print("===== 对候选特征再用 L1、RFE 做精筛 =====")  
# --------- L1 正则 -----------  
lr_l1 = LogisticRegression(penalty="l1", solver="liblinear",  
                           max_iter=1000,  
                           class_weight="balanced",  
                           random_state=42)  
lr_l1.fit(X_candidate, y)  
l1_coef = pd.Series(np.abs(lr_l1.coef_[0]), index=X_candidate.columns)  
l1_ranks = l1_coef.rank(ascending=False, method='min')  
print("L1 正则化特征排序完成")  

# --------- RFE -----------  
subsample_size_for_rfe = 5000  
if X_candidate.shape[0] > subsample_size_for_rfe:  
    X_sub_rfe, _, y_sub_rfe, _ = train_test_split(X_candidate, y,  
                                                  train_size=subsample_size_for_rfe,  
                                                  stratify=y, random_state=42)  
    print(f"RFE计算仅在子样本 ({subsample_size_for_rfe}行) 上进行.")  
else:  
    X_sub_rfe, y_sub_rfe = X_candidate, y  
    print("RFE计算在完整候选集上进行.")  

estimator = LogisticRegression(solver="liblinear",  
                               max_iter=1000,  
                               class_weight="balanced",  
                               random_state=42)  
rfe = RFE(estimator, n_features_to_select=1, n_jobs=-1)  
rfe.fit(X_sub_rfe, y_sub_rfe)  
rfe_ranks = pd.Series(rfe.ranking_, index=X_sub_rfe.columns)  
print("RFE 特征选择完成")  

print("===== 整合 LightGBM、L1 和 RFE 排名 =====")  
final_features = list(candidate_features)  
total_features = len(final_features)  

def borda_score(rank_series):  
    return total_features - rank_series + 1  

# 将 LightGBM 分数映射回候选特征  
lgb_in_cand = pd.Series(lgb_importances.loc[final_features], index=final_features)  
lgb_ranks_candidate = lgb_in_cand.rank(ascending=False, method='min')  

borda_lgb = borda_score(lgb_ranks_candidate)  
borda_l1  = borda_score(l1_ranks)  
borda_rfe = borda_score(rfe_ranks)  

# 三种方法的总分  
borda_total = (borda_lgb + borda_l1 + borda_rfe).sort_values(ascending=False)  
print(f"Borda Count 整合完成, 最终候选特征数: {len(borda_total)}")  

print("===== 输出并保存综合排名结果 =====")  
top_n = 50  
print(f"Top {top_n} features based on integrated ranking:")  
print(borda_total.head(top_n))  

output_file = r"feature_integrated_ranking.csv"  
borda_total.to_csv(output_file, header=["Integrated_Score"])  
print(f"【已导出】综合特征排名结果到: {output_file}")  

print("===== 程序全部执行完成 =====")  