# -*- coding: utf-8 -*-  
"""  
Created on Thu Apr  3 15:55:28 2025  

@author: ybh4600  
"""  

# 导入所需的第三方包：  
import pandas as pd           # pandas：用于数据读取、处理和操作，非常常用  
import numpy as np            # numpy：用于数值计算，提供了高效的数组操作  
from sklearn.linear_model import LogisticRegression   # 从scikit-learn中导入逻辑回归，用于构建分类模型  
from sklearn.feature_selection import RFE, mutual_info_classif  # 导入特征选择工具：RFE（递归特征消除）和互信息计算  
import lightgbm as lgb        # 导入LightGBM模块，用于构建高效的梯度提升机模型  
from sklearn.model_selection import train_test_split  # 导入数据集拆分函数  

print("===== 程序开始：数据读取 =====")  
# ------------------------  
# 基本设置  
# ------------------------  
# 指定数据文件的完整路径（读取 CSV 格式的数据文件）  
input_file_path = r"C:\Users\ybh4600\Desktop\第二届瑞智杯建模大赛\1缺失值处理\out_trn_data_cleaned_int.csv"  

# 读取 CSV 文件内容到一个 DataFrame 对象，DataFrame 类似于表格数据结构（行、列）  
df = pd.read_csv(input_file_path)  
print("数据读取完成，数据形状：", df.shape)  

# 根据字段名称筛选出特征变量（自变量），假设所有自变量字段名均以 "x" 开头  
features = [col for col in df.columns if col.startswith("x")]  
# X 为自变量数据，只包含特征 (字段名以 "x" 开头)  
X = df[features]  
# y 为因变量（标签），假定标签的字段名称为 "y"  
y = df["y"]  
print("特征个数：", len(features), "，标签读取完成")  

print("===== 开始区分数值型与类别型字段 =====")  
# ------------------------  
# 区分数值型与类别型字段  
# ------------------------  
# 使用 pandas 的 select_dtypes 方法来获取数据类型分别为数值型的数据字段名称（如 int、float）  
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()  
# 使用 pandas 的 select_dtypes 方法选择非数值型的字段（通常为文本、类别型数据）  
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()  
print("数值型字段数:", len(numeric_cols))  
print("类别型字段数:", len(categorical_cols))  

print("===== 为 LightGBM 准备数据（不预处理缺失值） =====")  
# ------------------------  
# 1. 针对 LightGBM 的数据（原始数据，无缺失值处理）  
# ------------------------  
# 复制一份自变量数据，用于 LightGBM 模型（因为 LightGBM 可以自动处理缺失值和类别变量）  
X_lgb = X.copy()  
# 遍历所有类别型字段，将每个字段转换为 pandas 中的 category 类型，LightGBM 能针对该数据类型进行特殊处理  
for c in categorical_cols:  
    X_lgb[c] = X_lgb[c].astype('category')  
print("LightGBM 数据准备完成，数据形状：", X_lgb.shape)  

print("===== 针对其他算法预处理数据：缺失值处理 =====")  
# ------------------------  
# 2. 针对其他算法的数据（预处理缺失值）  
# ------------------------  
# 数值型字段处理：  
# 先计算每个数值字段的缺失率（缺失值的比例），.isnull() 检查数据是否为空，.mean() 计算比例  
missing_rate_num = X[numeric_cols].isnull().mean()  
# 保留那些缺失率小于等于 30% 的数值型字段，删除缺失率大于 30% 的字段  
numeric_keep = [c for c in numeric_cols if missing_rate_num[c] <= 0.3]  
print("保留的数值型字段数：", len(numeric_keep))  
# 复制保留的数值型数据  
X_numeric = X[numeric_keep].copy()  
# 对数值型字段中的缺失值使用每一列的中位数进行填充（填充为空值的地方）  
X_numeric = X_numeric.fillna(X_numeric.median())  

# 类别型字段处理：  
# 复制原始类别型数据（注意：这里不进行缺失值的填充处理，保持原状态）  
X_cate = X[categorical_cols].copy()  
# 如果存在类别型字段，则使用 pd.get_dummies 进行 One-Hot 编码转换，同时设置 dummy_na=True，使得缺失值也被转换为一个独立的类别  
if len(categorical_cols) > 0:  
    X_cate_dummies = pd.get_dummies(X_cate, dummy_na=True)  
else:  
    # 如果没有类别型字段，则创建一个空的 DataFrame，行索引与 X 保持一致  
    X_cate_dummies = pd.DataFrame(index=X.index)  
print("类别型字段经过 One-Hot 编码后，新特征数：", X_cate_dummies.shape[1])  

# 组合经过预处理的数值型数据和编码后的类别型数据（按列进行组合）  
X_proc = pd.concat([X_numeric, X_cate_dummies], axis=1)  
# 获取组合后数据的所有特征名称列表  
features_proc = X_proc.columns.tolist()  
# 打印预处理后剩余的特征数量  
print("经缺失值预处理后剩余特征数:", len(features_proc))  

print("===== 设定类别不平衡参数 =====")  
# ------------------------  
# 3. 设定类别不平衡参数  
# ------------------------  
# 假设因变量 y 中 y=1 表示好客户（少数类），y=0 表示坏客户（多数类）  
n_pos = (y == 1).sum()  # 计算标签中 y 等于 1 的样本数  
n_neg = (y == 0).sum()  # 计算标签中 y 等于 0 的样本数  
# 计算 LightGBM 中使用的类别不平衡参数 scale_pos_weight （一般为负样本数/正样本数），若正样本数量为 0 则设为 1  
scale_pos_weight = n_neg / n_pos if n_pos != 0 else 1  
print("正例数：{}, 负例数：{}, scale_pos_weight：{:.2f}".format(n_pos, n_neg, scale_pos_weight))  

print("===== 抽取子样本，用于加快 RFE 和互信息计算 =====")  
# ------------------------  
# 4. 为加快部分计算，在预处理后的数据中抽取子样本供 RFE 和互信息计算  
# ------------------------  
subsample_size = 5000  # 定义抽取子样本的大小，取 5000 行记录（可以根据计算资源进行调整）  
# 使用 train_test_split 函数从预处理后的数据中随机抽取子样本，  
# 参数 stratify=y 表示按照 y 中的类别比例进行分层抽样，  
# random_state=42 保证每次抽样结果一致，  
# 这里只保留抽取的训练部分，其它部分用 _ 表示忽略  
X_sub, _, y_sub, _ = train_test_split(X_proc, y, train_size=subsample_size, stratify=y, random_state=42)  
print("子样本抽取完成，子样本数据形状：", X_sub.shape)  

print("===== 特征筛选方法：LightGBM 特征重要性 =====")  
# ------------------------  
# 5. 特征筛选方法  
# ------------------------  
# 5.1 LightGBM 特征重要性（使用原始数据 X_lgb）  
lgb_model = lgb.LGBMClassifier(n_jobs=-1,  
                               random_state=42,  
                               scale_pos_weight=scale_pos_weight)  
# 用 LightGBM 模型对原始数据 X_lgb 和标签 y 进行训练（fit）  
lgb_model.fit(X_lgb, y)  
print("LightGBM 模型训练完成")  
# 获取训练后 LightGBM 模型的各特征重要性分数，返回的是一个数组  
lgb_importances = pd.Series(lgb_model.feature_importances_, index=features)  
# 只保留那些经过预处理后保留下来的特征分数，确保与 X_proc 的特征一致  
lgb_importances = lgb_importances.loc[features_proc]  
# 对 LightGBM 返回的特征重要性进行排名：数值越高，排名越靠前  
lgb_ranks = lgb_importances.rank(ascending=False, method='min')  
print("LightGBM 特征重要性计算并排序完成")  

print("===== 特征筛选方法：L1 正则化 (LogisticRegression) =====")  
# 5.2 L1 正则化（基于 LogisticRegression）的特征选择（使用预处理后的数据 X_proc）  
lr_l1 = LogisticRegression(penalty="l1",  
                           solver="liblinear",  
                           max_iter=1000,  
                           class_weight="balanced",  
                           random_state=42)  
# 用逻辑回归模型拟合预处理后的数据和标签  
lr_l1.fit(X_proc, y)  
print("L1 正则化模型训练完成")  
# 提取逻辑回归模型的系数（coef_），由于是多维数组，所以取第一行，之后用 np.abs 求取绝对值  
lr_coef = pd.Series(np.abs(lr_l1.coef_[0]), index=features_proc)  
# 对模型系数进行降序排序排名  
lr_ranks = lr_coef.rank(ascending=False, method='min')  
print("L1 正则化特征排序完成")  

print("===== 特征筛选方法：RFE（递归特征消除） =====")  
# 5.3 使用递归特征消除（RFE）进行特征选择  
estimator = LogisticRegression(solver="liblinear",  
                               max_iter=1000,  
                               class_weight="balanced",  
                               random_state=42)  
# 创建 RFE 对象，设置 n_features_to_select=1 表示对所有特征进行排名，n_jobs=-1 表示使用所有 CPU 核心  
rfe = RFE(estimator, n_features_to_select=1, n_jobs=-1)  
# 在抽取的子样本上进行 RFE 特征选择  
rfe.fit(X_sub, y_sub)  
# 通过 rfe.ranking_ 得到每个特征的排名，数值越小表示特征越重要  
rfe_ranks = pd.Series(rfe.ranking_, index=X_sub.columns)  
print("RFE 特征选择完成")  

print("===== 特征筛选方法：互信息 =====")  
# 5.4 使用信息论方法（互信息）进行特征选择  
mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)  
# 将互信息得分转换为带特征名称索引的 Series 对象  
mi_scores_series = pd.Series(mi_scores, index=X_sub.columns)  
# 根据互信息得分进行排名，得分越高则排名越靠前  
mi_ranks = mi_scores_series.rank(ascending=False, method='min')  
print("互信息计算及排序完成")  

print("===== Borda Count 投票整合各方法的排名 =====")  
# ------------------------  
# 6. Borda Count 投票整合各方法的排名  
# ------------------------  
# 获取预处理后的特征总数  
total_features = len(features_proc)  

# 定义一个函数，将传入的排名 Series 转换成 Borda 得分  
def borda_score(rank_series):  
    # 返回：总特征数减去每个特征的排名再加1  
    return total_features - rank_series + 1  

# 分别计算四种特征选择方法的 Borda 得分  
borda_lgb = borda_score(lgb_ranks)  
borda_lr  = borda_score(lr_ranks)  
borda_rfe = borda_score(rfe_ranks)  
borda_mi  = borda_score(mi_ranks)  

# 将四种方法的得分逐项相加，得到每个特征的综合得分  
borda_total = borda_lgb + borda_lr + borda_rfe + borda_mi  
# 对综合得分按降序排序，得分越高表示特征重要性越大  
borda_total = borda_total.sort_values(ascending=False)  
print("Borda Count 整合完成")  

print("===== 输出并保存综合排名结果 =====")  
# ------------------------  
# 7. 输出并保存综合排名结果  
# ------------------------  
top_n = 50  # 指定输出前50个特征  
print("Top {} features based on integrated Borda Count:".format(top_n))  
print(borda_total.head(top_n))  
# 指定综合得分结果保存到的 CSV 文件路径  
output_file = r"feature_integrated_borda_count_ranking.csv"  
# 将综合得分保存到 CSV 文件中，header 参数指定列名为 "Borda_Score"  
borda_total.to_csv(output_file, header=["Borda_Score"])  
print("综合特征排名结果已保存到：", output_file)  
print("===== 程序全部执行完成 =====")  