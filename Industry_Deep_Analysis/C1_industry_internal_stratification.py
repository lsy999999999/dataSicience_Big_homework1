"""
C1 行业深度分析: 行业内部分层
========================================

核心问题:
1. 同一行业内,AI Impact Level如何影响风险和薪资?
2. 能否从Job Title中提取职位类型规律?
3. IT内的"赢家"和"输家"是谁?
4. 每个行业内部的异质性有多大?

分析框架:
- 维度1: Industry (8个行业)
- 维度2: AI Impact Level (High/Moderate/Low)
- 维度3: Job Type (从Job Title提取)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import seaborn as sns
import warnings
import sys
import os
from collections import Counter
import re
warnings.filterwarnings('ignore')

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
font = FontProperties(family='Microsoft YaHei', size=12)

print("=" * 80)
print("C1 分析: 行业内部分层——揭示同行业内的赢家与输家")
print("=" * 80)

# 创建输出目录
os.makedirs('Industry_Deep_Analysis/C1_outputs', exist_ok=True)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# 计算岗位变化
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# ============= 分析1: AI Impact Level的分布和效应 =============
print("\n" + "=" * 80)
print("分析1: AI Impact Level的行业分布")
print("=" * 80)

print("\nAI Impact Level 整体分布:")
print(df['AI Impact Level'].value_counts())

# 按行业和AI影响级别交叉分析
industry_ai_impact = df.groupby(['Industry', 'AI Impact Level']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
industry_ai_impact = industry_ai_impact.rename(columns={'Job Title': 'Count'})

print("\n各行业×AI影响级别统计 (前20行):")
print(industry_ai_impact.sort_values('Automation Risk (%)', ascending=False).head(20))

# 保存完整数据
industry_ai_impact.to_csv('Industry_Deep_Analysis/C1_outputs/industry_ai_impact_stats.csv')

# ============= 分析2: 从Job Title提取职位类型 =============
print("\n" + "=" * 80)
print("分析2: 职位类型分类 (从Job Title提取)")
print("=" * 80)

# 提取职位关键词
def classify_job_type(title):
    """
    根据职位名称中的关键词分类
    """
    title_lower = str(title).lower()

    # 管理类
    if any(word in title_lower for word in ['manager', 'director', 'executive', 'head', 'chief', 'president']):
        return 'Management'

    # 工程技术类
    elif any(word in title_lower for word in ['engineer', 'developer', 'programmer', 'architect', 'scientist']):
        return 'Engineering'

    # 分析咨询类
    elif any(word in title_lower for word in ['analyst', 'consultant', 'advisor', 'strategist']):
        return 'Analysis'

    # 医疗专业类
    elif any(word in title_lower for word in ['doctor', 'physician', 'nurse', 'surgeon', 'therapist', 'dentist']):
        return 'Medical_Professional'

    # 教育培训类
    elif any(word in title_lower for word in ['teacher', 'professor', 'instructor', 'educator', 'trainer']):
        return 'Education'

    # 创意设计类
    elif any(word in title_lower for word in ['designer', 'artist', 'creative', 'photographer', 'writer']):
        return 'Creative'

    # 销售市场类
    elif any(word in title_lower for word in ['sales', 'marketing', 'account', 'business development']):
        return 'Sales_Marketing'

    # 行政文员类
    elif any(word in title_lower for word in ['assistant', 'secretary', 'clerk', 'administrator', 'coordinator', 'officer']):
        return 'Administrative'

    # 技工操作类
    elif any(word in title_lower for word in ['technician', 'operator', 'driver', 'worker', 'mechanic']):
        return 'Technical_Operator'

    # 其他
    else:
        return 'Other'

df['Job_Type'] = df['Job Title'].apply(classify_job_type)

print("\n职位类型分布:")
job_type_dist = df['Job_Type'].value_counts()
print(job_type_dist)

# 按职位类型统计
job_type_stats = df.groupby('Job_Type').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
job_type_stats = job_type_stats.rename(columns={'Job Title': 'Count'})
job_type_stats = job_type_stats.sort_values('Automation Risk (%)', ascending=False)

print("\n职位类型风险排行:")
print(job_type_stats)

job_type_stats.to_csv('Industry_Deep_Analysis/C1_outputs/job_type_stats.csv')

# ============= 分析3: 行业内部异质性——以IT为例 =============
print("\n" + "=" * 80)
print("分析3: IT行业内部分层 (标杆案例)")
print("=" * 80)

it_data = df[df['Industry'] == 'IT']
print(f"\nIT行业总记录数: {len(it_data)}")

# IT内部按AI Impact分层
it_by_ai = it_data.groupby('AI Impact Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
it_by_ai = it_by_ai.rename(columns={'Job Title': 'Count'})

print("\nIT行业按AI影响级别分层:")
print(it_by_ai)

# IT内部按Job Type分层
it_by_job_type = it_data.groupby('Job_Type').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)
it_by_job_type = it_by_job_type.rename(columns={'Job Title': 'Count'})
it_by_job_type = it_by_job_type.sort_values('Automation Risk (%)')

print("\nIT行业按职位类型分层:")
print(it_by_job_type)

# ============= 分析4: 所有行业的内部异质性量化 =============
print("\n" + "=" * 80)
print("分析4: 各行业内部异质性量化")
print("=" * 80)

# 计算每个行业内部的标准差(异质性指标)
industry_heterogeneity = df.groupby('Industry').agg({
    'Automation Risk (%)': ['mean', 'std', 'min', 'max'],
    'Median Salary (USD)': ['mean', 'std', 'min', 'max']
}).round(2)

# 计算变异系数 (CV = std/mean)
industry_heterogeneity['Risk_CV'] = (
    industry_heterogeneity[('Automation Risk (%)', 'std')] /
    industry_heterogeneity[('Automation Risk (%)', 'mean')]
).round(3)

industry_heterogeneity['Salary_CV'] = (
    industry_heterogeneity[('Median Salary (USD)', 'std')] /
    industry_heterogeneity[('Median Salary (USD)', 'mean')]
).round(3)

print("\n各行业内部异质性 (变异系数越大=内部分化越严重):")
print(industry_heterogeneity)

industry_heterogeneity.to_csv('Industry_Deep_Analysis/C1_outputs/industry_heterogeneity.csv')

# ============= 可视化部分 =============
print("\n开始生成可视化...")
sns.set_style("whitegrid")

# 图1: 行业×AI Impact Level热力图 (2×2)
fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# 1.1 风险热力图
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Industry',
    columns='AI Impact Level',
    aggfunc='mean'
)
pivot_risk = pivot_risk[['Low', 'Moderate', 'High']]  # 排序

sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0, 0], cbar_kws={'label': '自动化风险 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 12})
axes[0, 0].set_title('行业 × AI影响级别 → 自动化风险\n(绿=安全, 红=危险)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('AI影响级别', fontproperties=font, fontsize=14)
axes[0, 0].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[0, 0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[0, 0].collections[0].colorbar
cbar.set_label('自动化风险 (%)', fontproperties=font, fontsize=12)

# 1.2 薪资热力图
pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Industry',
    columns='AI Impact Level',
    aggfunc='mean'
)
pivot_salary = pivot_salary[['Low', 'Moderate', 'High']]

sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[0, 1], cbar_kws={'label': '平均薪资 (USD)'},
            linewidths=1, linecolor='white', annot_kws={'size': 12})
axes[0, 1].set_title('行业 × AI影响级别 → 平均薪资\n(深蓝=高薪, 浅黄=低薪)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('AI影响级别', fontproperties=font, fontsize=14)
axes[0, 1].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[0, 1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[0, 1].collections[0].colorbar
cbar.set_label('平均薪资 (USD)', fontproperties=font, fontsize=12)

# 1.3 岗位数量分布
pivot_count = df.pivot_table(
    values='Job Title',
    index='Industry',
    columns='AI Impact Level',
    aggfunc='count'
)
pivot_count = pivot_count[['Low', 'Moderate', 'High']]

sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='Blues',
            ax=axes[1, 0], cbar_kws={'label': '岗位数量'},
            linewidths=1, linecolor='white', annot_kws={'size': 12})
axes[1, 0].set_title('行业 × AI影响级别 → 岗位数量分布\n(深蓝=岗位多)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('AI影响级别', fontproperties=font, fontsize=14)
axes[1, 0].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[1, 0].collections[0].colorbar
cbar.set_label('岗位数量', fontproperties=font, fontsize=12)

# 1.4 增长率热力图
pivot_growth = df.pivot_table(
    values='Openings_Pct_Change',
    index='Industry',
    columns='AI Impact Level',
    aggfunc='mean'
)
pivot_growth = pivot_growth[['Low', 'Moderate', 'High']]

sns.heatmap(pivot_growth, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=axes[1, 1], cbar_kws={'label': '岗位增长率 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 12})
axes[1, 1].set_title('行业 × AI影响级别 → 岗位增长率\n(绿=增长, 红=萎缩)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('AI影响级别', fontproperties=font, fontsize=14)
axes[1, 1].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[1, 1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[1, 1].collections[0].colorbar
cbar.set_label('岗位增长率 (%)', fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/01_industry_ai_impact_heatmaps.png',
            dpi=300, bbox_inches='tight')
print("✓ 01_industry_ai_impact_heatmaps.png")
plt.close()

# 图2: 职位类型风险排行
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 2.1 风险排行
job_type_stats_sorted = job_type_stats.sort_values('Automation Risk (%)', ascending=True)
colors_risk = ['green' if x < 40 else 'orange' if x < 45 else 'red'
               for x in job_type_stats_sorted['Automation Risk (%)']]
job_type_stats_sorted['Automation Risk (%)'].plot(kind='barh', ax=axes[0], color=colors_risk)
axes[0].set_title('职位类型自动化风险排行\n(绿=相对安全, 红=高危)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('平均自动化风险 (%)', fontproperties=font, fontsize=14)
axes[0].set_ylabel('职位类型', fontproperties=font, fontsize=14)
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
axes[0].axvline(df['Automation Risk (%)'].mean(), color='black',
                linestyle='--', linewidth=2, alpha=0.7, label='全局平均')
axes[0].legend(prop=font)
axes[0].grid(True, alpha=0.3)

# 2.2 薪资排行
job_type_stats_sorted_salary = job_type_stats.sort_values('Median Salary (USD)', ascending=True)
job_type_stats_sorted_salary['Median Salary (USD)'].plot(kind='barh', ax=axes[1], color='skyblue')
axes[1].set_title('职位类型薪资排行',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
axes[1].set_ylabel('职位类型', fontproperties=font, fontsize=14)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
axes[1].axvline(df['Median Salary (USD)'].mean(), color='black',
                linestyle='--', linewidth=2, alpha=0.7, label='全局平均')
axes[1].legend(prop=font)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/02_job_type_rankings.png',
            dpi=300, bbox_inches='tight')
print("✓ 02_job_type_rankings.png")
plt.close()

# 图3: IT行业内部分层案例
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 3.1 IT按AI Impact分层
it_ai_sorted = it_by_ai.sort_values('Automation Risk (%)')
x_pos = np.arange(len(it_ai_sorted))
bars = axes[0, 0].bar(x_pos, it_ai_sorted['Automation Risk (%)'],
                      color=['green', 'orange', 'red'])
axes[0, 0].set_title('IT行业: AI影响级别 vs 自动化风险',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[0, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=13)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(it_ai_sorted.index, fontproperties=font, fontsize=12)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 0].grid(True, alpha=0.3, axis='y')
# 添加数值标签
for i, (idx, row) in enumerate(it_ai_sorted.iterrows()):
    axes[0, 0].text(i, row['Automation Risk (%)'] + 1, f"{row['Automation Risk (%)']:.1f}%",
                    ha='center', va='bottom', fontproperties=font, fontsize=11, fontweight='bold')

# 3.2 IT按AI Impact薪资
bars = axes[0, 1].bar(x_pos, it_ai_sorted['Median Salary (USD)'], color='steelblue')
axes[0, 1].set_title('IT行业: AI影响级别 vs 薪资',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[0, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(it_ai_sorted.index, fontproperties=font, fontsize=12)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, (idx, row) in enumerate(it_ai_sorted.iterrows()):
    axes[0, 1].text(i, row['Median Salary (USD)'] + 2000, f"${row['Median Salary (USD)']:,.0f}",
                    ha='center', va='bottom', fontproperties=font, fontsize=10, fontweight='bold')

# 3.3 IT按Job Type风险
it_job_sorted = it_by_job_type[it_by_job_type['Count'] >= 30].sort_values('Automation Risk (%)', ascending=True)
y_pos = np.arange(len(it_job_sorted))
colors_it = ['green' if x < 40 else 'orange' if x < 45 else 'red'
             for x in it_job_sorted['Automation Risk (%)']]
axes[1, 0].barh(y_pos, it_job_sorted['Automation Risk (%)'], color=colors_it)
axes[1, 0].set_title('IT行业: 职位类型 vs 自动化风险\n(样本量≥30)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[1, 0].set_xlabel('平均自动化风险 (%)', fontproperties=font, fontsize=13)
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(it_job_sorted.index, fontproperties=font, fontsize=11)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 3.4 IT按Job Type薪资
axes[1, 1].barh(y_pos, it_job_sorted['Median Salary (USD)'], color='skyblue')
axes[1, 1].set_title('IT行业: 职位类型 vs 薪资\n(样本量≥30)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[1, 1].set_xlabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
axes[1, 1].set_yticks(y_pos)
axes[1, 1].set_yticklabels(it_job_sorted.index, fontproperties=font, fontsize=11)
for label in axes[1, 1].get_xticklabels():
    label.set_fontproperties(font)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/03_IT_internal_stratification.png',
            dpi=300, bbox_inches='tight')
print("✓ 03_IT_internal_stratification.png")
plt.close()

# 图4: 行业内部异质性对比
fig, ax = plt.subplots(figsize=(16, 10))

industries = industry_heterogeneity.index.tolist()
risk_cv = industry_heterogeneity['Risk_CV'].values
salary_cv = industry_heterogeneity['Salary_CV'].values

x = np.arange(len(industries))
width = 0.35

bars1 = ax.bar(x - width/2, risk_cv, width, label='风险变异系数', color='coral', alpha=0.8)
bars2 = ax.bar(x + width/2, salary_cv, width, label='薪资变异系数', color='skyblue', alpha=0.8)

ax.set_title('各行业内部异质性对比\n(变异系数越大 = 内部分化越严重)',
             fontproperties=font, fontsize=17, fontweight='bold', pad=20)
ax.set_xlabel('行业', fontproperties=font, fontsize=15)
ax.set_ylabel('变异系数 (CV = 标准差/均值)', fontproperties=font, fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(industries, fontproperties=font, fontsize=13, rotation=45, ha='right')
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
ax.legend(prop=font, fontsize=14, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom',
                fontproperties=font, fontsize=9)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/04_industry_heterogeneity.png',
            dpi=300, bbox_inches='tight')
print("✓ 04_industry_heterogeneity.png")
plt.close()

print("\n" + "=" * 80)
print("C1 分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_industry_ai_impact_heatmaps.png - 行业×AI影响级别四维热力图")
print("  2. 02_job_type_rankings.png - 职位类型风险和薪资排行")
print("  3. 03_IT_internal_stratification.png - IT行业内部分层详解")
print("  4. 04_industry_heterogeneity.png - 行业内部异质性对比")
print("\n统计数据:")
print("  - industry_ai_impact_stats.csv - 行业×AI影响级别详细数据")
print("  - job_type_stats.csv - 职位类型统计")
print("  - industry_heterogeneity.csv - 行业异质性指标")
