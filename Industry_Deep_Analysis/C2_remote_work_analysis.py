"""
C2 行业深度分析: Remote Work的双刃剑效应
========================================

核心问题:
1. 能远程工作的岗位,是更安全(灵活性)还是更危险(易被外包/自动化)?
2. 不同行业的远程工作模式如何影响风险和薪资?
3. 远程工作比例与自动化风险的关系?
4. 疫情后的新工作模式如何改变职业格局?
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
print("C2 分析: Remote Work的双刃剑效应")
print("=" * 80)

# 创建输出目录
os.makedirs('Industry_Deep_Analysis/C2_outputs', exist_ok=True)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# 计算岗位变化
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# ============= 分析1: Remote Work整体分布 =============
print("\n" + "=" * 80)
print("分析1: Remote Work Ratio 整体分布")
print("=" * 80)

print("\nRemote Work Ratio 统计:")
print(df['Remote Work Ratio (%)'].describe())

# 创建远程工作级别分类
df['Remote_Level'] = pd.cut(
    df['Remote Work Ratio (%)'],
    bins=[0, 20, 50, 80, 100],
    labels=['低远程(0-20%)', '中低远程(20-50%)', '中高远程(50-80%)', '高远程(80-100%)']
)

print("\nRemote Level 分布:")
print(df['Remote_Level'].value_counts().sort_index())

# 按远程级别统计
remote_stats = df.groupby('Remote_Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
remote_stats = remote_stats.rename(columns={'Job Title': 'Count'})

print("\n远程工作级别整体统计:")
print(remote_stats)

remote_stats.to_csv('Industry_Deep_Analysis/C2_outputs/remote_level_stats.csv')

# ============= 分析2: 行业×远程工作交叉效应 =============
print("\n" + "=" * 80)
print("分析2: 行业×远程工作模式的交叉效应")
print("=" * 80)

# 各行业的平均远程工作比例
industry_remote = df.groupby('Industry').agg({
    'Remote Work Ratio (%)': 'mean',
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)
industry_remote = industry_remote.rename(columns={'Job Title': 'Count'})
industry_remote = industry_remote.sort_values('Remote Work Ratio (%)', ascending=False)

print("\n各行业远程工作比例:")
print(industry_remote)

# 行业×远程级别交叉
industry_remote_cross = df.groupby(['Industry', 'Remote_Level']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)
industry_remote_cross = industry_remote_cross.rename(columns={'Job Title': 'Count'})

print("\n行业×远程级别交叉统计 (样本量>50):")
print(industry_remote_cross[industry_remote_cross['Count'] > 50])

industry_remote_cross.to_csv('Industry_Deep_Analysis/C2_outputs/industry_remote_cross.csv')

# ============= 分析3: 远程工作的双刃剑——IT vs 其他 =============
print("\n" + "=" * 80)
print("分析3: 远程工作的行业异质性效应")
print("=" * 80)

# IT行业的远程效应
it_remote = df[df['Industry'] == 'IT'].groupby('Remote_Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)
it_remote = it_remote.rename(columns={'Job Title': 'Count'})

print("\nIT行业按远程级别:")
print(it_remote)

# Transportation行业的远程效应
trans_remote = df[df['Industry'] == 'Transportation'].groupby('Remote_Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)
trans_remote = trans_remote.rename(columns={'Job Title': 'Count'})

print("\nTransportation行业按远程级别:")
print(trans_remote)

# ============= 分析4: 远程工作与自动化风险的相关性 =============
print("\n" + "=" * 80)
print("分析4: 远程工作比例与自动化风险的相关性")
print("=" * 80)

# 整体相关性
overall_corr = df['Remote Work Ratio (%)'].corr(df['Automation Risk (%)'])
print(f"\n整体相关系数: {overall_corr:.4f}")

# 各行业的相关性
industry_corr = df.groupby('Industry').apply(
    lambda x: x['Remote Work Ratio (%)'].corr(x['Automation Risk (%)'])
).sort_values()

print("\n各行业内部相关性:")
print(industry_corr)

# ============= 分析5: 远程工作的"灵活性溢价" =============
print("\n" + "=" * 80)
print("分析5: 远程工作的薪资溢价/惩罚")
print("=" * 80)

# 控制行业后,远程工作对薪资的影响
industry_remote_salary = df.groupby(['Industry', 'Remote_Level'])['Median Salary (USD)'].mean().unstack()

print("\n各行业×远程级别的薪资矩阵:")
print(industry_remote_salary)

# ============= 可视化部分 =============
print("\n开始生成可视化...")
sns.set_style("whitegrid")

# 图1: Remote Level整体效应 (2×2)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1.1 远程级别 vs 风险
remote_sorted = remote_stats.sort_index()
x_pos = np.arange(len(remote_sorted))
colors_risk = ['green', 'orange', 'orange', 'red']
bars = axes[0, 0].bar(x_pos, remote_sorted['Automation Risk (%)'], color=colors_risk, alpha=0.7)
axes[0, 0].set_title('远程工作级别 vs 自动化风险\n远程比例越高,风险如何变化?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=14)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(remote_sorted.index, fontproperties=font, fontsize=12, rotation=15, ha='right')
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(remote_sorted['Automation Risk (%)']):
    axes[0, 0].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                    fontproperties=font, fontsize=11, fontweight='bold')

# 1.2 远程级别 vs 薪资
bars = axes[0, 1].bar(x_pos, remote_sorted['Median Salary (USD)'], color='steelblue', alpha=0.7)
axes[0, 1].set_title('远程工作级别 vs 薪资\n远程工作有"灵活性溢价"吗?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(remote_sorted.index, fontproperties=font, fontsize=12, rotation=15, ha='right')
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(remote_sorted['Median Salary (USD)']):
    axes[0, 1].text(i, val + 1000, f'${val:,.0f}', ha='center', va='bottom',
                    fontproperties=font, fontsize=10, fontweight='bold')

# 1.3 远程级别 vs 增长率
bars = axes[1, 0].bar(x_pos, remote_sorted['Openings_Pct_Change'],
                      color=['red' if x < 0 else 'green' for x in remote_sorted['Openings_Pct_Change']],
                      alpha=0.7)
axes[1, 0].set_title('远程工作级别 vs 岗位增长率\n哪类远程模式更有前景?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_ylabel('平均岗位增长率 (%)', fontproperties=font, fontsize=14)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(remote_sorted.index, fontproperties=font, fontsize=12, rotation=15, ha='right')
axes[1, 0].axhline(0, color='black', linewidth=2)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(remote_sorted['Openings_Pct_Change']):
    axes[1, 0].text(i, val + 2 if val > 0 else val - 2, f'{val:.1f}%',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontproperties=font, fontsize=11, fontweight='bold')

# 1.4 样本量分布
bars = axes[1, 1].bar(x_pos, remote_sorted['Count'], color='coral', alpha=0.7)
axes[1, 1].set_title('远程工作级别的岗位分布\n疫情后的新常态',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_ylabel('岗位数量', fontproperties=font, fontsize=14)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(remote_sorted.index, fontproperties=font, fontsize=12, rotation=15, ha='right')
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(remote_sorted['Count']):
    axes[1, 1].text(i, val + 100, f'{val:,}', ha='center', va='bottom',
                    fontproperties=font, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C2_outputs/01_remote_level_overview.png',
            dpi=300, bbox_inches='tight')
print("✓ 01_remote_level_overview.png")
plt.close()

# 图2: 行业远程工作比例对比
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 2.1 行业平均远程比例
industry_remote_sorted = industry_remote.sort_values('Remote Work Ratio (%)', ascending=True)
y_pos = np.arange(len(industry_remote_sorted))
axes[0].barh(y_pos, industry_remote_sorted['Remote Work Ratio (%)'], color='teal', alpha=0.7)
axes[0].set_title('各行业平均远程工作比例\n哪些行业更适合远程?',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('平均远程工作比例 (%)', fontproperties=font, fontsize=14)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(industry_remote_sorted.index, fontproperties=font, fontsize=13)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
axes[0].grid(True, alpha=0.3, axis='x')
for i, val in enumerate(industry_remote_sorted['Remote Work Ratio (%)']):
    axes[0].text(val + 1, i, f'{val:.1f}%', va='center',
                 fontproperties=font, fontsize=11)

# 2.2 远程比例 vs 薪资散点图
axes[1].scatter(industry_remote['Remote Work Ratio (%)'],
                industry_remote['Median Salary (USD)'],
                s=industry_remote['Count']*0.5,
                c=industry_remote['Automation Risk (%)'],
                cmap='RdYlGn_r',
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5)
axes[1].set_title('行业: 远程工作比例 vs 薪资\n(气泡大小=岗位数, 颜色=风险)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('平均远程工作比例 (%)', fontproperties=font, fontsize=14)
axes[1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
for idx, row in industry_remote.iterrows():
    axes[1].annotate(idx,
                     xy=(row['Remote Work Ratio (%)'], row['Median Salary (USD)']),
                     xytext=(5, 5), textcoords='offset points',
                     fontproperties=font, fontsize=11, alpha=0.8)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C2_outputs/02_industry_remote_comparison.png',
            dpi=300, bbox_inches='tight')
print("✓ 02_industry_remote_comparison.png")
plt.close()

# 图3: 行业×远程级别热力图
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 3.1 风险热力图
pivot_risk_remote = df.pivot_table(
    values='Automation Risk (%)',
    index='Industry',
    columns='Remote_Level',
    aggfunc='mean'
)

sns.heatmap(pivot_risk_remote, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 11})
axes[0].set_title('行业 × 远程级别 → 自动化风险\n(绿=安全, 红=危险)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('远程工作级别', fontproperties=font, fontsize=14)
axes[0].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
    label.set_rotation(30)
    label.set_ha('right')
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[0].collections[0].colorbar
cbar.set_label('自动化风险 (%)', fontproperties=font, fontsize=12)

# 3.2 薪资热力图
pivot_salary_remote = df.pivot_table(
    values='Median Salary (USD)',
    index='Industry',
    columns='Remote_Level',
    aggfunc='mean'
)

sns.heatmap(pivot_salary_remote, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '平均薪资 (USD)'},
            linewidths=1, linecolor='white', annot_kws={'size': 11})
axes[1].set_title('行业 × 远程级别 → 平均薪资\n(深蓝=高薪)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('远程工作级别', fontproperties=font, fontsize=14)
axes[1].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
    label.set_rotation(30)
    label.set_ha('right')
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[1].collections[0].colorbar
cbar.set_label('平均薪资 (USD)', fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C2_outputs/03_industry_remote_heatmaps.png',
            dpi=300, bbox_inches='tight')
print("✓ 03_industry_remote_heatmaps.png")
plt.close()

# 图4: IT vs Transportation 远程效应对比
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 准备数据
it_remote_clean = it_remote[it_remote['Count'] > 30]
trans_remote_clean = trans_remote[trans_remote['Count'] > 30]

if len(it_remote_clean) > 0:
    x_it = np.arange(len(it_remote_clean))

    # 4.1 IT风险
    axes[0, 0].bar(x_it, it_remote_clean['Automation Risk (%)'],
                   color='green', alpha=0.7)
    axes[0, 0].set_title('IT行业: 远程级别 vs 自动化风险\n远程=优势',
                         fontproperties=font, fontsize=15, fontweight='bold', pad=12)
    axes[0, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=13)
    axes[0, 0].set_xticks(x_it)
    axes[0, 0].set_xticklabels(it_remote_clean.index, fontproperties=font,
                                fontsize=11, rotation=20, ha='right')
    for label in axes[0, 0].get_yticklabels():
        label.set_fontproperties(font)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(it_remote_clean['Automation Risk (%)']):
        axes[0, 0].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                        fontproperties=font, fontsize=11, fontweight='bold')

    # 4.2 IT薪资
    axes[0, 1].bar(x_it, it_remote_clean['Median Salary (USD)'],
                   color='steelblue', alpha=0.7)
    axes[0, 1].set_title('IT行业: 远程级别 vs 薪资',
                         fontproperties=font, fontsize=15, fontweight='bold', pad=12)
    axes[0, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
    axes[0, 1].set_xticks(x_it)
    axes[0, 1].set_xticklabels(it_remote_clean.index, fontproperties=font,
                                fontsize=11, rotation=20, ha='right')
    for label in axes[0, 1].get_yticklabels():
        label.set_fontproperties(font)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(it_remote_clean['Median Salary (USD)']):
        axes[0, 1].text(i, val + 1000, f'${val:,.0f}', ha='center', va='bottom',
                        fontproperties=font, fontsize=10, fontweight='bold')

if len(trans_remote_clean) > 0:
    x_trans = np.arange(len(trans_remote_clean))

    # 4.3 Transportation风险
    axes[1, 0].bar(x_trans, trans_remote_clean['Automation Risk (%)'],
                   color='red', alpha=0.7)
    axes[1, 0].set_title('Transportation: 远程级别 vs 自动化风险\n远程=劣势?',
                         fontproperties=font, fontsize=15, fontweight='bold', pad=12)
    axes[1, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=13)
    axes[1, 0].set_xticks(x_trans)
    axes[1, 0].set_xticklabels(trans_remote_clean.index, fontproperties=font,
                                fontsize=11, rotation=20, ha='right')
    for label in axes[1, 0].get_yticklabels():
        label.set_fontproperties(font)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(trans_remote_clean['Automation Risk (%)']):
        axes[1, 0].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                        fontproperties=font, fontsize=11, fontweight='bold')

    # 4.4 Transportation薪资
    axes[1, 1].bar(x_trans, trans_remote_clean['Median Salary (USD)'],
                   color='coral', alpha=0.7)
    axes[1, 1].set_title('Transportation: 远程级别 vs 薪资',
                         fontproperties=font, fontsize=15, fontweight='bold', pad=12)
    axes[1, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
    axes[1, 1].set_xticks(x_trans)
    axes[1, 1].set_xticklabels(trans_remote_clean.index, fontproperties=font,
                                fontsize=11, rotation=20, ha='right')
    for label in axes[1, 1].get_yticklabels():
        label.set_fontproperties(font)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, val in enumerate(trans_remote_clean['Median Salary (USD)']):
        axes[1, 1].text(i, val + 1000, f'${val:,.0f}', ha='center', va='bottom',
                        fontproperties=font, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C2_outputs/04_IT_vs_Transportation_remote.png',
            dpi=300, bbox_inches='tight')
print("✓ 04_IT_vs_Transportation_remote.png")
plt.close()

print("\n" + "=" * 80)
print("C2 分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_remote_level_overview.png - 远程级别整体效应(四维)")
print("  2. 02_industry_remote_comparison.png - 行业远程工作对比")
print("  3. 03_industry_remote_heatmaps.png - 行业×远程级别热力图")
print("  4. 04_IT_vs_Transportation_remote.png - IT vs Transportation远程效应对比")
print("\n统计数据:")
print("  - remote_level_stats.csv - 远程级别统计")
print("  - industry_remote_cross.csv - 行业×远程级别交叉数据")
