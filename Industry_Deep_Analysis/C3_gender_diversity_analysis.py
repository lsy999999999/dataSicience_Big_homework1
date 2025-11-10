"""
C3 行业深度分析: Gender Diversity与抗风险能力
========================================

核心问题:
1. 多样性高的团队/岗位是否更能适应AI变革?
2. 不同行业的性别多样性水平如何?
3. 多样性与自动化风险、薪资、增长率的关系?
4. "多样性红利"是否存在?
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
print("C3 分析: Gender Diversity与抗风险能力")
print("=" * 80)

# 创建输出目录
os.makedirs('Industry_Deep_Analysis/C3_outputs', exist_ok=True)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# 计算岗位变化
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# ============= 分析1: Gender Diversity整体分布 =============
print("\n" + "=" * 80)
print("分析1: Gender Diversity 整体分布")
print("=" * 80)

print("\nGender Diversity 统计:")
print(df['Gender Diversity (%)'].describe())

# 创建多样性级别分类
df['Diversity_Level'] = pd.cut(
    df['Gender Diversity (%)'],
    bins=[0, 40, 45, 55, 60, 100],
    labels=['低多样性(0-40%)', '中低多样性(40-45%)', '均衡(45-55%)', '中高多样性(55-60%)', '高多样性(60-100%)']
)

print("\nDiversity Level 分布:")
print(df['Diversity_Level'].value_counts().sort_index())

# 按多样性级别统计
diversity_stats = df.groupby('Diversity_Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
diversity_stats = diversity_stats.rename(columns={'Job Title': 'Count'})

print("\n多样性级别整体统计:")
print(diversity_stats)

diversity_stats.to_csv('Industry_Deep_Analysis/C3_outputs/diversity_level_stats.csv')

# ============= 分析2: 行业的多样性水平 =============
print("\n" + "=" * 80)
print("分析2: 各行业的多样性水平")
print("=" * 80)

industry_diversity = df.groupby('Industry').agg({
    'Gender Diversity (%)': 'mean',
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
industry_diversity = industry_diversity.rename(columns={'Job Title': 'Count'})
industry_diversity = industry_diversity.sort_values('Gender Diversity (%)', ascending=False)

print("\n各行业多样性排名:")
print(industry_diversity)

industry_diversity.to_csv('Industry_Deep_Analysis/C3_outputs/industry_diversity.csv')

# ============= 分析3: 多样性与风险/薪资的相关性 =============
print("\n" + "=" * 80)
print("分析3: 多样性与关键指标的相关性")
print("=" * 80)

# 整体相关性
corr_risk = df['Gender Diversity (%)'].corr(df['Automation Risk (%)'])
corr_salary = df['Gender Diversity (%)'].corr(df['Median Salary (USD)'])
corr_growth = df['Gender Diversity (%)'].corr(df['Openings_Pct_Change'])

print(f"\n整体相关系数:")
print(f"  多样性 vs 自动化风险: {corr_risk:.4f}")
print(f"  多样性 vs 薪资:       {corr_salary:.4f}")
print(f"  多样性 vs 增长率:     {corr_growth:.4f}")

# 各行业的相关性
print("\n各行业内部: 多样性 vs 自动化风险 相关系数:")
industry_corr_risk = df.groupby('Industry').apply(
    lambda x: x['Gender Diversity (%)'].corr(x['Automation Risk (%)'])
).sort_values()
print(industry_corr_risk)

print("\n各行业内部: 多样性 vs 薪资 相关系数:")
industry_corr_salary = df.groupby('Industry').apply(
    lambda x: x['Gender Diversity (%)'].corr(x['Median Salary (USD)'])
).sort_values()
print(industry_corr_salary)

# ============= 分析4: 行业×多样性交叉效应 =============
print("\n" + "=" * 80)
print("分析4: 行业×多样性级别交叉分析")
print("=" * 80)

industry_diversity_cross = df.groupby(['Industry', 'Diversity_Level']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
industry_diversity_cross = industry_diversity_cross.rename(columns={'Job Title': 'Count'})

print("\n行业×多样性交叉 (样本量>50):")
print(industry_diversity_cross[industry_diversity_cross['Count'] > 50])

industry_diversity_cross.to_csv('Industry_Deep_Analysis/C3_outputs/industry_diversity_cross.csv')

# ============= 分析5: "多样性红利"验证 =============
print("\n" + "=" * 80)
print("分析5: '多样性红利'是否存在?")
print("=" * 80)

# 对比高多样性 vs 低多样性岗位
high_div = df[df['Gender Diversity (%)'] >= 60]
low_div = df[df['Gender Diversity (%)'] <= 40]

print(f"\n高多样性岗位 (≥60%): {len(high_div):,} 个")
print(f"  平均风险: {high_div['Automation Risk (%)'].mean():.2f}%")
print(f"  平均薪资: ${high_div['Median Salary (USD)'].mean():,.0f}")
print(f"  平均增长: {high_div['Openings_Pct_Change'].mean():.2f}%")

print(f"\n低多样性岗位 (≤40%): {len(low_div):,} 个")
print(f"  平均风险: {low_div['Automation Risk (%)'].mean():.2f}%")
print(f"  平均薪资: ${low_div['Median Salary (USD)'].mean():,.0f}")
print(f"  平均增长: {low_div['Openings_Pct_Change'].mean():.2f}%")

# ============= 可视化部分 =============
print("\n开始生成可视化...")
sns.set_style("whitegrid")

# 图1: 多样性级别整体效应 (2×2)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1.1 多样性 vs 风险
diversity_sorted = diversity_stats.sort_index()
x_pos = np.arange(len(diversity_sorted))
colors_div = ['red', 'orange', 'green', 'orange', 'red']
bars = axes[0, 0].bar(x_pos, diversity_sorted['Automation Risk (%)'], color=colors_div, alpha=0.7)
axes[0, 0].set_title('多样性级别 vs 自动化风险\n均衡的团队更安全?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=14)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(diversity_sorted.index, fontproperties=font,
                            fontsize=11, rotation=20, ha='right')
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(diversity_sorted['Automation Risk (%)']):
    axes[0, 0].text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                    fontproperties=font, fontsize=11, fontweight='bold')

# 1.2 多样性 vs 薪资
bars = axes[0, 1].bar(x_pos, diversity_sorted['Median Salary (USD)'], color='steelblue', alpha=0.7)
axes[0, 1].set_title('多样性级别 vs 薪资\n存在"多样性溢价"吗?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(diversity_sorted.index, fontproperties=font,
                            fontsize=11, rotation=20, ha='right')
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(diversity_sorted['Median Salary (USD)']):
    axes[0, 1].text(i, val + 1000, f'${val:,.0f}', ha='center', va='bottom',
                    fontproperties=font, fontsize=10, fontweight='bold')

# 1.3 多样性 vs 增长率
bars = axes[1, 0].bar(x_pos, diversity_sorted['Openings_Pct_Change'],
                      color=['red' if x < 0 else 'green' for x in diversity_sorted['Openings_Pct_Change']],
                      alpha=0.7)
axes[1, 0].set_title('多样性级别 vs 岗位增长率\n多样性团队更有未来?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_ylabel('平均岗位增长率 (%)', fontproperties=font, fontsize=14)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(diversity_sorted.index, fontproperties=font,
                            fontsize=11, rotation=20, ha='right')
axes[1, 0].axhline(0, color='black', linewidth=2)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(diversity_sorted['Openings_Pct_Change']):
    axes[1, 0].text(i, val + 2 if val > 0 else val - 2, f'{val:.1f}%',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontproperties=font, fontsize=11, fontweight='bold')

# 1.4 样本量分布
bars = axes[1, 1].bar(x_pos, diversity_sorted['Count'], color='coral', alpha=0.7)
axes[1, 1].set_title('多样性级别的岗位分布',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_ylabel('岗位数量', fontproperties=font, fontsize=14)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(diversity_sorted.index, fontproperties=font,
                            fontsize=11, rotation=20, ha='right')
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(diversity_sorted['Count']):
    axes[1, 1].text(i, val + 100, f'{val:,}', ha='center', va='bottom',
                    fontproperties=font, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C3_outputs/01_diversity_level_overview.png',
            dpi=300, bbox_inches='tight')
print("✓ 01_diversity_level_overview.png")
plt.close()

# 图2: 行业多样性对比
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 2.1 行业多样性排名
y_pos = np.arange(len(industry_diversity))
axes[0, 0].barh(y_pos, industry_diversity['Gender Diversity (%)'], color='teal', alpha=0.7)
axes[0, 0].set_title('各行业性别多样性水平\n哪些行业更包容?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('平均性别多样性 (%)', fontproperties=font, fontsize=14)
axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(industry_diversity.index, fontproperties=font, fontsize=13)
axes[0, 0].axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='完全均衡(50%)')
for label in axes[0, 0].get_xticklabels():
    label.set_fontproperties(font)
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].legend(prop=font, fontsize=12)
for i, val in enumerate(industry_diversity['Gender Diversity (%)']):
    axes[0, 0].text(val + 0.5, i, f'{val:.1f}%', va='center',
                    fontproperties=font, fontsize=11)

# 2.2 多样性 vs 自动化风险散点图
axes[0, 1].scatter(industry_diversity['Gender Diversity (%)'],
                   industry_diversity['Automation Risk (%)'],
                   s=industry_diversity['Count']*0.5,
                   c=industry_diversity['Median Salary (USD)'],
                   cmap='YlGnBu',
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=1.5)
axes[0, 1].set_title('行业: 多样性 vs 自动化风险\n(气泡大小=岗位数, 颜色=薪资)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('平均性别多样性 (%)', fontproperties=font, fontsize=14)
axes[0, 1].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=14)
for idx, row in industry_diversity.iterrows():
    axes[0, 1].annotate(idx,
                        xy=(row['Gender Diversity (%)'], row['Automation Risk (%)']),
                        xytext=(5, 5), textcoords='offset points',
                        fontproperties=font, fontsize=11, alpha=0.8)
for label in axes[0, 1].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[0, 1].grid(True, alpha=0.3)

# 2.3 多样性 vs 薪资散点图
axes[1, 0].scatter(industry_diversity['Gender Diversity (%)'],
                   industry_diversity['Median Salary (USD)'],
                   s=industry_diversity['Count']*0.5,
                   c=industry_diversity['Automation Risk (%)'],
                   cmap='RdYlGn_r',
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=1.5)
axes[1, 0].set_title('行业: 多样性 vs 薪资\n(气泡大小=岗位数, 颜色=风险)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('平均性别多样性 (%)', fontproperties=font, fontsize=14)
axes[1, 0].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=14)
for idx, row in industry_diversity.iterrows():
    axes[1, 0].annotate(idx,
                        xy=(row['Gender Diversity (%)'], row['Median Salary (USD)']),
                        xytext=(5, 5), textcoords='offset points',
                        fontproperties=font, fontsize=11, alpha=0.8)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 0].grid(True, alpha=0.3)

# 2.4 多样性 vs 增长率散点图
axes[1, 1].scatter(industry_diversity['Gender Diversity (%)'],
                   industry_diversity['Openings_Pct_Change'],
                   s=industry_diversity['Count']*0.5,
                   c=industry_diversity['Median Salary (USD)'],
                   cmap='YlGnBu',
                   alpha=0.7,
                   edgecolors='black',
                   linewidth=1.5)
axes[1, 1].set_title('行业: 多样性 vs 岗位增长\n(气泡大小=岗位数, 颜色=薪资)',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('平均性别多样性 (%)', fontproperties=font, fontsize=14)
axes[1, 1].set_ylabel('平均岗位增长率 (%)', fontproperties=font, fontsize=14)
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
for idx, row in industry_diversity.iterrows():
    axes[1, 1].annotate(idx,
                        xy=(row['Gender Diversity (%)'], row['Openings_Pct_Change']),
                        xytext=(5, 5), textcoords='offset points',
                        fontproperties=font, fontsize=11, alpha=0.8)
for label in axes[1, 1].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C3_outputs/02_industry_diversity_comparison.png',
            dpi=300, bbox_inches='tight')
print("✓ 02_industry_diversity_comparison.png")
plt.close()

# 图3: 行业×多样性热力图
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# 3.1 风险热力图
pivot_risk_div = df.pivot_table(
    values='Automation Risk (%)',
    index='Industry',
    columns='Diversity_Level',
    aggfunc='mean'
)

sns.heatmap(pivot_risk_div, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 10})
axes[0].set_title('行业 × 多样性级别 → 自动化风险\n(绿=安全, 红=危险)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('多样性级别', fontproperties=font, fontsize=14)
axes[0].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(10)
    label.set_rotation(30)
    label.set_ha('right')
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[0].collections[0].colorbar
cbar.set_label('自动化风险 (%)', fontproperties=font, fontsize=12)

# 3.2 薪资热力图
pivot_salary_div = df.pivot_table(
    values='Median Salary (USD)',
    index='Industry',
    columns='Diversity_Level',
    aggfunc='mean'
)

sns.heatmap(pivot_salary_div, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '平均薪资 (USD)'},
            linewidths=1, linecolor='white', annot_kws={'size': 10})
axes[1].set_title('行业 × 多样性级别 → 平均薪资\n(深蓝=高薪)',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('多样性级别', fontproperties=font, fontsize=14)
axes[1].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(10)
    label.set_rotation(30)
    label.set_ha('right')
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
cbar = axes[1].collections[0].colorbar
cbar.set_label('平均薪资 (USD)', fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C3_outputs/03_industry_diversity_heatmaps.png',
            dpi=300, bbox_inches='tight')
print("✓ 03_industry_diversity_heatmaps.png")
plt.close()

print("\n" + "=" * 80)
print("C3 分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_diversity_level_overview.png - 多样性级别整体效应(四维)")
print("  2. 02_industry_diversity_comparison.png - 行业多样性对比(四维散点)")
print("  3. 03_industry_diversity_heatmaps.png - 行业×多样性热力图")
print("\n统计数据:")
print("  - diversity_level_stats.csv - 多样性级别统计")
print("  - industry_diversity.csv - 行业多样性数据")
print("  - industry_diversity_cross.csv - 行业×多样性交叉数据")
