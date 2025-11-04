"""
B2 分析: 绘制"战场"——结构性重塑的地图

目标: 识别哪些行业(Industry)和地区(Location)是AI时代的"热点"和"冷点"
核心问题:
1. 哪些行业正在岗位净增长?(热点战场)
2. 哪些行业正在岗位净减少?(冷点战场)
3. 不同地区的行业表现如何?(地区放大效应)
4. 这种重塑如何改变"盔甲"(教育/经验)的价值?(为交叉分析铺垫)
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
print("B2 分析: 绘制'战场'——AI重塑的行业与地区地图")
print("=" * 80)

# 创建输出目录
if not os.path.exists('B2_outputs'):
    os.makedirs('B2_outputs')

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# ============= 数据预处理 =============
# 计算岗位变化
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# 检查数据分布
print(f"\n行业分布: {df['Industry'].nunique()} 个行业")
print(df['Industry'].value_counts())

print(f"\n地区分布: {df['Location'].nunique()} 个地区")
print(df['Location'].value_counts())

print(f"\n岗位状态分布:")
print(df['Job Status'].value_counts())

# ============= 核心洞察 1: 行业的"热点"与"冷点" =============
print("\n" + "=" * 80)
print("核心洞察 1: 行业战场——谁在扩张?谁在萎缩?")
print("=" * 80)

# 按行业汇总
industry_stats = df.groupby('Industry').agg({
    'Job Openings (2024)': 'sum',
    'Projected Openings (2030)': 'sum',
    'Openings_Abs_Change': 'sum',
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

industry_stats['Openings_Pct_Change'] = (
    industry_stats['Openings_Abs_Change'] /
    industry_stats['Job Openings (2024)'] * 100
).round(2)

industry_stats = industry_stats.rename(columns={'Job Title': 'Job_Count'})
industry_stats = industry_stats.sort_values('Openings_Pct_Change', ascending=False)

print("\n行业排名 (按岗位增长率):")
print(industry_stats[['Job Openings (2024)', 'Projected Openings (2030)',
                      'Openings_Pct_Change', 'Automation Risk (%)', 'Median Salary (USD)']])

# 分类行业
hot_industries = industry_stats[industry_stats['Openings_Pct_Change'] > 0]
cold_industries = industry_stats[industry_stats['Openings_Pct_Change'] < 0]

print(f"\n热点行业 (岗位增长): {len(hot_industries)} 个")
print(hot_industries.nlargest(5, 'Openings_Pct_Change')[['Openings_Pct_Change', 'Median Salary (USD)']])

print(f"\n冷点行业 (岗位减少): {len(cold_industries)} 个")
print(cold_industries.nsmallest(5, 'Openings_Pct_Change')[['Openings_Pct_Change', 'Median Salary (USD)']])

# ============= 核心洞察 2: 地区差异 =============
print("\n" + "=" * 80)
print("核心洞察 2: 地区战场——全球化的结构性重塑")
print("=" * 80)

location_stats = df.groupby('Location').agg({
    'Job Openings (2024)': 'sum',
    'Projected Openings (2030)': 'sum',
    'Openings_Abs_Change': 'sum',
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

location_stats['Openings_Pct_Change'] = (
    location_stats['Openings_Abs_Change'] /
    location_stats['Job Openings (2024)'] * 100
).round(2)

location_stats = location_stats.rename(columns={'Job Title': 'Job_Count'})
location_stats = location_stats.sort_values('Openings_Pct_Change', ascending=False)

print("\n地区排名 (按岗位增长率):")
print(location_stats[['Job Openings (2024)', 'Projected Openings (2030)',
                      'Openings_Pct_Change', 'Automation Risk (%)', 'Median Salary (USD)']])

# ============= 核心洞察 3: 行业×地区交叉分析 =============
print("\n" + "=" * 80)
print("核心洞察 3: 行业×地区——地区放大效应")
print("=" * 80)

# 创建交叉分析表
cross_stats = df.groupby(['Industry', 'Location']).agg({
    'Job Openings (2024)': 'sum',
    'Projected Openings (2030)': 'sum',
    'Openings_Abs_Change': 'sum',
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

cross_stats['Openings_Pct_Change'] = (
    cross_stats['Openings_Abs_Change'] /
    cross_stats['Job Openings (2024)'] * 100
).round(2)

cross_stats = cross_stats.rename(columns={'Job Title': 'Job_Count'})
cross_stats = cross_stats[cross_stats['Job_Count'] >= 50]  # 足够样本量

print(f"\n行业×地区组合数: {len(cross_stats)}")
print("\n最具增长潜力的组合 (Top 10):")
print(cross_stats.nlargest(10, 'Openings_Pct_Change')[['Openings_Pct_Change',
                                                         'Median Salary (USD)',
                                                         'Automation Risk (%)']])

print("\n岗位缩减最严重的组合 (Bottom 10):")
print(cross_stats.nsmallest(10, 'Openings_Pct_Change')[['Openings_Pct_Change',
                                                          'Median Salary (USD)',
                                                          'Automation Risk (%)']])

# ============= 可视化部分 =============
print("\n开始生成可视化...")
sns.set_style("whitegrid")
sns.set_palette("husl")

# 图1: 行业岗位变化全景图
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1.1 行业岗位净变化 (绝对值)
industry_sorted = industry_stats.sort_values('Openings_Abs_Change')
colors = ['red' if x < 0 else 'green' for x in industry_sorted['Openings_Abs_Change']]
industry_sorted['Openings_Abs_Change'].plot(kind='barh', ax=axes[0, 0], color=colors)
axes[0, 0].set_title('各行业岗位净变化 (2024-2030)\n红色=减少, 绿色=增加',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('岗位净变化数', fontproperties=font, fontsize=13)
axes[0, 0].set_ylabel('行业', fontproperties=font, fontsize=13)
axes[0, 0].axvline(0, color='black', linewidth=2)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[0, 0].get_xticklabels():
    label.set_fontproperties(font)

# 1.2 行业岗位增长率
industry_sorted_pct = industry_stats.sort_values('Openings_Pct_Change')
colors_pct = ['red' if x < 0 else 'green' for x in industry_sorted_pct['Openings_Pct_Change']]
industry_sorted_pct['Openings_Pct_Change'].plot(kind='barh', ax=axes[0, 1], color=colors_pct)
axes[0, 1].set_title('各行业岗位增长率 (%)\n红色=萎缩, 绿色=扩张',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('岗位增长率 (%)', fontproperties=font, fontsize=13)
axes[0, 1].set_ylabel('行业', fontproperties=font, fontsize=13)
axes[0, 1].axvline(0, color='black', linewidth=2)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[0, 1].get_xticklabels():
    label.set_fontproperties(font)

# 1.3 行业平均薪资
industry_sorted_salary = industry_stats.sort_values('Median Salary (USD)')
industry_sorted_salary['Median Salary (USD)'].plot(kind='barh', ax=axes[1, 0], color='skyblue')
axes[1, 0].set_title('各行业平均薪资水平',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
axes[1, 0].set_ylabel('行业', fontproperties=font, fontsize=13)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)

# 1.4 行业自动化风险
industry_sorted_risk = industry_stats.sort_values('Automation Risk (%)')
industry_sorted_risk['Automation Risk (%)'].plot(kind='barh', ax=axes[1, 1], color='coral')
axes[1, 1].set_title('各行业平均自动化风险',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=13)
axes[1, 1].set_ylabel('行业', fontproperties=font, fontsize=13)
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[1, 1].get_xticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B2_outputs/01_industry_overview.png', dpi=300, bbox_inches='tight')
print("✓ 01_industry_overview.png")
plt.close()

# 图2: 战场效能图 (岗位增长 vs 薪资)
fig, ax = plt.subplots(figsize=(16, 10))

# 行业散点
scatter = ax.scatter(industry_stats['Openings_Pct_Change'],
                     industry_stats['Median Salary (USD)'],
                     s=industry_stats['Job_Count']/10,
                     c=industry_stats['Automation Risk (%)'],
                     cmap='RdYlGn_r',
                     alpha=0.7,
                     edgecolors='black',
                     linewidth=1.5)

# 标注每个行业
for idx, row in industry_stats.iterrows():
    ax.annotate(idx,
                xy=(row['Openings_Pct_Change'], row['Median Salary (USD)']),
                xytext=(5, 5), textcoords='offset points',
                fontproperties=font, fontsize=10, alpha=0.8)

ax.set_xlabel('岗位增长率 (%)', fontproperties=font, fontsize=15)
ax.set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=15)
ax.set_title('战场效能图: 岗位增长 vs 薪资\n(气泡大小=岗位数, 颜色=自动化风险)',
             fontproperties=font, fontsize=17, fontweight='bold', pad=20)

# 添加象限线
ax.axhline(industry_stats['Median Salary (USD)'].median(),
           color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)

# 象限标注
q_font = FontProperties(family='Microsoft YaHei', size=13, weight='bold')
ax.text(industry_stats['Openings_Pct_Change'].max() * 0.7,
        industry_stats['Median Salary (USD)'].max() * 0.97,
        '黄金战场\n高增长+高薪',
        fontproperties=q_font, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='gold', alpha=0.7,
                  edgecolor='darkgoldenrod', linewidth=2))

ax.text(industry_stats['Openings_Pct_Change'].min() * 0.7,
        industry_stats['Median Salary (USD)'].max() * 0.97,
        '高薪萎缩区\n夕阳产业',
        fontproperties=q_font, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='orange', alpha=0.7,
                  edgecolor='darkorange', linewidth=2))

ax.text(industry_stats['Openings_Pct_Change'].max() * 0.7,
        industry_stats['Median Salary (USD)'].min() * 1.03,
        '低薪增长区\n机会型',
        fontproperties=q_font, ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7,
                  edgecolor='blue', linewidth=2))

ax.text(industry_stats['Openings_Pct_Change'].min() * 0.7,
        industry_stats['Median Salary (USD)'].min() * 1.03,
        '双重困境\n低薪+萎缩',
        fontproperties=q_font, ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', alpha=0.7,
                  edgecolor='darkred', linewidth=2))

ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)

# 添加colorbar
cbar = plt.colorbar(scatter, ax=ax, label='自动化风险 (%)')
cbar.set_label('自动化风险 (%)', fontproperties=font, fontsize=13)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B2_outputs/02_battlefield_efficiency_map.png', dpi=300, bbox_inches='tight')
print("✓ 02_battlefield_efficiency_map.png")
plt.close()

# 图3: 地区对比
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 3.1 地区岗位增长率
location_sorted = location_stats.sort_values('Openings_Pct_Change')
colors_loc = ['red' if x < 0 else 'green' for x in location_sorted['Openings_Pct_Change']]
location_sorted['Openings_Pct_Change'].plot(kind='barh', ax=axes[0], color=colors_loc)
axes[0].set_title('各地区岗位增长率\n哪些地区是AI时代的赢家?',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('岗位增长率 (%)', fontproperties=font, fontsize=13)
axes[0].set_ylabel('地区', fontproperties=font, fontsize=13)
axes[0].axvline(0, color='black', linewidth=2)
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)

# 3.2 地区平均薪资
location_sorted_salary = location_stats.sort_values('Median Salary (USD)')
location_sorted_salary['Median Salary (USD)'].plot(kind='barh', ax=axes[1], color='skyblue')
axes[1].set_title('各地区平均薪资\n地区决定了盔甲的价值上限',
                  fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
axes[1].set_ylabel('地区', fontproperties=font, fontsize=13)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(12)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B2_outputs/03_location_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 03_location_comparison.png")
plt.close()

# 图4: 行业×地区热力图
# 选择主要行业和地区
pivot_growth = df.groupby(['Industry', 'Location']).agg({
    'Openings_Pct_Change': 'mean'
}).reset_index()
pivot_growth_matrix = pivot_growth.pivot(index='Industry',
                                          columns='Location',
                                          values='Openings_Pct_Change')

pivot_salary = df.groupby(['Industry', 'Location']).agg({
    'Median Salary (USD)': 'mean'
}).reset_index()
pivot_salary_matrix = pivot_salary.pivot(index='Industry',
                                          columns='Location',
                                          values='Median Salary (USD)')

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# 热力图1: 增长率
sns.heatmap(pivot_growth_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
            center=0, ax=axes[0], cbar_kws={'label': '岗位增长率 (%)'},
            linewidths=0.5, linecolor='white', annot_kws={'size': 9})
axes[0].set_title('行业 × 地区 → 岗位增长率\n(绿=增长, 红=萎缩)',
                  fontproperties=font, fontsize=17, fontweight='bold', pad=20)
axes[0].set_xlabel('地区', fontproperties=font, fontsize=14)
axes[0].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
cbar1 = axes[0].collections[0].colorbar
cbar1.set_label('岗位增长率 (%)', fontproperties=font, fontsize=12)

# 热力图2: 薪资
sns.heatmap(pivot_salary_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '平均薪资 (USD)'},
            linewidths=0.5, linecolor='white', annot_kws={'size': 9})
axes[1].set_title('行业 × 地区 → 平均薪资\n(深蓝=高薪, 浅黄=低薪)',
                  fontproperties=font, fontsize=17, fontweight='bold', pad=20)
axes[1].set_xlabel('地区', fontproperties=font, fontsize=14)
axes[1].set_ylabel('行业', fontproperties=font, fontsize=14)
for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(11)
cbar2 = axes[1].collections[0].colorbar
cbar2.set_label('平均薪资 (USD)', fontproperties=font, fontsize=12)

plt.tight_layout()
plt.savefig('B2_outputs/04_industry_location_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 04_industry_location_heatmap.png")
plt.close()

# 保存统计数据
industry_stats.to_csv('B2_outputs/industry_stats.csv')
location_stats.to_csv('B2_outputs/location_stats.csv')
cross_stats.to_csv('B2_outputs/industry_location_cross.csv')

print("\n" + "=" * 80)
print("B2 分析完成!")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_industry_overview.png - 行业全景图 (2×2网格)")
print("  2. 02_battlefield_efficiency_map.png - 战场效能图 (增长vs薪资)")
print("  3. 03_location_comparison.png - 地区对比")
print("  4. 04_industry_location_heatmap.png - 行业×地区热力图")
print("\n统计数据:")
print("  - industry_stats.csv")
print("  - location_stats.csv")
print("  - industry_location_cross.csv")
