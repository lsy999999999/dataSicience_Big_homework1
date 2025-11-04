"""
B1 最终版分析 - 完全修复中文显示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============= 彻底修复中文字体 =============
# 方法1: 直接设置matplotlib的全局字体
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# 方法2: 确保所有text元素都使用中文字体
from matplotlib.font_manager import FontProperties
font = FontProperties(family='Microsoft YaHei', size=12)

print("=" * 80)
print("B1 分析: 定义'盔甲'——教育与经验的个体生存法则 (最终版)")
print("=" * 80)

# 创建输出目录
if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# 教育层级编码
education_order = {
    'High School': 1,
    'Associate Degree': 2,
    'Bachelor\'s Degree': 3,
    'Master\'s Degree': 4,
    'PhD': 5
}
df['EduLevel_Ordinal'] = df['Required Education'].map(education_order)

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

# 创建教育水平的中文映射
edu_chinese = {
    'High School': '高中',
    'Associate Degree': '专科',
    'Bachelor\'s Degree': '本科',
    'Master\'s Degree': '硕士',
    'PhD': '博士'
}

sns.set_style("whitegrid")
sns.set_palette("husl")

print("\n开始生成可视化...")

# ============= 图1: 2x2 基础分析 =============
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

edu_order_list = ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD']
edu_labels = ['高中', '专科', '本科', '硕士', '博士']

# 1.1 教育 vs 自动化风险
bp1 = sns.boxplot(data=df, x='Required Education', y='Automation Risk (%)',
                   order=edu_order_list, ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title('教育水平 vs 自动化风险\n学历越高,风险越低?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 0].set_ylabel('自动化风险 (%)', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 0].set_xticklabels(edu_labels, fontproperties=font, fontsize=12, rotation=0)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)

# 1.2 教育 vs 薪资
bp2 = sns.boxplot(data=df, x='Required Education', y='Median Salary (USD)',
                   order=edu_order_list, ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('教育水平 vs 薪资水平\n学历越高,收入越高?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 1].set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 1].set_xticklabels(edu_labels, fontproperties=font, fontsize=12, rotation=0)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)

# 1.3 经验 vs 自动化风险
bp3 = sns.boxplot(data=df, x='Experience_Group', y='Automation Risk (%)',
                   ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title('经验水平 vs 自动化风险\n经验越多,风险越低?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('经验水平', fontproperties=font, fontsize=14, labelpad=10)
axes[1, 0].set_ylabel('自动化风险 (%)', fontproperties=font, fontsize=14, labelpad=10)
for label in axes[1, 0].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[1, 0].get_yticklabels():
    label.set_fontproperties(font)

# 1.4 经验 vs 薪资
bp4 = sns.boxplot(data=df, x='Experience_Group', y='Median Salary (USD)',
                   ax=axes[1, 1], palette='Oranges')
axes[1, 1].set_title('经验水平 vs 薪资水平\n经验越多,收入越高?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('经验水平', fontproperties=font, fontsize=14, labelpad=10)
axes[1, 1].set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=14, labelpad=10)
for label in axes[1, 1].get_xticklabels():
    label.set_fontproperties(font)
for label in axes[1, 1].get_yticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B1_outputs/01_armor_basic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 01_armor_basic_analysis.png")
plt.close()

# ============= 图2: 热力图 =============
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_order_list)
pivot_risk.index = edu_labels

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_order_list)
pivot_salary.index = edu_labels

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 热力图1: 风险
hm1 = sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
                   ax=axes[0], cbar_kws={'label': '自动化风险 (%)'},
                   linewidths=1, linecolor='white', annot_kws={'size': 13})
axes[0].set_title('教育 × 经验 → 自动化风险\n(绿色=安全, 红色=危险)',
                  fontproperties=font, fontsize=17, fontweight='bold', pad=20)
axes[0].set_xlabel('经验水平', fontproperties=font, fontsize=15, labelpad=12)
axes[0].set_ylabel('教育水平', fontproperties=font, fontsize=15, labelpad=12)

# 设置X轴刻度标签
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
# 设置Y轴刻度标签
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
    label.set_rotation(0)

# 设置colorbar标签
cbar1 = axes[0].collections[0].colorbar
cbar1.set_label('自动化风险 (%)', fontproperties=font, fontsize=13)
for label in cbar1.ax.get_yticklabels():
    label.set_fontproperties(font)

# 热力图2: 薪资
hm2 = sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
                   ax=axes[1], cbar_kws={'label': '中位薪资 (USD)'},
                   linewidths=1, linecolor='white', annot_kws={'size': 13})
axes[1].set_title('教育 × 经验 → 薪资水平\n(深蓝=高薪, 浅黄=低薪)',
                  fontproperties=font, fontsize=17, fontweight='bold', pad=20)
axes[1].set_xlabel('经验水平', fontproperties=font, fontsize=15, labelpad=12)
axes[1].set_ylabel('教育水平', fontproperties=font, fontsize=15, labelpad=12)

for label in axes[1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in axes[1].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
    label.set_rotation(0)

cbar2 = axes[1].collections[0].colorbar
cbar2.set_label('中位薪资 (USD)', fontproperties=font, fontsize=13)
for label in cbar2.ax.get_yticklabels():
    label.set_fontproperties(font)

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 02_armor_combo_heatmap.png")
plt.close()

# ============= 图3: 散点图 - 经验vs薪资 =============
fig, ax = plt.subplots(figsize=(16, 10))

for i, edu in enumerate(edu_order_list):
    subset = df[df['Required Education'] == edu]
    ax.scatter(subset['Experience Required (Years)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=40, label=edu_labels[i])

ax.set_xlabel('所需经验 (年)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('经验 vs 薪资: 教育水平的调节效应\n(相同经验,不同教育的回报差异)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)

legend = ax.legend(title='教育水平', loc='upper left', fontsize=14,
                   title_fontsize=15, framealpha=0.95)
legend.get_title().set_fontproperties(font)
for text in legend.get_texts():
    text.set_fontproperties(font)

ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)

plt.tight_layout()
plt.savefig('B1_outputs/03_experience_salary_by_education.png', dpi=300, bbox_inches='tight')
print("✓ 03_experience_salary_by_education.png")
plt.close()

# ============= 图4: 效能散点图 =============
fig, ax = plt.subplots(figsize=(18, 11))

colors = ['#ff6b6b', '#f9ca24', '#6ab04c', '#4834d4', '#be2edd']
for i, edu in enumerate(edu_order_list):
    subset = df[df['Required Education'] == edu]
    ax.scatter(subset['Automation Risk (%)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=25, label=edu_labels[i], color=colors[i])

ax.set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('盔甲效能图: 自动化风险 vs 薪资\n(理想区: 左上角 = 低风险+高薪)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)

# 添加象限线
median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
ax.axhline(median_salary, color='gray', linestyle='--', alpha=0.6, linewidth=2)
ax.axvline(median_risk, color='gray', linestyle='--', alpha=0.6, linewidth=2)

# 象限标注
q_font = FontProperties(family='Microsoft YaHei', size=13, weight='bold')

ax.text(8, df['Median Salary (USD)'].quantile(0.96), '理想区\n低风险+高薪',
        fontproperties=q_font, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2))

ax.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.96),
        '高薪高风险区\n需谨慎',
        fontproperties=q_font, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7, edgecolor='orange', linewidth=2))

ax.text(8, df['Median Salary (USD)'].quantile(0.04), '安全低薪区\n生存型',
        fontproperties=q_font, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=2))

ax.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.04),
        '危险区\n高风险+低薪',
        fontproperties=q_font, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=2))

legend = ax.legend(title='教育水平', loc='center right', fontsize=14,
                   title_fontsize=15, framealpha=0.95)
legend.get_title().set_fontproperties(font)
for text in legend.get_texts():
    text.set_fontproperties(font)

ax.grid(True, alpha=0.3)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)

plt.tight_layout()
plt.savefig('B1_outputs/05_armor_efficiency_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 05_armor_efficiency_scatter.png")
plt.close()

print("\n" + "=" * 80)
print("B1 分析完成! 所有图表中文显示已完全修复")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_armor_basic_analysis.png - 教育与经验的基础效应 (2×2网格)")
print("  2. 02_armor_combo_heatmap.png - 教育×经验组合热力图")
print("  3. 03_experience_salary_by_education.png - 经验-薪资散点图")
print("  4. 05_armor_efficiency_scatter.png - 盔甲效能四象限图")
print("\n字体: Microsoft YaHei")
print("分辨率: 300 DPI")
print("格式: PNG")
