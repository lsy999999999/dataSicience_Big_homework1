"""
B1 分析 (简化版): 定义"盔甲"——个体的生存法则
教育水平简化为3档: 基础教育 | 高等教育 | 高级学位
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
print("B1 分析: 定义'盔甲'——教育与经验的个体生存法则 (简化3档版)")
print("=" * 80)

# 创建输出目录
if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# ============= 教育水平简化为3档 =============
def simplify_education(edu):
    if edu in ['High School', 'Associate Degree']:
        return '基础教育'
    elif edu in ['Bachelor\'s Degree', 'Master\'s Degree']:
        return '高等教育'
    else:  # PhD
        return '高级学位'

df['Education_3Level'] = df['Required Education'].apply(simplify_education)

# 教育层级编码
education_order_3 = {
    '基础教育': 1,
    '高等教育': 2,
    '高级学位': 3
}
df['EduLevel_Ordinal'] = df['Education_3Level'].map(education_order_3)

print("\n教育水平分组:")
print(df['Education_3Level'].value_counts())

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

sns.set_style("whitegrid")
sns.set_palette("husl")

# ============= 核心统计 =============
print("\n" + "=" * 80)
print("核心洞察 1: 教育水平(3档)的保护效应")
print("=" * 80)

edu_3level_stats = df.groupby('Education_3Level').agg({
    'Automation Risk (%)': ['mean', 'median', 'std'],
    'Median Salary (USD)': ['mean', 'median', 'std'],
    'Job Title': 'count'
}).round(2)

edu_labels = ['基础教育', '高等教育', '高级学位']
edu_3level_stats = edu_3level_stats.reindex(edu_labels)

print("\n按教育水平的统计摘要:")
print(edu_3level_stats)

# 相关系数
corr_edu_risk = df['EduLevel_Ordinal'].corr(df['Automation Risk (%)'])
corr_edu_salary = df['EduLevel_Ordinal'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  教育水平 vs 自动化风险: {corr_edu_risk:.3f}")
print(f"  教育水平 vs 薪资水平: {corr_edu_salary:.3f}")

print("\n" + "=" * 80)
print("核心洞察 2: 经验积累的回报")
print("=" * 80)

exp_stats = df.groupby('Experience_Group').agg({
    'Automation Risk (%)': ['mean', 'median'],
    'Median Salary (USD)': ['mean', 'median'],
    'Job Title': 'count'
}).round(2)

print("\n按经验水平的统计摘要:")
print(exp_stats)

corr_exp_risk = df['Experience Required (Years)'].corr(df['Automation Risk (%)'])
corr_exp_salary = df['Experience Required (Years)'].corr(df['Median Salary (USD)'])

print(f"\n相关系数分析:")
print(f"  经验年限 vs 自动化风险: {corr_exp_risk:.3f}")
print(f"  经验年限 vs 薪资水平: {corr_exp_salary:.3f}")

print("\n" + "=" * 80)
print("核心洞察 3: 教育(3档)×经验的盔甲叠加")
print("=" * 80)

combo_stats = df.groupby(['Education_3Level', 'Experience_Group']).agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Job Title': 'count'
}).round(2)

print("\n教育×经验组合统计:")
for edu in edu_labels:
    print(f"\n{edu}:")
    if edu in combo_stats.index:
        print(combo_stats.loc[edu])

# ============= 可视化 =============
print("\n开始生成可视化...")

# 图1: 教育与经验的基础效应 (2x2网格)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1.1 教育(3档) vs 自动化风险
sns.boxplot(data=df, x='Education_3Level', y='Automation Risk (%)',
            order=edu_labels, ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title('教育水平 vs 自动化风险\n学历越高,风险越低?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 0].set_ylabel('自动化风险 (%)', fontproperties=font, fontsize=14, labelpad=10)
for label in axes[0, 0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)

# 1.2 教育(3档) vs 薪资
sns.boxplot(data=df, x='Education_3Level', y='Median Salary (USD)',
            order=edu_labels, ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('教育水平 vs 薪资水平\n学历越高,收入越高?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 1].set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=14, labelpad=10)
for label in axes[0, 1].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)

# 1.3 经验 vs 自动化风险
sns.boxplot(data=df, x='Experience_Group', y='Automation Risk (%)',
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
sns.boxplot(data=df, x='Experience_Group', y='Median Salary (USD)',
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
plt.savefig('B1_outputs/01_armor_basic_3level.png', dpi=300, bbox_inches='tight')
print("✓ 01_armor_basic_3level.png")
plt.close()

# 图2: 教育(3档)×经验热力图
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Education_3Level',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_labels)

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Education_3Level',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(edu_labels)

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 14})
axes[0].set_title('教育(3档) × 经验 → 自动化风险\n(绿色=安全, 红色=危险)',
                  fontproperties=font, fontsize=17, fontweight='bold', pad=20)
axes[0].set_xlabel('经验水平', fontproperties=font, fontsize=15, labelpad=12)
axes[0].set_ylabel('教育水平', fontproperties=font, fontsize=15, labelpad=12)
for label in axes[0].get_xticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
for label in axes[0].get_yticklabels():
    label.set_fontproperties(font)
    label.set_fontsize(13)
    label.set_rotation(0)
cbar1 = axes[0].collections[0].colorbar
cbar1.set_label('自动化风险 (%)', fontproperties=font, fontsize=13)

sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': '中位薪资 (USD)'},
            linewidths=1, linecolor='white', annot_kws={'size': 14})
axes[1].set_title('教育(3档) × 经验 → 薪资水平\n(深蓝=高薪, 浅黄=低薪)',
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

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_3level.png', dpi=300, bbox_inches='tight')
print("✓ 02_armor_combo_3level.png")
plt.close()

# 图3: 散点图 - 经验vs薪资(按3档教育着色)
fig, ax = plt.subplots(figsize=(16, 10))

colors_3level = ['#e74c3c', '#f39c12', '#27ae60']  # 红、橙、绿
for i, edu in enumerate(edu_labels):
    subset = df[df['Education_3Level'] == edu]
    ax.scatter(subset['Experience Required (Years)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=40, label=edu, color=colors_3level[i])

ax.set_xlabel('所需经验 (年)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('经验 vs 薪资: 教育水平(3档)的调节效应\n(相同经验,不同教育的回报差异)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)
legend = ax.legend(title='教育水平', loc='upper left', fontsize=15, title_fontsize=16)
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
plt.savefig('B1_outputs/03_experience_salary_3level.png', dpi=300, bbox_inches='tight')
print("✓ 03_experience_salary_3level.png")
plt.close()

# 图4: 效能散点图
fig, ax = plt.subplots(figsize=(18, 11))

for i, edu in enumerate(edu_labels):
    subset = df[df['Education_3Level'] == edu]
    ax.scatter(subset['Automation Risk (%)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=30, label=edu, color=colors_3level[i])

ax.set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('盔甲效能图: 自动化风险 vs 薪资\n(理想区: 左上角 = 低风险+高薪)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)

# 象限线
median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
ax.axhline(median_salary, color='gray', linestyle='--', alpha=0.6, linewidth=2)
ax.axvline(median_risk, color='gray', linestyle='--', alpha=0.6, linewidth=2)

# 象限标注
q_font = FontProperties(family='Microsoft YaHei', size=13, weight='bold')
ax.text(8, df['Median Salary (USD)'].quantile(0.96), '理想区\n低风险+高薪',
        fontproperties=q_font, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.7,
                  edgecolor='darkgreen', linewidth=2))
ax.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.96),
        '高薪高风险区\n需谨慎',
        fontproperties=q_font, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7,
                  edgecolor='orange', linewidth=2))
ax.text(8, df['Median Salary (USD)'].quantile(0.04), '安全低薪区\n生存型',
        fontproperties=q_font, ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7,
                  edgecolor='blue', linewidth=2))
ax.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.04),
        '危险区\n高风险+低薪',
        fontproperties=q_font, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', alpha=0.7,
                  edgecolor='darkred', linewidth=2))

legend = ax.legend(title='教育水平', loc='center right', fontsize=15, title_fontsize=16)
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
plt.savefig('B1_outputs/05_armor_efficiency_3level.png', dpi=300, bbox_inches='tight')
print("✓ 05_armor_efficiency_3level.png")
plt.close()

# 保存统计数据
edu_3level_stats.to_csv('B1_outputs/education_3level_stats.csv')
exp_stats.to_csv('B1_outputs/experience_stats.csv')
combo_stats.to_csv('B1_outputs/education_experience_combo_3level.csv')

print("\n" + "=" * 80)
print("B1 分析完成! (简化3档版)")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_armor_basic_3level.png - 教育(3档)与经验的基础效应")
print("  2. 02_armor_combo_3level.png - 教育(3档)×经验组合热力图")
print("  3. 03_experience_salary_3level.png - 经验-薪资散点图(3档)")
print("  4. 05_armor_efficiency_3level.png - 盔甲效能图(3档)")
print("\n教育分组:")
print("  • 基础教育 = 高中 + 专科")
print("  • 高等教育 = 本科 + 硕士")
print("  • 高级学位 = 博士")
