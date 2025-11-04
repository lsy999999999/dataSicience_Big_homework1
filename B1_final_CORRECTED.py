"""
B1 分析最终修正版: 修复引号问题,正确显示全部5个教育层级
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
print("B1 分析: 定义'盔甲'——教育与经验的个体生存法则")
print("=" * 80)

# 创建输出目录
if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
print(f"\n数据集: {len(df):,} 条记录")

# ============= 获取实际的教育水平值(避免引号问题) =============
actual_edu_values = sorted(df['Required Education'].unique())
print("\n实际的教育水平值:")
for edu in actual_edu_values:
    print(f"  - {edu}: {len(df[df['Required Education'] == edu])} 条")

# 教育层级编码和排序
education_order_map = {}
edu_order_list = []
edu_labels = []

for edu in actual_edu_values:
    if 'High School' in edu:
        education_order_map[edu] = 1
        edu_order_list.insert(0, edu)
        edu_labels.insert(0, '高中')
    elif 'Associate' in edu:
        education_order_map[edu] = 2
        if len(edu_order_list) < 2:
            edu_order_list.append(edu)
            edu_labels.append('专科')
        else:
            edu_order_list.insert(1, edu)
            edu_labels.insert(1, '专科')
    elif 'Bachelor' in edu:
        education_order_map[edu] = 3
        edu_order_list.append(edu) if len(edu_order_list) < 3 else edu_order_list.insert(2, edu)
        edu_labels.append('本科') if len(edu_labels) < 3 else edu_labels.insert(2, '本科')
    elif 'Master' in edu:
        education_order_map[edu] = 4
        edu_order_list.append(edu)
        edu_labels.append('硕士')
    elif 'PhD' in edu:
        education_order_map[edu] = 5
        edu_order_list.append(edu)
        edu_labels.append('博士')

# 最终排序
sorted_pairs = sorted(zip(edu_order_list, edu_labels), key=lambda x: education_order_map[x[0]])
edu_order_list = [x[0] for x in sorted_pairs]
edu_labels = [x[1] for x in sorted_pairs]

print(f"\n排序后的教育顺序:")
for i, (edu, label) in enumerate(zip(edu_order_list, edu_labels)):
    print(f"  {i+1}. {edu} → {label}")

df['EduLevel_Ordinal'] = df['Required Education'].map(education_order_map)

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

sns.set_style("whitegrid")
sns.set_palette("husl")

print("\n开始生成可视化...")

# ============= 图1: 2x2 基础分析 =============
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1.1 教育 vs 自动化风险
sns.boxplot(data=df, x='Required Education', y='Automation Risk (%)',
            order=edu_order_list, ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title('教育水平 vs 自动化风险\n学历越高,风险越低?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 0].set_ylabel('自动化风险 (%)', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 0].set_xticklabels(edu_labels, fontproperties=font, fontsize=13, rotation=0)
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)

# 1.2 教育 vs 薪资
sns.boxplot(data=df, x='Required Education', y='Median Salary (USD)',
            order=edu_order_list, ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title('教育水平 vs 薪资水平\n学历越高,收入越高?',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('教育水平', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 1].set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=14, labelpad=10)
axes[0, 1].set_xticklabels(edu_labels, fontproperties=font, fontsize=13, rotation=0)
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
plt.savefig('B1_outputs/01_armor_basic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 01_armor_basic_analysis.png")
plt.close()

# ============= 图2: 热力图 =============
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
)

# 不使用reindex,而是直接用实际顺序,然后替换index为中文
pivot_risk = pivot_risk.loc[edu_order_list]
pivot_risk.index = edu_labels

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
)
pivot_salary = pivot_salary.loc[edu_order_list]
pivot_salary.index = edu_labels

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': '自动化风险 (%)'},
            linewidths=1, linecolor='white', annot_kws={'size': 14})
axes[0].set_title('教育 × 经验 → 自动化风险\n(绿色=安全, 红色=危险)',
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

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 02_armor_combo_heatmap.png")
plt.close()

# ============= 图3: 散点图 =============
fig, ax = plt.subplots(figsize=(16, 10))

colors_5 = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#27ae60']  # 红橙蓝紫绿
for i, (edu, label) in enumerate(zip(edu_order_list, edu_labels)):
    subset = df[df['Required Education'] == edu]
    ax.scatter(subset['Experience Required (Years)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=40, label=label, color=colors_5[i])

ax.set_xlabel('所需经验 (年)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('经验 vs 薪资: 教育水平的调节效应\n(相同经验,不同教育的回报差异)',
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
plt.savefig('B1_outputs/03_experience_salary_by_education.png', dpi=300, bbox_inches='tight')
print("✓ 03_experience_salary_by_education.png")
plt.close()

# ============= 图4: 效能散点图 =============
fig, ax = plt.subplots(figsize=(18, 11))

for i, (edu, label) in enumerate(zip(edu_order_list, edu_labels)):
    subset = df[df['Required Education'] == edu]
    ax.scatter(subset['Automation Risk (%)'],
               subset['Median Salary (USD)'],
               alpha=0.5, s=30, label=label, color=colors_5[i])

ax.set_xlabel('自动化风险 (%)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_ylabel('中位薪资 (USD)', fontproperties=font, fontsize=16, labelpad=12)
ax.set_title('盔甲效能图: 自动化风险 vs 薪资\n(理想区: 左上角 = 低风险+高薪)',
             fontproperties=font, fontsize=18, fontweight='bold', pad=20)

median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
ax.axhline(median_salary, color='gray', linestyle='--', alpha=0.6, linewidth=2)
ax.axvline(median_risk, color='gray', linestyle='--', alpha=0.6, linewidth=2)

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
plt.savefig('B1_outputs/05_armor_efficiency_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 05_armor_efficiency_scatter.png")
plt.close()

print("\n" + "=" * 80)
print("B1 分析完成! (全部5个教育层级正确显示)")
print("=" * 80)
print("\n生成的图表:")
print("  1. 01_armor_basic_analysis.png - 教育(5档)与经验的基础效应")
print("  2. 02_armor_combo_heatmap.png - 教育(5档)×经验组合热力图")
print("  3. 03_experience_salary_by_education.png - 经验-薪资散点图")
print("  4. 05_armor_efficiency_scatter.png - 盔甲效能四象限图")
print("\n教育层级: 高中 | 专科 | 本科 | 硕士 | 博士")
