"""
修复中文字体显示问题并重新运行B1分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============= 修复中文字体 =============
print("检查可用的中文字体...")

import matplotlib.font_manager as fm

# 列出所有可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 常见中文字体列表
chinese_fonts = [
    'Microsoft YaHei',
    'SimHei',
    'SimSun',
    'KaiTi',
    'FangSong',
    'STSong',
    'STKaiti',
    'STHeiti',
    'Arial Unicode MS'
]

# 找到第一个���用的中文字体
selected_font = None
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        print(f"✓ 找到中文字体: {font}")
        break

if selected_font is None:
    print("⚠ 未找到预设中文字体,尝试使用默认字体...")
    # 尝试找任何包含"微软"或"宋体"的字体
    for font in available_fonts:
        if '微软' in font or '宋体' in font or 'Microsoft' in font or 'SimHei' in font:
            selected_font = font
            print(f"✓ 找到备用字体: {font}")
            break

if selected_font is None:
    print("❌ 无法找到中文字体,将使用英文标签")
    use_chinese = False
    selected_font = 'DejaVu Sans'
else:
    use_chinese = True

# 设置字体
plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10

# 测试中文显示
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, '中文测试 Chinese Test',
        fontsize=20, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

if not os.path.exists('B1_outputs'):
    os.makedirs('B1_outputs')

plt.savefig('B1_outputs/00_font_test.png', dpi=150, bbox_inches='tight')
print(f"✓ 字体测试图已保存: B1_outputs/00_font_test.png")
print(f"  使用字体: {selected_font}")
plt.close()

# ============= 重新运行B1分析 =============
sns.set_style("whitegrid")
sns.set_palette("husl")

print("\n" + "=" * 80)
print("开始B1分析 (使用修复后的中文字体)")
print("=" * 80)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print(f"\n数据集基本信息:")
print(f"- 总记录数: {len(df):,}")
print(f"- 列数: {len(df.columns)}")

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
    labels=['Junior(0-3y)', 'Mid(4-7y)', 'Senior(8-12y)', 'Expert(13-20y)'] if not use_chinese
           else ['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

# 中英文标签映射
if use_chinese:
    labels = {
        'edu_order': ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD'],
        'edu_labels': ['高中', '专科', '本科', '硕士', '博士'],
        'titles': {
            'edu_risk': '教育水平 vs 自动化风险\n学历越高,风险越低?',
            'edu_salary': '教育水平 vs 薪资\n学历越高,收入越高?',
            'exp_risk': '经验水平 vs 自动化风险\n经验越多,风险越低?',
            'exp_salary': '经验水平 vs 薪资\n经验越多,收入越高?',
            'heatmap_risk': '教育 × 经验 → 自动化风险\n(绿色=安全,红色=危险)',
            'heatmap_salary': '教育 × 经验 → 薪资水平\n(深蓝=高薪,浅黄=低薪)',
            'scatter_title': '经验 vs 薪资: 教育水平的调节效应\n(相同经验,不同教育的回报差异)',
            'efficiency_title': '盔甲效能图: 自动化风险 vs 薪资\n(理想区: 左上角 = 低风险+高薪)'
        },
        'axes': {
            'edu': '教育水平',
            'exp': '经验水平',
            'risk': '自动化风险 (%)',
            'salary': '中位薪资 (USD)',
            'years': '所需经验 (年)'
        },
        'quadrants': {
            'ideal': '理想区\n低风险+高薪',
            'high_both': '高薪高风险区',
            'safe_low': '安全低薪区',
            'danger': '危险区\n高风险+低薪'
        }
    }
else:
    labels = {
        'edu_order': ['High School', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD'],
        'edu_labels': ['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
        'titles': {
            'edu_risk': 'Education Level vs Automation Risk\nDoes higher education reduce risk?',
            'edu_salary': 'Education Level vs Salary\nDoes higher education increase income?',
            'exp_risk': 'Experience Level vs Automation Risk\nDoes more experience reduce risk?',
            'exp_salary': 'Experience Level vs Salary\nDoes more experience increase income?',
            'heatmap_risk': 'Education × Experience → Automation Risk\n(Green=Safe, Red=Risky)',
            'heatmap_salary': 'Education × Experience → Salary\n(Dark Blue=High Pay, Yellow=Low Pay)',
            'scatter_title': 'Experience vs Salary: Moderation by Education\n(Same experience, different education returns)',
            'efficiency_title': 'Armor Efficiency Map: Risk vs Salary\n(Ideal: Top-Left = Low Risk + High Pay)'
        },
        'axes': {
            'edu': 'Education Level',
            'exp': 'Experience Level',
            'risk': 'Automation Risk (%)',
            'salary': 'Median Salary (USD)',
            'years': 'Required Experience (Years)'
        },
        'quadrants': {
            'ideal': 'Ideal Zone\nLow Risk + High Pay',
            'high_both': 'High Pay\nHigh Risk',
            'safe_low': 'Safe but\nLow Pay',
            'danger': 'Danger Zone\nHigh Risk + Low Pay'
        }
    }

print(f"\n使用语言: {'中文' if use_chinese else 'English'}")

# ============= 可视化部分 =============
print("\n开始生成可视化...")

# 图1: 教育与经验的基础效应 (2x2网格)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 教育 vs 自动化风险
sns.boxplot(data=df, x='Required Education', y='Automation Risk (%)',
            order=labels['edu_order'], ax=axes[0, 0], palette='Reds_r')
axes[0, 0].set_title(labels['titles']['edu_risk'], fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel(labels['axes']['edu'], fontsize=12)
axes[0, 0].set_ylabel(labels['axes']['risk'], fontsize=12)
axes[0, 0].set_xticklabels(labels['edu_labels'], rotation=45)

# 1.2 教育 vs 薪资
sns.boxplot(data=df, x='Required Education', y='Median Salary (USD)',
            order=labels['edu_order'], ax=axes[0, 1], palette='Greens')
axes[0, 1].set_title(labels['titles']['edu_salary'], fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel(labels['axes']['edu'], fontsize=12)
axes[0, 1].set_ylabel(labels['axes']['salary'], fontsize=12)
axes[0, 1].set_xticklabels(labels['edu_labels'], rotation=45)

# 1.3 经验 vs 自动化风险
sns.boxplot(data=df, x='Experience_Group', y='Automation Risk (%)',
            ax=axes[1, 0], palette='Blues_r')
axes[1, 0].set_title(labels['titles']['exp_risk'], fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel(labels['axes']['exp'], fontsize=12)
axes[1, 0].set_ylabel(labels['axes']['risk'], fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# 1.4 经验 vs 薪资
sns.boxplot(data=df, x='Experience_Group', y='Median Salary (USD)',
            ax=axes[1, 1], palette='Oranges')
axes[1, 1].set_title(labels['titles']['exp_salary'], fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel(labels['axes']['exp'], fontsize=12)
axes[1, 1].set_ylabel(labels['axes']['salary'], fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('B1_outputs/01_armor_basic_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/01_armor_basic_analysis.png")
plt.close()

# 图2: 教育×经验热力图
pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(labels['edu_order'])

pivot_salary = df.pivot_table(
    values='Median Salary (USD)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
).reindex(labels['edu_order'])

# 重命名行索引为中文
if use_chinese:
    pivot_risk.index = labels['edu_labels']
    pivot_salary.index = labels['edu_labels']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(pivot_risk, annot=True, fmt='.1f', cmap='RdYlGn_r',
            ax=axes[0], cbar_kws={'label': labels['axes']['risk']}, linewidths=0.5)
axes[0].set_title(labels['titles']['heatmap_risk'], fontsize=14, fontweight='bold')
axes[0].set_xlabel(labels['axes']['exp'], fontsize=12)
axes[0].set_ylabel(labels['axes']['edu'], fontsize=12)

sns.heatmap(pivot_salary, annot=True, fmt='.0f', cmap='YlGnBu',
            ax=axes[1], cbar_kws={'label': labels['axes']['salary']}, linewidths=0.5)
axes[1].set_title(labels['titles']['heatmap_salary'], fontsize=14, fontweight='bold')
axes[1].set_xlabel(labels['axes']['exp'], fontsize=12)
axes[1].set_ylabel(labels['axes']['edu'], fontsize=12)

plt.tight_layout()
plt.savefig('B1_outputs/02_armor_combo_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/02_armor_combo_heatmap.png")
plt.close()

# 图3: 散点图 - 经验 vs 薪资 (按教育着色)
plt.figure(figsize=(14, 9))

for i, edu in enumerate(labels['edu_order']):
    subset = df[df['Required Education'] == edu]
    label = labels['edu_labels'][i] if use_chinese else edu
    plt.scatter(subset['Experience Required (Years)'],
                subset['Median Salary (USD)'],
                alpha=0.5, s=30, label=label)

plt.xlabel(labels['axes']['years'], fontsize=13)
plt.ylabel(labels['axes']['salary'], fontsize=13)
plt.title(labels['titles']['scatter_title'], fontsize=15, fontweight='bold')
plt.legend(title=labels['axes']['edu'], loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B1_outputs/03_experience_salary_by_education.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/03_experience_salary_by_education.png")
plt.close()

# 图4: 盔甲效能散点图 (风险 vs 薪资)
plt.figure(figsize=(16, 10))

for i, edu in enumerate(labels['edu_order']):
    subset = df[df['Required Education'] == edu]
    label = labels['edu_labels'][i] if use_chinese else edu
    plt.scatter(subset['Automation Risk (%)'],
                subset['Median Salary (USD)'],
                alpha=0.4, s=20, label=label)

plt.xlabel(labels['axes']['risk'], fontsize=13)
plt.ylabel(labels['axes']['salary'], fontsize=13)
plt.title(labels['titles']['efficiency_title'], fontsize=15, fontweight='bold')

# 添加象限线
median_risk = df['Automation Risk (%)'].median()
median_salary = df['Median Salary (USD)'].median()
plt.axhline(median_salary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.axvline(median_risk, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 象限标注
plt.text(5, df['Median Salary (USD)'].quantile(0.95), labels['quadrants']['ideal'],
         fontsize=11, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
plt.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.95),
         labels['quadrants']['high_both'],
         fontsize=11, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
plt.text(5, df['Median Salary (USD)'].quantile(0.05), labels['quadrants']['safe_low'],
         fontsize=11, ha='left', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.text(df['Automation Risk (%)'].quantile(0.95), df['Median Salary (USD)'].quantile(0.05),
         labels['quadrants']['danger'],
         fontsize=11, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.legend(title=labels['axes']['edu'], loc='center right', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('B1_outputs/05_armor_efficiency_scatter.png', dpi=300, bbox_inches='tight')
print("✓ 保存: B1_outputs/05_armor_efficiency_scatter.png")
plt.close()

print("\n" + "=" * 80)
print("B1 分析完成! (中文字体已修复)")
print("=" * 80)
print("\n生成的图表:")
print("  0. 00_font_test.png - 字体测试")
print("  1. 01_armor_basic_analysis.png - 教育与经验的基础效应")
print("  2. 02_armor_combo_heatmap.png - 教育×经验组合热力图")
print("  3. 03_experience_salary_by_education.png - 经验-薪资散点图")
print("  4. 05_armor_efficiency_scatter.png - 盔甲效能散点图")
print(f"\n所有图表使用字体: {selected_font}")
print("\n请检查 00_font_test.png 确认中文显示正常!")
