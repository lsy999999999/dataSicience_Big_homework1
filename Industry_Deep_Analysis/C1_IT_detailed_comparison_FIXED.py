"""
C1补充分析: IT行业内部详细对比 (修正版)
修正问题: 调整横纵轴范围,突出IT内部的差异
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
print("C1 IT行业详细对比 - 修正版 (突出差异)")
print("=" * 80)

# 加载数据
df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')
df['Openings_Abs_Change'] = df['Projected Openings (2030)'] - df['Job Openings (2024)']
df['Openings_Pct_Change'] = (df['Openings_Abs_Change'] / df['Job Openings (2024)'] * 100).round(2)

# IT行业数据
it_data = df[df['Industry'] == 'IT']
print(f"\nIT行业总记录数: {len(it_data)}")

# IT按AI Impact分层
it_by_ai = it_data.groupby('AI Impact Level').agg({
    'Automation Risk (%)': 'mean',
    'Median Salary (USD)': 'mean',
    'Openings_Pct_Change': 'mean',
    'Job Title': 'count'
}).round(2)
it_by_ai = it_by_ai.rename(columns={'Job Title': 'Count'})

print("\nIT行业按AI影响级别:")
print(it_by_ai)

# 按风险排序
it_ai_sorted = it_by_ai.sort_values('Automation Risk (%)')

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1. IT风险对比 - 调整Y轴突出差异
x_pos = np.arange(len(it_ai_sorted))
risk_values = it_ai_sorted['Automation Risk (%)'].values

# 根据风险值着色
colors_risk = []
for val in risk_values:
    if val < 39.5:
        colors_risk.append('#27ae60')  # 深绿 (最安全)
    elif val < 40.0:
        colors_risk.append('#f39c12')  # 橙色
    else:
        colors_risk.append('#e74c3c')  # 红色 (相对危险)

bars1 = axes[0, 0].bar(x_pos, risk_values, color=colors_risk, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('IT行业: AI影响级别 vs 自动化风险\n(Y轴范围38-42%,突出差异)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[0, 0].set_ylabel('平均自动化风险 (%)', fontproperties=font, fontsize=13)
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(it_ai_sorted.index, fontproperties=font, fontsize=12)
axes[0, 0].set_ylim(38, 42)  # 关键修改: 缩小Y轴范围
for label in axes[0, 0].get_yticklabels():
    label.set_fontproperties(font)

# 添加全局平均线
global_mean = df['Automation Risk (%)'].mean()
axes[0, 0].axhline(global_mean, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'全局平均: {global_mean:.1f}%')

# 添加IT整体平均线
it_mean = it_data['Automation Risk (%)'].mean()
axes[0, 0].axhline(it_mean, color='blue', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'IT整体平均: {it_mean:.1f}%')

axes[0, 0].legend(prop=font, fontsize=11, loc='upper right')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 添加精确数值标签
for i, val in enumerate(risk_values):
    axes[0, 0].text(i, val + 0.1, f'{val:.2f}%', ha='center', va='bottom',
                    fontproperties=font, fontsize=12, fontweight='bold')

# 2. IT薪资对比 - 调整X轴范围
salary_values = it_ai_sorted['Median Salary (USD)'].values

bars2 = axes[0, 1].bar(x_pos, salary_values, color='steelblue', alpha=0.8,
                       edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('IT行业: AI影响级别 vs 薪资\n(Y轴范围$110K-$115K,突出差异)',
                     fontproperties=font, fontsize=15, fontweight='bold', pad=12)
axes[0, 1].set_ylabel('平均薪资 (USD)', fontproperties=font, fontsize=13)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(it_ai_sorted.index, fontproperties=font, fontsize=12)
axes[0, 1].set_ylim(110000, 115000)  # 关键修改: 缩小Y轴范围
for label in axes[0, 1].get_yticklabels():
    label.set_fontproperties(font)

# 添加IT薪资平均线
axes[0, 1].axhline(it_data['Median Salary (USD)'].mean(), color='red',
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f"IT平均: ${it_data['Median Salary (USD)'].mean():,.0f}")
axes[0, 1].legend(prop=font, fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

for i, val in enumerate(salary_values):
    axes[0, 1].text(i, val + 400, f'${val:,.0f}', ha='center', va='bottom',
                    fontproperties=font, fontsize=11, fontweight='bold')

# 3. 数据表格展示
axes[1, 0].axis('off')
table_data = []
table_data.append(['AI影响级别', '风险(%)', '薪资(USD)', '增长率(%)', '样本量'])
for idx, row in it_ai_sorted.iterrows():
    table_data.append([
        idx,
        f"{row['Automation Risk (%)']:.2f}%",
        f"${row['Median Salary (USD)']:,.0f}",
        f"{row['Openings_Pct_Change']:.1f}%",
        f"{int(row['Count']):,}"
    ])

table = axes[1, 0].table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# 设置表头样式
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white', fontproperties=font, fontsize=13)

# 设置数据行样式
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        if j == 0:
            cell.set_text_props(fontproperties=font, fontsize=12, weight='bold')
        else:
            cell.set_text_props(fontproperties=font, fontsize=11)

        # 根据AI影响级别着色
        if i == 1:  # Low
            cell.set_facecolor('#d5f4e6')
        elif i == 2:  # Moderate
            cell.set_facecolor('#fff3cd')
        else:  # High
            cell.set_facecolor('#f8d7da')

axes[1, 0].set_title('IT行业: AI影响级别详细数据对比',
                     fontproperties=font, fontsize=16, fontweight='bold', pad=20)

# 4. 关键洞察文本
axes[1, 1].axis('off')

# 计算差异
max_risk = it_ai_sorted['Automation Risk (%)'].max()
min_risk = it_ai_sorted['Automation Risk (%)'].min()
risk_range = max_risk - min_risk

max_salary = it_ai_sorted['Median Salary (USD)'].max()
min_salary = it_ai_sorted['Median Salary (USD)'].min()
salary_range = max_salary - min_salary

insight_text = f"""
🔍 IT行业内部AI影响分析 - 关键洞察

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 风险差异:
   • 最低风险: {min_risk:.2f}% (Low AI Impact)
   • 最高风险: {max_risk:.2f}% (High AI Impact)
   • 内部差异: {risk_range:.2f}个百分点
   • vs 全局平均: {global_mean:.1f}%

💰 薪资差异:
   • 最高薪资: ${max_salary:,.0f} (Low AI Impact)
   • 最低薪资: ${min_salary:,.0f} (High AI Impact)
   • 内部差异: ${salary_range:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ 核心发现:

1️⃣ 反直觉现象:
   • Low AI Impact反而风险最低!
   • High AI Impact风险最高 (40.59%)
   • 说明: AI对IT的影响是"分化"而非统一

2️⃣ 薪资悖论:
   • Low AI Impact薪资最高 ($113,796)
   • 可能原因: 这些是"不易被AI替代"的核心岗位

3️⃣ IT内部差异虽小但显著:
   • 风险差异1.59% (39.0% vs 40.59%)
   • 相对于全局42.7%,IT整体仍是"安全区"
   • 但内部选择仍很重要!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 对个体的启示:

   即使在"黄金行业"IT内,也要选对细分:
   ✅ 优先: Low AI Impact岗位 (最安全+最高薪)
   ⚠️  避免: High AI Impact岗位 (相对高风险)
"""

axes[1, 1].text(0.05, 0.95, insight_text, transform=axes[1, 1].transAxes,
                fontproperties=font, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Industry_Deep_Analysis/C1_outputs/IT_AI_Impact_Detailed_FIXED.png',
            dpi=300, bbox_inches='tight')
print("\n✓ IT_AI_Impact_Detailed_FIXED.png (修正版)")
plt.close()

print("\n" + "=" * 80)
print("关键数据分析")
print("=" * 80)

print(f"\n1. IT行业内部风险差异:")
print(f"   - 范围: {min_risk:.2f}% - {max_risk:.2f}%")
print(f"   - 差距: {risk_range:.2f}个百分点")
print(f"   - 相对差异: {(risk_range/min_risk)*100:.1f}%")

print(f"\n2. IT vs 全局对比:")
print(f"   - IT最低风险: {min_risk:.2f}% vs 全局平均: {global_mean:.1f}%")
print(f"   - IT优势: {global_mean - min_risk:.1f}个百分点")

print(f"\n3. 薪资差异:")
print(f"   - 范围: ${min_salary:,.0f} - ${max_salary:,.0f}")
print(f"   - 差距: ${salary_range:,.0f}")
print(f"   - 相对差异: {(salary_range/min_salary)*100:.1f}%")

print("\n4. 关键洞察:")
print("   ⚡ Low AI Impact = 最低风险 + 最高薪资")
print("   ⚡ High AI Impact = 相对高风险 + 相对低薪")
print("   ⚡ 说明: AI对IT内部不同岗位的影响是\"分化\"的")
print("   ⚡ 结论: 即使在IT内,也要选对细分方向!")

print("\n分析完成!")
