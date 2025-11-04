import pandas as pd
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

# 经验分组
df['Experience_Group'] = pd.cut(
    df['Experience Required (Years)'],
    bins=[0, 3, 7, 12, 20],
    labels=['初级(0-3年)', '中级(4-7年)', '高级(8-12年)', '专家(13-20年)']
)

print("检查pivot table:")
print("="*60)

pivot_risk = df.pivot_table(
    values='Automation Risk (%)',
    index='Required Education',
    columns='Experience_Group',
    aggfunc='mean'
)

print("\n原始pivot (未排序):")
print(pivot_risk)

edu_order_list = ['High School', 'Associate Degree', 'Bachelor\'s Degree',
                   'Master\'s Degree', 'PhD']

print(f"\n尝试用的顺序: {edu_order_list}")
print(f"实际的index: {list(pivot_risk.index)}")

pivot_risk_reindexed = pivot_risk.reindex(edu_order_list)

print("\nReindexed pivot:")
print(pivot_risk_reindexed)

print("\n直接分组查看:")
for edu in edu_order_list:
    count = len(df[df['Required Education'] == edu])
    print(f"{edu}: {count} 条记录")
