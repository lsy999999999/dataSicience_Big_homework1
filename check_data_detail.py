import pandas as pd
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print("教育水平详细统计:")
print("="*60)

for edu in sorted(df['Required Education'].unique()):
    subset = df[df['Required Education'] == edu]
    risk_mean = subset['Automation Risk (%)'].mean()
    salary_mean = subset['Median Salary (USD)'].mean()
    count = len(subset)

    print(f"\n{edu}:")
    print(f"  样本量: {count}")
    print(f"  风险均值: {risk_mean:.2f}%")
    print(f"  薪资均值: ${salary_mean:.2f}")
    print(f"  风险缺失: {subset['Automation Risk (%)'].isna().sum()}")
    print(f"  薪资缺失: {subset['Median Salary (USD)'].isna().sum()}")
