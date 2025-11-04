import pandas as pd

df = pd.read_csv('ai_job_trends_dataset_adjusted.csv')

print('教育水平分布:')
print(df['Required Education'].value_counts())

print('\n各教育水平的统计:')
for edu in sorted(df['Required Education'].unique()):
    subset = df[df['Required Education'] == edu]
    print(f'{edu}: {len(subset)} 条, 风险均值={subset["Automation Risk (%)"].mean():.2f}, 薪资均值=${subset["Median Salary (USD)"].mean():.0f}')
