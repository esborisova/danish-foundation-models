import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

mapping_dict = {
    "retsinformationdk": "Legal",
    "skat": "Legal",
    "retspraksis": "Legal",
    "hest": "Social Media",
    "cc": "Web",
    "adl": "Wiki & Books",
    "botxt": "Other",
    "danavis": "News",
    "dannet": "dannet",
    "depbank": "Other",
    "ep": "Conversation",
    "ft": "Conversation",
    "gutenberg": "Wiki & Books",
    "jvj": "Wiki & Books",
    "naat": "Conversation",
    "opensub": "Conversation",
    "relig": "Wiki & Books",
    "spont": "Conversation",
    "synne": "Other",
    "tv2r": "News",
    "wiki": "Wiki & Books",
    "wikibooks": "Wiki & Books",
    "wikisource": "Wiki & Books",
}


df = pd.read_csv('csv/dagw2.csv')

df['source_mapped'] = df['source'].apply(lambda x: mapping_dict[x])

df.to_csv('csv/dagw_new.csv')

tokens = df[['tokens', 'source']].copy()

df0 = df.drop(['tokens'], axis=1)

df0['sum'] = df0.sum(axis=1)

df0['tokens'] =  df['tokens']

df0['ratio'] = df0['sum']/df0['tokens']

df0.replace([np.inf, -np.inf], np.nan, inplace=True)
df0['ratio'] = df0['ratio'].fillna(0)

df0.to_pickle('pkl/dagw_statistics.pkl')

plt.figure(figsize = (18,10))
sns.set_style("ticks")
plot = sns.violinplot(x = 'source_mapped', y = 'sum',  scale = "width", data = df0)
plot.set_xlabel("Count", fontsize = 20)
plot.set_ylabel("Source", fontsize = 20)
plot.tick_params(axis='x', labelsize = 15, colors = 'grey')   
plot.tick_params(axis='y', labelsize = 15, colors = 'grey') 
plot.figure.savefig('figs/count_violin.pdf')


fig = sns.catplot(x = 'sum', y = 'source_mapped', kind = 'box', data = df0, height = 10, aspect = 9/7)
fig.savefig('figs/count_box.pdf')

plt.figure(figsize = (18,10))
sns.set_style("ticks")
plot = sns.violinplot(x = 'ratio', y = 'source_mapped',  scale = "width", data = df0)
plot.set_xlabel("Ratio", fontsize = 20)
plot.set_ylabel("Source", fontsize = 20)
plot.tick_params(axis='x', labelsize = 15, colors = 'grey')   
plot.tick_params(axis='y', labelsize = 15, colors = 'grey') 
plot.figure.savefig('figs/ratio_violin.pdf')

fig = sns.catplot(x="ratio", y="source_mapped", kind="box", data=df0, height = 10, aspect=9/7)
fig.savefig('figs/ratio_box.pdf')
