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


df = pd.read_csv('csv/dagw2.csv', index_col = [0])  # This is a bit of a hack. Next time when using .to_csv, consider setting the "index = False" flag when we don't need the index.

df['source_category'] = df['source'].apply(lambda x: mapping_dict[x])

df.to_csv('csv/dagw_new.csv')

####
# df_porn = df.drop(['tokens'], axis=1) 
# This was the problem. When doing .to_csv, it generates an index column. 
# This column was included in the sum – so adult_token_count = index + adult_tokens. 
# If index > tokens, adult_token_count > tokens.  
# To avoid, instead keep only the columns prefixed with porn_
####

df_porn = df.filter(regex='^porn_', axis = 1)

df_porn['adult_token_count'] = df_porn.sum(axis=1)
df_porn["tokens"] = df["tokens"]
df_porn['adult_token_proportion'] = df_porn['adult_token_count']/df_porn['tokens']
df_porn.replace([np.inf, -np.inf], np.nan, inplace=True)

df['adult_token_count'] = df_porn["adult_token_count"]
df['adult_token_proportion'] = df_porn['adult_token_proportion'].fillna(0)

df.to_pickle('pkl/dagw_statistics.pkl')

plt.figure(figsize = (18,10))
sns.set_style("ticks")
plot = sns.violinplot(x = 'source_mapped', y = 'adult_token_count',  scale = "width", data = df_porn)
plot.set_xlabel("Count", fontsize = 20)
plot.set_ylabel("Source", fontsize = 20)
plot.tick_params(axis='x', labelsize = 15, colors = 'grey')   
plot.tick_params(axis='y', labelsize = 15, colors = 'grey') 
plot.figure.savefig('figs/count_violin.pdf')


fig = sns.catplot(x = 'adult_token_count', y = 'source_mapped', kind = 'box', data = df_porn, height = 10, aspect = 9/7)
fig.savefig('figs/count_box.pdf')

plt.figure(figsize = (18,10))
sns.set_style("ticks")
plot = sns.violinplot(x = 'adult_token_proportion', y = 'source_mapped',  scale = "width", data = df_porn)
plot.set_xlabel("adult_token_proportion", fontsize = 20)
plot.set_ylabel("Source", fontsize = 20)
plot.tick_params(axis='x', labelsize = 15, colors = 'grey')   
plot.tick_params(axis='y', labelsize = 15, colors = 'grey') 
plot.figure.savefig('figs/adult_token_proportion_violin.pdf')

fig = sns.catplot(x="adult_token_proportion", y="source_mapped", kind="box", data=df_porn, height = 10, aspect=9/7)
fig.savefig('figs/adult_token_proportion_box.pdf')
