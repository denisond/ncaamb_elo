import plotly 
import plotly.graph_objs as go
import pandas as pd
import plotly.plotly as py

from plotly.offline import plot
# and use this “plot” to plot your figure.


plotly.tools.set_config_file(world_readable=True,
                             sharing='public')

df_temp = pd.read_csv('data/elo_rating.csv')
df = df_temp.set_index(pd.to_datetime(df_temp['date'])).drop('date',axis=1)

teams_samp = list(set(df.loc[df['ConfAbbrev'] == 'big_ten','TeamName']))

data = []
for team_name in teams_samp:


     data.append(go.Scatter(x=list(df.loc[df['TeamName'] == team_name].index),
                            y=list(df.loc[df['TeamName'] == team_name,'elo']),
                            name='{}'.format(team_name)))

buttons  = [dict(label = '{}'.format(team_name),method = 'update',
args = [{'visible': [True if i==k else False for i in range(len(teams_samp))]},
{'title': '{}'.format(team_name)}]) for k,team_name in enumerate(teams_samp)]

buttons.append(dict(label = 'Reset',method = 'update', args = [{'visible': [True for i in range(len(teams_samp))]},{'title': 'All Teams'}]))


updatemenus = list([dict(active=-1,
                     buttons=list(buttons),)])

layout = dict(title='NCAAMB Big Ten Elo Ratings', showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)


plot(fig, filename='elo_ratings.html')

