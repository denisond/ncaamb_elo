# NCAA men's basketball Elo rating app
Python scripts generating NCAA men's basketball [Elo ratings](https://en.wikipedia.org/wiki/Elo_rating_system) and [displaying them](https://denisond.github.io/elo_ratings.html)

## Contents
### `elo_creation.py`
Python script with following structure:
- read in match level kaggle NCAAMB dataset (1985-2018)
- create function to calculate and append Elo rating scores onto dataset
- transform dataset from one row per match into one row per team per date
- create datetime index
- merge on conference and team name data
- write to csv our Elo data indexed by team and datetime 

### `app.py`
Python script with following structure:
- read-in csv output by `elo_creation.py`
- limit data to schools in Big Ten confereence
- create a graph and buttons for each team
- create a menu to hold buttons
- set layout and create app as 'elo_ratings.html'

### `elo_ratings.html`
html file that is rendered [here](https://denisond.github.io/elo_ratings.html)
