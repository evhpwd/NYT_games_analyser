import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('mode.copy_on_write', True)
pd.set_option('colheader_justify', 'left')

def create_df(text: str) -> pd.DataFrame:
    messages = re.findall(r'\[(.*?)\] (.*?):((?:.|\n)*?)(?=\[.*\]|$)', text)
    df = pd.DataFrame(messages, columns=['timestamp', 'sender', 'message'])
    df['message'] = df['message'].str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M:%S')
    return df

def get_game_df(df: pd.DataFrame, game_re: str) -> pd.DataFrame:
    return df[df['message'].str.contains(game_re)]


parser = argparse.ArgumentParser()
parser.add_argument('chatfile')
parser.add_argument('outfile', default='default_out.csv')
args = parser.parse_args()

with open(args.chatfile, 'r', encoding='utf8') as cf:
    chat = cf.read()

chatdf = create_df(chat)
senders = chatdf['sender'].unique()

game_res = {'wordle'      : r'Wordle [\d,]+ [\dX]/\d\*?',
            'connections' : r'Connections \nPuzzle #[\d,]+',
            'mini1'       : r'I solved the \d+/\d+/\d+ New York Times Mini Crossword in \d+:\d+!',
            'mini2'       : r'https://www.nytimes.com/badges/games/mini.html\?d=\d{4}-\d{2}-\d{2}&t=\d+'}

game_dfs: dict = {str: pd.DataFrame}

for game, regex in game_res.items():
    game_dfs[game] = get_game_df(chatdf, regex)

#print(game_dfs['wordle'])

# so we wanna just make dataframes w the scores in, average,freq, total, etc
# wordle scores = number or X
# also hard *
# also average guess 
#       add up values for each colour for each cell across all guesses and messages
#       account light/dark mode ig
# also longest streak

# so dataframe + stats
# maybe index = sender, columns score, guesses
# new df with sender, stats
#🟩⬛🟨

# graph plotting:
#fig, ax = plt.subplots()

#for key, grp in df.groupby(['sender']):
#    ax = grp.plot(ax=ax, kind='line', x='timestamp', y='score', label=key)

#plt.xlabel('Date')
#plt.ylabel('Score (7=failed)')
#plt.legend(loc='best')
#plt.show()

# results will have index sender, columns stats

def extract_wordle_scores(df: pd.DataFrame) -> pd.DataFrame:
    df['wordle']  = df['message'].str.extract(r'Wordle ([\d,]+) ')
    df['score']   = df['message'].str.extract(r'([\dX])/\d\*?')
    df['guesses'] = df['message'].str.extract(r'((?:[🟩⬜⬛🟨]+\n?)+)')
    return df

def get_average_guess(guesses: str, counts: list):
    for guess in guesses.strip().split('\n'):
        for i in range(0, 5):
            match guess[i]:
                case '⬜':
                    counts[i][0] += 1
                case '⬛':
                    counts[i][0] += 1
                case '🟨':
                    counts[i][1] += 1
                case '🟩':
                    counts[i][2] += 1

def analyse_wordle_scores(df: pd.DataFrame) -> pd.DataFrame:
    resultdf = pd.DataFrame(index=df['sender'].unique(), columns=['average', 'X/6', '6/6', '5/6', '4/6', '3/6', '2/6', '1/6', 'average_guess'])
    # remove CHEAT wordles >:(
    #print(df.loc[(df['sender'] == 'CIA') and (df['wordle'] == '693') and (df['score'] == '1')])

    for sender, grp in df.groupby(['sender']):
        sender = sender[0]
        print(sender)
        print(grp.loc[grp['score'] == '1']['message'])

        average = grp['score'].loc[grp['score'].isin(['1', '2', '3', '4', '5', '6'])].astype(int).mean()
        freqs   = grp['score'].value_counts()
        counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        grp['guesses'].apply(get_average_guess, counts=counts)
        average_guess = ''
        for c in counts:
            match c.index(max(c)):
                case 0:
                    average_guess += '⬛'
                case 1:
                    average_guess += '🟨'
                case 2:
                    average_guess += '🟩'

        resultdf.loc[sender, 'average'] = average
        resultdf.loc[sender, 'average_guess'] = average_guess
        for score, freq in freqs.items():
            resultdf.loc[sender, score + '/6'] = freq

        resultdf.fillna(0, inplace=True)
#        print(sender, average, freqs)

    return resultdf

wdf = extract_wordle_scores(game_dfs['wordle'])
#print(wdf)
rwdf = analyse_wordle_scores(wdf)
#print(analyse_wordle_scores(wdf))

with open(args.outfile, 'w') as of:
    of.write('overall:\n' + rwdf.to_csv(encoding='utf8') + '\n\nstats:\n' + wdf.to_csv(encoding='utf8'))