import argparse
import re
import pandas as pd

pd.set_option('mode.copy_on_write', True)
pd.set_option('colheader_justify', 'left')

# ----------- Chat Parsing -----------------------------------------------------

# set up dataframe from exported whatsapp chat
def create_chat_df(text: str) -> pd.DataFrame:
    messages = re.findall(r'\[(.*?)\] (.*?):((?:.|\n)*?)(?=\[.*\]|$)', text)
    df = pd.DataFrame(messages, columns=['timestamp', 'sender', 'message'])
    df['message'] = df['message'].str.strip()
    df['message'] = df['message'].str.replace('\n', '|')
    df['message'] = df['message'].str.replace('\u200e', '')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M:%S')
    return df

# create dataframe where message matches regex
def get_game_df(df: pd.DataFrame, game_re: str) -> pd.DataFrame:
    return df[df['message'].str.contains(game_re, regex=True)]

# ----------- Game Functions ---------------------------------------------------

# generate results for a given type of game and dataframe
def get_results(game: str, df: pd.DataFrame) -> tuple:
    rdf = pd.DataFrame(index=df['sender'].unique())
    rdf.rename_axis('User', inplace=True)
    fdf = pd.DataFrame()
    match game:
        case 'Wordle':
            return get_wordle_results(build_wordle_df(df), rdf)
        case 'Connections':
            return get_connections_results(build_connections_df(df), rdf)
        case 'mini1':
            pass
            #return get_mini_results(build_mini1_df(df), rdf)
        case 'mini2':
            pass
            #return get_mini_results(build_mini2_df(df), rdf)

    return rdf, fdf

# calculate longest streak of consecutive numbers in column
def get_longest_streak(df: pd.DataFrame) -> int:
    # column containing true/false if next value is not consecutive
    df['is_consecutive'] = (df['game'].shift() + 1).ne(df['game'])
    # column with cumulative sum of true values
    df['streak'] = df['is_consecutive'].cumsum()
    # groups by streak and counts length
    df['streak_length'] = df.groupby('streak').cumcount() + 1

    return df['streak_length'].max()

# ----------- Wordle Functions --------------------------------------------------

# extract wordle data from messages into columns
def build_wordle_df(df: pd.DataFrame) -> pd.DataFrame:
    df['game']    = df['message'].str.extract(r'Wordle ([\d,]+) ')
    df['game']    = df['game'].str.replace(',', '').astype(int)
    df['score']   = df['message'].str.extract(r'([\dX])/\d\*?')
    df['guesses'] = df['message'].str.extract(r'((?:[🟩⬜⬛🟨]+\|?)+)')
    return df

# get frequencies of type of square for each guess
def calc_wordle_guess_freq(guesses: str, counts: list):
    for guess in guesses.strip('|').split('|'):
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

# calculate stats for a given dataframe
def calc_wordle_stats(df: pd.DataFrame) -> tuple:
        stats: dict = {'Games Played'   : len(df),
                       'Longest Streak' : str(get_longest_streak(df)) + ' days',
                       'Average Score'  : df['score'].loc[df['score'].isin(['1', '2', '3', '4', '5', '6'])].astype(int).mean()}
        score_freqs = df['score'].value_counts()
        
        # get average guess from frequencies of square colours
        counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        df['guesses'].apply(calc_wordle_guess_freq, counts=counts)
        avg_guess = ''
        for count in counts:
            match count.index(max(count)):
                case 0:
                    avg_guess += '⬛'
                case 1:
                    avg_guess += '🟨'
                case 2:
                    avg_guess += '🟩'
        
        return (stats, score_freqs, avg_guess)

# build dataframe containing stats per person
def get_wordle_results(df: pd.DataFrame, rdf: pd.DataFrame) -> tuple:
    # remove CHEAT wordles by CHEATERS >:(
    df = df.loc[~((df['sender'] == 'Dad') & (df['score'] == '1'))]

    # group wordle dataframe by sender and calculate stats for each
    for sender, sender_df in df.groupby(['sender']):
        sender = sender[0]
        stats, score_freqs, avg_guess = calc_wordle_stats(sender_df)

        for stat, value in stats.items():
            rdf.loc[sender, stat] = value
        
        for score in ['1', '2', '3', '4', '5', '6', 'X']:
            freq = 0
            if score in score_freqs.index:
                freq = score_freqs[score]
            rdf.loc[sender, score + '/6'] = freq

        rdf['Average Guess'] = avg_guess
    
    if rdf.empty:
        return rdf, pd.DataFrame()
    
    for col in ['Games Played', '1/6', '2/6', '3/6', '4/6', '5/6', '6/6', 'X/6']:
        rdf[col] = rdf[col].astype(int)
        
    # retrieve all failed wordles and associated message
    fdf = df.loc[df['score'] == 'X']
    fdf['message'] = fdf['message'].str.replace(r'(?:' + game_res['Wordle'] + r')|[\|⬜⬛🟨🟩]', '', regex=True)
    fdf = fdf.replace('', pd.NA).dropna()[['timestamp', 'sender', 'message', 'guesses']]

    return (rdf, fdf)

# ----------- Connections Functions ---------------------------------------------

# extract connections data into columns
def build_connections_df(df: pd.DataFrame) -> pd.DataFrame:
    df['game']    = df['message'].str.extract(r'Puzzle #([\d,]+)')
    df['game']    = df['game'].str.replace(',', '').astype(int)
    df['guesses'] = df['message'].str.extract(r'((?:[🟨🟩🟦🟪]+\|?)+)')
    return df

# calculate frequency of succesful guesses and mistakes as well as wins
def calc_connections_guess_freq(guesses: str, counts: dict):
    solved = 0
    for guess in guesses.strip('|').split('|'):
        match guess:
            case '🟨🟨🟨🟨':
                counts['Yellows'] += 1
                solved += 1
            case '🟩🟩🟩🟩':
                counts['Greens']  += 1
                solved += 1
            case '🟦🟦🟦🟦':
                counts['Blues']   += 1
                solved += 1
            case '🟪🟪🟪🟪':
                counts['Purples'] += 1
                solved += 1
            case _:
                counts['mistakes'] += 1
    if solved == 4:
        counts['Won'] += 1
    else:
        counts['Lost'] += 1

# calculate connections stats for a dataframe
def calc_connections_stats(df: pd.DataFrame):
        stats: dict = {'Games Played'   : len(df),
                       'Longest Streak' : str(get_longest_streak(df)) + ' days',
                       'Won'  : 0,
                       'Lost' : 0,
                       'Yellows' : 0,
                       'Greens'  : 0,
                       'Blues'   : 0,
                       'Purples' : 0,
                       'mistakes' : 0}
        
        df['guesses'].apply(calc_connections_guess_freq, counts=stats)
        stats['Average Mistakes'] = stats['mistakes'] / stats['Games Played']
        stats.pop('mistakes')
        return stats

# build dataframe containing stats per person
def get_connections_results(df: pd.DataFrame, rdf: pd.DataFrame) -> tuple:
    # group wordle dataframe by sender and calculate stats for each
    for sender, sender_df in df.groupby(['sender']):
        sender = sender[0]
        stats = calc_connections_stats(sender_df)

        for stat, value in stats.items():
            rdf.loc[sender, stat] = value

    if rdf.empty:
        return rdf, pd.DataFrame()
    
    for col in ['Games Played', 'Won', 'Lost', 'Yellows', 'Greens', 'Blues', 'Purples']:
        rdf[col] = rdf[col].astype(int)

    df['won'] = df['guesses'].apply(lambda x: all(s in x for s in ['🟨🟨🟨🟨','🟩🟩🟩🟩','🟦🟦🟦🟦','🟪🟪🟪🟪']))
    fdf = df.loc[~df['won']]
    fdf['message'] = fdf['message'].str.replace(r'(?:' + game_res['Connections'] + r')|[\|🟨🟩🟦🟪]', '', regex=True)
    fdf = fdf.replace('', pd.NA).dropna()[['timestamp', 'sender', 'message', 'guesses']]

    return rdf, fdf

    
# ----------- Mini Functions ----------------------------------------------------

# TBD

# ----------- Entry Point -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--chat-file', dest='chat_file',  required=True            )
parser.add_argument('--out-file',  dest='out_file',   default='default_out.csv')
args = parser.parse_args()

with open(args.chat_file, 'r', encoding='utf8') as chatf:
    chat = chatf.read()

chatdf = create_chat_df(chat)
senders = chatdf['sender'].unique()

game_res = {'Wordle'      : r'Wordle [\d,]+ [\dX]/\d\*?',
            'Connections' : r'Connections ?\|Puzzle #[\d,]+',
            'mini1'       : r'I solved the \d+/\d+/\d+ New York Times Mini Crossword in \d+:\d+!',
            'mini2'       : r'https://www.nytimes.com/badges/games/mini.html\?d=\d{4}-\d{2}-\d{2}&t=\d+'}
game_dfs:     dict = {}
result_dfs:   dict = {}
failed_games: dict = {}

for game, regex in game_res.items():
    game_dfs[game] = get_game_df(chatdf, regex)

for game, df in game_dfs.items():
    result_df, fail_df = get_results(game, df)
    result_dfs[game] = result_df
    failed_games[game] = fail_df

# concat mini1 and mini2 into Mini and del mini1/mini2

out_string = ''
for game, rdf in result_dfs.items():
    out_string += game + ':\n' + rdf.to_csv(encoding='utf8', lineterminator='\n') + '\n'
    out_string += failed_games[game].to_csv(encoding='utf8', lineterminator='\n', index=False) + '\n'

print(out_string)

with open(args.out_file, 'w', encoding='utf-8') as outf:
    outf.write(out_string)
    print('Results written to', args.out_file)