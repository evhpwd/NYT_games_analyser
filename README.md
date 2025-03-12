
## NYT Games Whatsapp Chat Analyser

This script takes an exported Whatsapp chat and extracts messages with NYT game results like Wordle, Connections and the Mini Crossword\
Bypass the need for a NYT login/subscription with this one neat trick 😎

### To Use:

Install Python https://www.python.org/downloads/

1. Download script by clicking '<> Code' and then 'Download ZIP'
2. Export your Whatsapp Chat
    1. Go to chat
    2. Go to group info from top of screen
    3. Scroll down and select 'Export chat'
3. Extract zipped chat on computer
4. Install libraries from command line with 'pip install -r requirements.txt'
5. Run script from command line with 'python3 analyse.py --chat-file [path/to/chat.txt]'

The results will be saved to default_out.csv by default, change this by specifying '--out-file'

### Other:

* I will make this more OO aligned in future (i.e. classes for games, methods, etc)
    * This should make it easier to add support for other games as desired
    * (And remove the wanky passing dictionaries and dataframes between functions)
* This was tested with my group chat on iOS